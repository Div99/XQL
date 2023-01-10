from typing import Tuple

import jax.numpy as jnp
import jax
from functools import partial

from common import Batch, InfoDict, Model, Params, PRNGKey


def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def loss_exp(diff, alpha=1.0, args=None):
    diff = diff
    x = diff/alpha
    loss_partial = grad_gumbel(x, alpha, args.max_clip)
    loss = jax.lax.stop_gradient(loss_partial) * x * diff.shape[0]
    return loss  # jnp.minimum(loss, 200)

def gumbel_rescale_loss(diff, alpha, args=None):
    z = diff/alpha
    if args.max_clip is not None:
        z = jnp.minimum(z, args.max_clip) # jnp.clip(x, a_min=-1000, a_max=args.max_clip)  
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = jnp.exp(z - max_z) - z*jnp.exp(-max_z) - jnp.exp(-max_z)
    return loss # jnp.minimum(loss.mean(), 200)


def grad_gumbel(x, alpha, clip_max=7):
    """Calculate grads of log gumbel_loss: (e^x - 1)/[(e^x - x - 1).mean() + 1]
    We add e^-a to both numerator and denominator to get: (e^(x-a) - e^(-a))/[(e^(x-a) - xe^(-a)).mean()]
    """
    # clip inputs to grad in [-10, 10] to improve stability (gradient clipping)
    x = jnp.minimum(x, clip_max)  # jnp.clip(x, a_min=-10, a_max=10)

    # calculate an offset `a` to prevent overflow issues
    x_max = jnp.max(x, axis=0)
    # choose `a` as max(x_max, -1) as its possible for x_max to be very small and we want the offset to be reasonable
    x_max = jnp.where(x_max < -1, -1, x_max)

    # keep track of original x
    x_orig = x
    # offsetted x
    x1 = x - x_max

    grad = (jnp.exp(x1) - jnp.exp(-x_max)) / \
        (jnp.mean(jnp.exp(x1) - x_orig * jnp.exp(-x_max), axis=0, keepdims=True))
    return grad


def loss_gumbel(diff, alpha=1.0):
    """ Gumbel loss J: E[e^x - x - 1]. We calculate the log of Gumbel loss for stability, i.e. Log(J + 1)
    log_gumbel_loss: log((e^x - x - 1).mean() + 1)
    """
    diff = diff
    x = diff/alpha
    grad = grad_gumbel(x, alpha)
    # use analytic gradients to improve stability
    loss = jax.lax.stop_gradient(grad) * x
    return loss


def update_v(critic: Model, value: Model, batch: Batch,
             expectile: float, loss_temp: float, double: bool, vanilla: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:
    actions = batch.actions

    rng1, rng2 = jax.random.split(key)
    if args.sample_random_times > 0:
        # add random actions to smooth loss computation (use 1/2(rho + Unif))
        times = args.sample_random_times
        random_action = jax.random.uniform(
            rng1, shape=(times * actions.shape[0],
                         actions.shape[1]),
            minval=-1.0, maxval=1.0)
        obs = jnp.concatenate([batch.observations, jnp.repeat(
            batch.observations, times, axis=0)], axis=0)
        acts = jnp.concatenate([batch.actions, random_action], axis=0)
    else:
        obs = batch.observations
        acts = batch.actions

    if args.noise:
        std = args.noise_std
        noise = jax.random.normal(rng2, shape=(acts.shape[0], acts.shape[1]))
        noise = jnp.clip(noise * std, -0.5, 0.5)
        acts = (batch.actions + noise)
        acts = jnp.clip(acts, -1, 1)

    q1, q2 = critic(obs, acts)
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)

        if vanilla:
            value_loss = loss(q - v, expectile).mean()
        else:
            if args.log_loss:
                value_loss = loss_exp(q - v, alpha=loss_temp, args=args).mean()
            else:
                value_loss = gumbel_rescale_loss(q - v, alpha=loss_temp, args=args).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info


def update_q(critic: Model, target_value: Model, batch: Batch,
             discount: float, double: bool, key: PRNGKey, loss_temp: float, args) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)

    target_q = batch.rewards + discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        v = target_value(batch.observations)

        def mse_loss(q, q_target, *args):
            loss_dict = {}

            x = q-q_target
            loss = huber_loss(x, delta=20.0)  # x**2
            loss_dict['critic_loss'] = loss.mean()

            return loss.mean(), loss_dict

        critic_loss = mse_loss

        if double:
            loss1, dict1 = critic_loss(q1, target_q, v, loss_temp)
            loss2, dict2 = critic_loss(q2, target_q, v, loss_temp)

            critic_loss = (loss1 + loss2).mean()
            for k, v in dict2.items():
                dict1[k] += v
            loss_dict = dict1
        else:
            # critic_loss, loss_dict = dual_q_loss(q1, target_q, v, loss_temp)
            critic_loss, loss_dict = critic_loss(q1, target_q,  v, loss_temp)

        if args.grad_pen:
            lambda_ = args.lambda_gp
            q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, acts)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()

            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()

            critic_loss += lambda_ * gp_loss

        loss_dict.update({
            'q1': q1.mean(),
            'q2': q2.mean()
        })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def grad_norm(model, params, obs, action, lambda_=10):

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.jacrev, argnums=1)
    def input_grad_fn(obs, action):
        return model.apply({'params': params}, obs, action)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad1, axis=-1) - 1, 0)**2
        return penalty

    grad1, grad2 = input_grad_fn(obs, action)

    return grad_pen_fn(grad1), grad_pen_fn(grad2)


def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear
