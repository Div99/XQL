from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), layer_norm=self.layer_norm, dropout_rate=self.dropout_rate)(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations,
                     layer_norm=self.layer_norm)(inputs)
        return jnp.squeeze(critic, -1)

    def grad_norm(self, obs, action, interpolate=False, lambda_=1):

        data = jnp.concatenate([obs, action], 1)
        if interpolate:
            expert_data = jnp.concatenate([obs1, action1], 1)
            policy_data = jnp.concatenate([obs2, action2], 1)

            # Interpolate between fake and real images with epsilon
            alpha = jax.random.uniform(key, shape=(expert_data.shape[0], 1))
            alpha = alpha.expand_as(expert_data).to(expert_data.device)
            data_mix = data * epsilon + fake_data * (1 - epsilon)

        # Fetch the gradient penalty
        gradients = critic_forward(params_c, vars_c, data_mix)
        gradients = gradients.reshape((gradients.shape[0], -1))

        alpha = torch.rand(expert_data.size()[0], 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)

        interpolated_state, interpolated_action = torch.split(
            interpolated, [self.obs_dim, self.action_dim], dim=1)
        q = self.forward(interpolated_state, interpolated_action)
        ones = torch.ones(q.size()).to(policy_data.device)
        gradient = grad(
            outputs=q,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = lambda_ * (jnp.linalg.norm(gradient, axis=1) - 1).pow(2).mean()
        return grad_pen


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations,
                         layer_norm=self.layer_norm)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations,
                         layer_norm=self.layer_norm)(observations, actions)
        return critic1, critic2
