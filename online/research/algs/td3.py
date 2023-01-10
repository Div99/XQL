import torch
import numpy as np
import itertools

from .base import Algorithm
from research.networks.base import ActorCriticPolicy
from research.utils.utils import to_tensor, to_device


class TD3(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       policy_noise=0.1,
                       target_noise=0.2,
                       noise_clip=0.5,
                       env_freq=1,
                       critic_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       init_steps=1000,
                       **kwargs):
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)
        # Save extra parameters
        self.tau = tau
        self.policy_noise = policy_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.env_freq = env_freq
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.action_range = (self.env.action_space.low, self.env.action_space.high)
        self.action_range_tensor = to_device(to_tensor(self.action_range), self.device)
        self.init_steps = init_steps

    def setup_network(self, network_class, network_kwargs):
        self.network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network = network_class(self.env.observation_space, self.env.action_space, 
                                     **network_kwargs).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self, optim_class, optim_kwargs):
        # Default optimizer initialization
        self.optim['actor'] = optim_class(self.network.actor.parameters(), **optim_kwargs)
        # Update the encoder with the critic.
        critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())        
        self.optim['critic'] = optim_class(critic_params, **optim_kwargs)

    def _update_critic(self, batch):
        with torch.no_grad():
            noise = (torch.randn_like(batch['action']) * self.target_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = self.target_network.actor(batch['next_obs'])
            noisy_next_action = (next_action + noise).clamp(*self.action_range_tensor)
            target_q = self.target_network.critic(batch['next_obs'], noisy_next_action)
            target_q = torch.min(target_q, dim=0)[0]
            target_q = batch['reward'] + batch['discount']*target_q

        qs = self.network.critic(batch['obs'], batch['action'])
        # Note: Could also just compute the mean over a broadcasted target. TO investigate later.
        q_loss = sum([torch.nn.functional.mse_loss(qs[i], target_q) for i in range(qs.shape[0])])
        
        self.optim['critic'].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim['critic'].step()

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())
    
    def _update_actor(self, batch):
        obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
        action = self.network.actor(obs)
        qs = self.network.critic(obs, action)
        q = qs[0] # Take only the first Q function
        actor_loss = -q.mean()

        self.optim['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim['actor'].step()

        return dict(actor_loss=actor_loss.item())

    def _step_env(self):
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(self._current_obs)
            action += self.policy_noise * np.random.randn(action.shape[0])
            self.train_mode()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        next_obs, reward, done, info = self.env.step(action)
        self._episode_length += 1
        self._episode_reward += reward

        if 'discount' in info:
            discount = info['discount']
        elif hasattr(self.env, "_max_episode_steps") and self._episode_length == self.env._max_episode_steps:
            discount = 1.0
        else:
            discount = 1 - float(done)

        # Store the consequences
        self.dataset.add(next_obs, action, reward, done, discount)
        
        if done:
            self._num_ep += 1
            # update metrics
            metrics['reward'] = self._episode_reward
            metrics['length'] = self._episode_length
            metrics['num_ep'] = self._num_ep
            # Reset the environment
            self._current_obs = self.env.reset()
            self.dataset.add(self._current_obs) # Add the first timestep
            self._episode_length = 0
            self._episode_reward = 0
        else:
            self._current_obs = next_obs

        self._env_steps += 1
        metrics['env_steps'] = self._env_steps
        return metrics

    def _setup_train(self):
        # Now setup the logging parameters
        self._current_obs = self.env.reset()
        self._episode_reward = 0
        self._episode_length = 0
        self._num_ep = 0
        self._env_steps = 0
        self.dataset.add(self._current_obs) # Store the initial reset observation!

    def _train_step(self, batch):
        all_metrics = {}

        if self.steps % self.env_freq == 0 or self._env_steps < self.init_steps:
            # step the environment with freq env_freq or if we are before learning starts
            metrics = self._step_env()
            all_metrics.update(metrics)
            if self._env_steps < self.init_steps:
                return all_metrics # return here.
        
        if 'obs' not in batch:
            return all_metrics

        updating_critic = self.steps % self.critic_freq == 0
        updating_actor = self.steps % self.actor_freq == 0

        if updating_actor or updating_critic:
            batch['obs'] = self.network.encoder(batch['obs'])
            with torch.no_grad():
                batch['next_obs'] = self.target_network.encoder(batch['next_obs'])
        
        if updating_critic:
            metrics = self._update_critic(batch)
            all_metrics.update(metrics)

        if updating_actor:
            metrics = self._update_actor(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")
