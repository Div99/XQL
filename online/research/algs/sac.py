import torch
import numpy as np
import itertools
import os

from .base import Algorithm
from research.networks.base import ActorCriticPolicy

class SAC(Algorithm):

    def __init__(self, env, network_class, dataset_class, 
                       tau=0.005,
                       init_temperature=0.1,
                       env_freq=1,
                       critic_freq=1,
                       actor_freq=2,
                       target_freq=2,
                       init_steps=1000,
                       **kwargs):

        # Save values needed for optim setup.
        self.init_temperature = init_temperature
        super().__init__(env, network_class, dataset_class, **kwargs)
        assert isinstance(self.network, ActorCriticPolicy)

        # Save extra parameters
        self.tau = tau
        self.env_freq = env_freq
        self.critic_freq = critic_freq
        self.actor_freq = actor_freq
        self.target_freq = target_freq
        self.init_steps = init_steps
        import collections
        self._metric_storage = collections.deque(maxlen=10)

    @property
    def alpha(self):
        return self.log_alpha.exp()

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

        # Setup the learned entropy coefficients. This has to be done first so its present in the setup_optim call.
        self.log_alpha = torch.tensor(np.log(self.init_temperature), dtype=torch.float).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(self.env.action_space.low.shape)

        self.optim['log_alpha'] = optim_class([self.log_alpha], **optim_kwargs)

    def _update_critic(self, batch):
        with torch.no_grad():
            dist = self.network.actor(batch['next_obs'])
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(dim=-1)
            target_qs = self.target_network.critic(batch['next_obs'], next_action)
            target_v = torch.min(target_qs, dim=0)[0] - self.alpha.detach() * log_prob
            target_q = batch['reward'] + batch['discount']*target_v

        qs = self.network.critic(batch['obs'], batch['action'])
        # Note: Could also just compute the mean over a broadcasted target. TO investigate later.
        q_loss = sum([torch.nn.functional.mse_loss(qs[i], target_q) for i in range(qs.shape[0])])

        self.optim['critic'].zero_grad(set_to_none=True)
        q_loss.backward()
        self.optim['critic'].step()

        # Compute the bellman errors to save
        if self.steps % 100 == 0:
            with torch.no_grad():
                next_action = self.network.actor(batch['next_obs'].detach()).loc
                target = batch['reward'] + batch['discount']*self.network.critic(batch['next_obs'], next_action)[0] # Get only Q1
                error = target - qs[0]
                error = error.cpu().numpy()
            self._metric_storage.append(error)

        return dict(q_loss=q_loss.item(), target_q=target_q.mean().item())
    
    def _update_actor_and_alpha(self, batch):
        obs = batch['obs'].detach() # Detach the encoder so it isn't updated.
        dist = self.network.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        qs = self.network.critic(obs, action)
        q = torch.min(qs, dim=0)[0]
        actor_loss = (self.alpha.detach() * log_prob - q).mean()

        self.optim['actor'].zero_grad(set_to_none=True)
        actor_loss.backward()
        self.optim['actor'].step()
        entropy = -log_prob.mean()

        # Update the learned temperature
        self.optim['log_alpha'].zero_grad(set_to_none=True)
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.optim['log_alpha'].step()

        return dict(actor_loss=actor_loss.item(), entropy=entropy.item(), 
                    alpha_loss=alpha_loss.item(), alpha=self.alpha.detach().item())

    def _step_env(self):
        # Step the environment and store the transition data.
        metrics = dict()
        if self._env_steps < self.init_steps:
            action = self.env.action_space.sample()
        else:
            self.eval_mode()
            with torch.no_grad():
                action = self.predict(self._current_obs, sample=True)
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

        # Store the consequences.
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
            metrics = self._update_actor_and_alpha(batch)
            all_metrics.update(metrics)

        if self.steps % self.target_freq == 0:
            # Only update the critic and encoder for speed. Ignore the actor.
            with torch.no_grad():
                for param, target_param in zip(self.network.encoder.parameters(), self.target_network.encoder.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.critic.parameters(), self.target_network.critic.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return all_metrics

    def _validation_step(self, batch):
        raise NotImplementedError("RL Algorithm does not have a validation dataset.")

    def _validation_extras(self, path, step, dataloader):
        if self.steps < 2*self.init_steps:
            return {}
        arr = np.array(self._metric_storage).flatten()
        save_path = os.path.join(path, "errors_" + str(step))
        np.save(save_path, arr)
        return {}
