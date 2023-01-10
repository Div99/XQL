from torch import nn
import research

class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space, action_space, 
                       actor_class, critic_class, encoder_class=None, 
                       actor_kwargs={}, critic_kwargs={}, encoder_kwargs={}, **kwargs) -> None:
        super().__init__()
        # Update all dictionaries with the generic kwargs
        self.action_space = action_space
        self.observation_space = observation_space

        encoder_kwargs.update(kwargs)
        actor_kwargs.update(kwargs)
        critic_kwargs.update(kwargs)

        self.encoder_class, self.encoder_kwargs = encoder_class, encoder_kwargs
        self.actor_class, self.actor_kwargs = actor_class, actor_kwargs
        self.critic_class, self.critic_kwargs = critic_class, critic_kwargs

        self.reset_encoder()
        self.reset_actor()
        self.reset_critic()

    def reset_encoder(self):
        encoder_class = vars(research.networks)[self.encoder_class] if isinstance(self.encoder_class, str) else self.encoder_class
        if encoder_class is not None:
            self._encoder = encoder_class(self.observation_space, self.action_space, **self.encoder_kwargs)
        else:
            self._encoder = nn.Identity()

    def reset_actor(self):
        observation_space = self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        actor_class = vars(research.networks)[self.actor_class] if isinstance(self.actor_class, str) else self.actor_class
        self._actor = actor_class(observation_space, self.action_space, **self.actor_kwargs)

    def reset_critic(self):
        observation_space = self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        critic_class = vars(research.networks)[self.critic_class] if isinstance(self.critic_class, str) else self.critic_class
        self._critic = critic_class(observation_space, self.action_space, **self.critic_kwargs)

    @property
    def actor(self):
        return self._actor
    
    @property
    def critic(self):
        return self._critic

    @property
    def encoder(self):
        return self._encoder
        
    def predict(self, obs, **kwargs):
        obs = self._encoder(obs)
        if hasattr(self._actor, "predict"):
            return self._actor.predict(obs, **kwargs)
        else:
            return self._actor(obs)

class ActorCriticValuePolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, actor_class, critic_class, value_class, 
                       encoder_class=None, actor_kwargs={}, critic_kwargs={}, value_kwargs={}, encoder_kwargs={}, **kwargs) -> None:
        super().__init__(observation_space, action_space, actor_class, critic_class, encoder_class, actor_kwargs, critic_kwargs, encoder_kwargs, **kwargs)
        value_kwargs.update(kwargs)
        self.value_class, self.value_kwargs = value_class, value_kwargs
        self.reset_value()

    def reset_value(self):
        observation_space = self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        value_class = vars(research.networks)[self.value_class] if isinstance(self.value_class, str) else self.value_class
        self._value = value_class(observation_space, self.action_space, **self.value_kwargs)
    
    @property
    def value(self):
        return self._value

class ActorCriticDensityPolicy(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, actor_class, critic_class, density_class, 
                       encoder_class=None, actor_kwargs={}, critic_kwargs={}, density_kwargs={}, encoder_kwargs={}, **kwargs) -> None:
        super().__init__(observation_space, action_space, actor_class, critic_class, encoder_class, actor_kwargs, critic_kwargs, encoder_kwargs, **kwargs)
        density_kwargs.update(kwargs)
        self.density_class, self.density_kwargs = density_class, density_kwargs
        self.reset_density()

    def reset_density(self):
        observation_space = self.encoder.output_space if hasattr(self.encoder, "output_space") else self.observation_space
        density_class = vars(research.networks)[self.density_class] if isinstance(self.density_class, str) else self.density_class
        self._density = density_class(observation_space, self.action_space, **self.density_kwargs)
    
    @property
    def density(self):
        return self._density
