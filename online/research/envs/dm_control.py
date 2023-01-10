import gym
from dm_control import suite
import dm_env
from dm_env import specs
import numpy as np
from gym import spaces
import collections
from dm_control.rl.control import FLAT_OBSERVATION_KEY
from dm_control.suite.wrappers import action_scale, pixels

'''
Code in this file was largely borrowed from Denis Yarats. Here are the appropriate methods
DRQv2: https://github.com/facebookresearch/drqv2/blob/main/dmc.py

Note, however, that parts of it have been changed to afford a greater level of flexibility.
'''

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)

class FlattenWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

        original_spec = self._env.observation_spec()
        dtype = original_spec[next(iter(original_spec))].dtype
        # Check the spec and dtype
        assert all([v.dtype is dtype for v in original_spec.values()])
        assert all([isinstance(v, specs.Array) for v in original_spec.values()])
        # Combine the spaces
        num_elem = sum([np.prod(v.shape) for v in original_spec.values()])
        self._obs_spec = collections.OrderedDict()
        self._obs_spec[FLAT_OBSERVATION_KEY] = specs.Array([num_elem], dtype, name=FLAT_OBSERVATION_KEY)
        
    def _transform_time_step(self, time_step):
        observation = time_step.observation
        if isinstance(observation, collections.OrderedDict):
            keys = observation.keys()
        else:
            # Keep a consistent ordering for other mappings.
            keys = sorted(observation.keys())
        observation_arrays = [np.array([observation[k]]) if np.isscalar(observation[k]) else observation[k].ravel() for k in keys]
        observation = type(observation)([(FLAT_OBSERVATION_KEY, np.concatenate(observation_arrays))])
        return time_step._replace(observation=observation)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_time_step(time_step)
        
    def observation_spec(self):
        return self._obs_spec
        
    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._transform_time_step(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)

class ChannelsFirstWrapper(dm_env.Environment):
    '''
    Warning: This wrapper does an inplace operation on some of the arrays
    I think its fine (and better for mem), but there could be issues later.
    '''
    def __init__(self, env, pixels_key):
        self._env = env
        self._pixels_key = pixels_key

        observation_spec = self._env.observation_spec().copy()
        assert self._pixels_key in observation_spec
        pixel_spec = observation_spec[self._pixels_key]
        # Move the last dim to the front
        assert isinstance(pixel_spec, specs.Array)
        assert len(pixel_spec.shape) == 3
        h, w, c = pixel_spec.shape
        observation_spec[self._pixels_key] = specs.Array((c, h, w), dtype=pixel_spec.dtype, name=pixel_spec.name)
        self._obs_spec = observation_spec
        
    def _transform_time_step(self, time_step):
        observation = time_step.observation
        observation[self._pixels_key] = observation[self._pixels_key].transpose(2, 0, 1).copy()
        return time_step._replace(observation=observation)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_time_step(time_step)
        
    def observation_spec(self):
        return self._obs_spec
        
    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        time_step = self._env.reset()
        return self._transform_time_step(time_step)

    def __getattr__(self, name):
        return getattr(self._env, name)

class StackWrapper(dm_env.Environment):

    def __init__(self, env, stack):
        self._env = env
        self._stack = stack
        
        self._queues = collections.OrderedDict()
        self._obs_spec = collections.OrderedDict()
        for k, v in env.observation_spec().items():
            assert isinstance(v, specs.Array), "Observation conversion does not support bounded specs."
            # Add a temporal axis to each shape.
            new_shape = np.concatenate([[stack], v.shape], axis=0)
            self._obs_spec[k] = specs.Array(new_shape, dtype=v.dtype, name=v.name)
            self._queues[k] = collections.deque([], maxlen=stack)
    
    def _transform_time_step(self, time_step):
        new_observation = collections.OrderedDict()
        for k in time_step.observation.keys():
            new_observation[k] = np.stack(self._queues[k], axis=0)
        return time_step._replace(observation=new_observation)

    def reset(self):
        time_step = self._env.reset()
        # Add the obs to the deques, duplicating for multiple time steps.
        for k, v in time_step.observation.items():
            for _ in range(self._stack):
                self._queues[k].append(v)
        return self._transform_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        # Add the observations to the deques
        for k, v in time_step.observation.items():
            self._queues[k].append(v)
        return self._transform_time_step(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

def _spec_to_space(spec):
    '''
    In DM Control, observation specs are always dictionariers (usually OrderedDict)
    That contain spec.Arrray or spec.BoundedArray. 
    Action specs are just spec.BoundedArray
    '''
    if isinstance(spec, specs.BoundedArray):
        # Run the conversion
        dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
        zeros = np.zeros(spec.shape, dtype=dtype)
        return spaces.Box(spec.minimum + zeros, spec.maximum + zeros, dtype=dtype)
    elif isinstance(spec, specs.Array):
        # Run the conversion
        dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
        bound = np.inf * np.ones(spec.shape, dtype=dtype)
        return spaces.Box(-bound, bound, dtype=dtype)
    elif isinstance(spec, dict) or isinstance(spec, collections.OrderedDict):
        dict_of_spaces = collections.OrderedDict()
        for k, v in spec.items():
            dict_of_spaces[k] = _spec_to_space(v)
        return spaces.Dict(spaces=dict_of_spaces)
    else:
        raise ValueError("Invalid spec encountered.")

class DMControlEnv(gym.Env):

    def __init__(self, domain_name, task_name, 
                       task_kwargs=None,
                       environment_kwargs=None,
                       visualize_reward=False,
                       action_dtype=np.float32,
                       action_repeat=1,
                       action_minimum=None,
                       action_maximum=None,
                       from_pixels=False,
                       height=84,
                       width=84,
                       camera_id=0,
                       channels_first=True,
                       flatten=False,
                       stack=1):
                       
        self._pixels_key = 'pixels'
        self._height = height
        self._width = width
        self._camera_id = camera_id

        env = suite.load(domain_name, task_name,
                               task_kwargs=task_kwargs,
                               environment_kwargs=environment_kwargs,
                               visualize_reward=visualize_reward)

        env = ActionDTypeWrapper(env, action_dtype)
        if action_repeat > 1:
            env = ActionRepeatWrapper(env, action_repeat)
        if action_minimum is not None and action_maximum is not None:
            env = action_scale.Wrapper(env, minimum=action_minimum, maximum=action_maximum)
        if from_pixels:
            render_kwargs = dict(height=height, width=width, camera_id=camera_id)
            env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs, observation_key=self._pixels_key)
            if channels_first:
                env = ChannelsFirstWrapper(env, pixels_key=self._pixels_key)
        if flatten:
            env = FlattenWrapper(env)
        if stack > 1:
            env = StackWrapper(env, stack)
        
        self._env = env

        # Create the equivalent gym spaces
        obs_spec = self._env.observation_spec()
        if (isinstance(obs_spec, dict) or isinstance(obs_spec, collections.OrderedDict)) and len(obs_spec) == 1:
            self._unwrap_obs = True
            obs_spec = obs_spec[next(iter(obs_spec))]
        else:
            self._unwrap_obs = False
        self._observation_space = _spec_to_space(obs_spec)
        self._action_space = _spec_to_space(self._env.action_spec())

        # Seed the environment
        if task_kwargs is None:
            task_kwargs = {}
        self.seed(seed=task_kwargs.get('random', 1))

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def seed(self, seed):
        self._action_space.seed(seed)
        self._observation_space.seed(seed)

    def _extract_obs(self, time_step):
        obs = time_step.observation
        for k in obs.keys():
            if obs[k].dtype == np.float64:
                obs[k] = obs[k].astype(np.float32)
        if self._unwrap_obs:
            return obs[next(iter(obs))]
        else:
            return obs

    def step(self, action):
        time_step = self._env.step(action)
        info = {'discount': time_step.discount}
        if time_step.discount == 0.0:
            info['early_termination'] = 1.0
        else:
            info['early_termination'] = 0.0
        return self._extract_obs(time_step), time_step.reward, time_step.last(), info

    def reset(self):
        time_step = self._env.reset()
        return self._extract_obs(time_step)
        
    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
