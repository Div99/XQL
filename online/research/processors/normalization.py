from multiprocessing.sharedctypes import Value
import torch
import numpy as np
import gym

from .base import Processor
from research.utils import utils

class RunningMeanStd(object):

    def __init__(self, shape, dtype, epsilon=1e-6):
        self._mean = np.zeros(shape, dtype=dtype)
        self._m2 = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        self.count += 1
        delta = (x - self._mean)
        self._mean = self._mean +  delta / self.count
        # Update the second moment
        self._m2 = self._m2 + delta*(x - self._mean)
    
    @property
    def mean(self):
        return self._mean.astype(np.float32)

    @property
    def var(self):
        return (self._m2 / self.count).astype(np.float32)
    
    @property
    def std(self):
        return np.sqrt(self.var)

class RunningObservationNormalizer(Processor):
    '''
    A running observation normalizer. 
    Note that there are quite a few speed optimizations that could be performed:
    1. We could cache computation of the variance etc. so it doesn't run everytime.
    2. We could permanently store torch tensors so we don't recompute them and sync to GPU.
    '''

    def __init__(self, observation_space, action_space, epsilon=1e-7, clip=10):
        super().__init__(observation_space, action_space)
        assert isinstance(observation_space, gym.spaces.Box), "Currently only supports box spaces."
        self.rms = RunningMeanStd(observation_space.shape, observation_space.dtype, epsilon)
        self._updated_stats = True
        self.clip = clip

    def update(self, batch):
        if isinstance(batch, np.ndarray):
            self.rms.update(batch)
        elif isinstance(batch, dict):
            assert "obs" in batch, "Called ObsNormalizer with batch that did not contain 'obs' key."
            self.rms.update(batch["obs"])
        else:
            raise ValueError("Invalid input provided.")
        self._updated_stats = True
    
    def __call__(self, batch):
        if self._updated_stats:
            # Update the tensors
            device = utils.get_device(batch)
            if device is not None:
                self._mean_tensor = torch.from_numpy(self.rms.mean).to(device)
                self._std_tensor = torch.from_numpy(self.rms.std).to(device)
            self._updated_stats = False
        # Normalize by the input type
        if isinstance(batch, dict):
            for k in ('obs', 'next_obs'):
                if k in batch:
                    batch[k] = self(batch[k])
            return batch
        if isinstance(batch, torch.Tensor):
            batch = (batch - self._mean_tensor) / self._std_tensor
            return torch.clamp(batch, -self.clip, self.clip) if self.clip is not None else batch
        elif isinstance(batch, np.ndarray):
            batch = (batch - self.rms.mean) / self.rms.std
            return np.clip(batch, -self.clip, self.clip) if self.clip is not None else batch
        else:
            raise ValueError("Invalid input provided to ObservationNormalizer")
