import torch
from torch.nn import functional as F

from .base import Processor

class RandomShiftsAug(Processor):

    def __init__(self, observation_space, action_space, pad=4):
        super().__init__(observation_space, action_space)
        self.pad = pad

    def _flatten(self, x):
        num_dims = len(x.shape)
        if num_dims == 5:
            # Compress it
            n, s, c, h, w = x.size()
            return x.view(n, s*c, h, w)
        elif num_dims == 4:
            return x
        else:
            raise ValueError("Input was the wrong shape")

    def _aug(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

    def __call__(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self._flatten(batch.float())
        elif isinstance(batch, dict):
            for k in batch.keys():
                if 'obs' in k:
                    batch[k] = self._flatten(batch[k].float())
        if not self.training:
            return batch
        else:
            print("AUGED")
            for k in batch.keys():
                if 'obs' in k:
                    batch[k] = self._aug(batch[k])
            return batch
