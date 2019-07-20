__all__ = ['DistributedSampler']

import math
import torch
from torch.utils.data.sampler import Sampler
from ... import distributed as dist


class DistributedSampler(Sampler):

    def __init__(self, dataset, shuffle=False, pseudo_index=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.pseudo_index = pseudo_index

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_samples = int(math.ceil(len(self.dataset) / self.world_size))
        self.total_size = self.num_samples * self.world_size

        if self.shuffle:
            self.g = torch.Generator()
            state = torch.get_rng_state()
            # synchronize local random state
            backend = dist.get_backend()
            if backend != 'nccl':
                raise RuntimeError('only support nccl backend currently')
            state = state.cuda()
            dist.broadcast([state], 0)
            state = state.cpu()
            self.g.set_state(state)

    def __iter__(self):
        if self.shuffle:
            # shuffle based on (already synchronized) local random generator
            indices = list(torch.randperm(len(self.dataset), generator=self.g))
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        num_extra = self.total_size - len(indices)
        if self.pseudo_index is None:
            indices += indices[:num_extra]
        else:
            indices += [self.pseudo_index] * num_extra
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
