__all__ = ['get_world_size', 'get_rank', 'get_backend', 'barrier', 'all_reduce_mean', 'all_reduce_sum',
           'all_reduce_max', 'all_reduce_min', 'broadcast', 'all_gather_cat', 'dist_segment', 'dist_init']

import os
import math
import torch
import multiprocessing
import functools
import torch.distributed as dist


def _check_tensor_list(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')


def get_world_size():
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))


def get_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)


class MultiprocessingOnly(object):
    def __init__(self, inplace=True):
        self.inplace = inplace

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if get_world_size() == 1:
                return None if self.inplace else args[0]
            return func(*args, **kwargs)
        return wrapped_func


# fake barrier
@MultiprocessingOnly(inplace=True)
def barrier():
    sync_tensor = torch.zeros(1)
    if torch.cuda.is_available():
        sync_tensor = sync_tensor.cuda()
    dist.all_reduce(sync_tensor)
    _ = sync_tensor.item()


@MultiprocessingOnly(inplace=True)
def all_reduce_mean(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)
        tensor.div_(get_world_size())


@MultiprocessingOnly(inplace=True)
def all_reduce_sum(tensor_list):
    _check_tensor_list(tensor_list)
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)


@MultiprocessingOnly(inplace=True)
def all_reduce_max(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.MAX)


@MultiprocessingOnly(inplace=True)
def all_reduce_min(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=dist.reduce_op.MAX)
        tensor.neg_()


@MultiprocessingOnly(inplace=True)
def broadcast(tensor_list, src):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


@MultiprocessingOnly(inplace=False)
def all_gather_cat(tensor_list, dim=0):
    _check_tensor_list(tensor_list)
    world_size = get_world_size()
    result_list = []
    for tensor in tensor_list:
        gather_list = [tensor.new(tensor.size()) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        result_list.append(torch.cat(gather_list, dim=dim))
    return result_list


def dist_segment(full_size, world_size=None, rank=None):
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
    interval = math.ceil(full_size / world_size)
    offset = interval * rank
    part_size = min(full_size, offset + interval) - offset
    return offset, part_size


def dist_init(backend):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork', force=True)
    rank = get_rank()
    world_size = get_world_size()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        dist.init_process_group(backend=backend)

    return rank, world_size
