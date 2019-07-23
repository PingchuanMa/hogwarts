__all__ = ['get_world_size', 'get_rank', 'get_backend', 'barrier', 'all_reduce_sum',
           'all_reduce_max', 'all_reduce_min', 'broadcast', 'all_gather_cat', 'dist_segment',
           'dist_init', 'get_device', 'set_device', 'torch_dist_init']

import os
import math
import functools
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing


_DEVICE = torch.device('cpu')


def get_device():
    global _DEVICE
    return _DEVICE


def set_device(device):
    global _DEVICE
    _DEVICE = device


def _check_tensor_list(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')


def _get_host_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    return ip


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
def all_reduce_sum(tensor_list):
    _check_tensor_list(tensor_list)
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


@MultiprocessingOnly(inplace=True)
def all_reduce_max(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)


@MultiprocessingOnly(inplace=True)
def all_reduce_min(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
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


def dist_init(cuda=True, port=11442, backend='nccl', mp_method='forkserver'):
    if not cuda and backend == 'nccl':
        raise ValueError('nccl backend cannot be used without cuda')
    os.environ['DISTRIBUTED_BACKEND'] = backend
    if multiprocessing.get_start_method(allow_none=True) != mp_method:
        multiprocessing.set_start_method(mp_method, force=True)
    rank = get_rank()
    world_size = get_world_size()
    if cuda and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = rank % num_gpus
        device = torch.device('cuda', gpu_id)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    set_device(device)

    if world_size == 1:
        rank, world_size = 0, 1
    else:
        os.environ['MASTER_PORT'] = str(port)
        os.environ['MASTER_ADDR'] = str(_get_host_ip())
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        dist.init_process_group(backend=backend)

    return rank, world_size


def torch_dist_init(local_rank, backend='nccl', mp_method='fork'):
    if multiprocessing.get_start_method(allow_none=True) != mp_method:
        multiprocessing.set_start_method(mp_method, force=True)
    rank, world_size = local_rank, os.environ['WORLD_SIZE']
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    return rank, world_size
