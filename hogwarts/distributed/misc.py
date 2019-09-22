__all__ = ['get_world_size', 'get_rank', 'get_backend', 'barrier', 'all_reduce_sum',
           'all_reduce_max', 'all_reduce_min', 'broadcast', 'all_gather_cat', 'dist_segment',
           'dist_init', 'get_host_ip']

import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing


def _check_tensor_list(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')


def get_host_ip():
    import socket
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


def barrier():
    sync_tensor = torch.zeros(1)
    if torch.cuda.is_available():
        sync_tensor = sync_tensor.cuda()
    dist.all_reduce(sync_tensor)
    _ = sync_tensor.item()


def all_reduce_sum(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def all_reduce_max(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)


def all_reduce_min(tensor_list):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        tensor.neg_()


def broadcast(tensor_list, src):
    _check_tensor_list(tensor_list)
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


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


def dist_init(local_rank, backend='nccl', mp_method='fork'):
    if multiprocessing.get_start_method(allow_none=True) != mp_method:
        multiprocessing.set_start_method(mp_method, force=True)
    rank, world_size = int(local_rank), int(os.environ['WORLD_SIZE'])
    os.environ['OMPI_COMM_WORLD_SIZE'] = str(world_size)
    os.environ['OMPI_COMM_WORLD_RANK'] = str(rank)
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    return rank, world_size
