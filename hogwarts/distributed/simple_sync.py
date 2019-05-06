__all__ = ['sync_state', 'sync_grad_mean', 'sync_grad_sum', 'sync_bn_stat']

import torch
from . import misc


@misc.MultiprocessingOnly(inplace=True)
def sync_state(network, src=0):
    tensor_list = list(network.state_dict().values())
    if src == 'all':
        misc.all_reduce_mean(tensor_list)
    else:
        misc.broadcast(tensor_list, src)


@misc.MultiprocessingOnly(inplace=True)
def sync_grad_mean(network):
    misc.all_reduce_mean([param.grad.data for param in network.parameters() if param.grad is not None])


@misc.MultiprocessingOnly(inplace=True)
def sync_grad_sum(network):
    misc.all_reduce_sum([param.grad.data for param in network.parameters() if param.grad is not None])


@misc.MultiprocessingOnly(inplace=True)
def sync_bn_stat(network):
    tensor_list = []
    for mod in network.modules():
        if 'Norm' in mod.__class__.__name__:
            tensor_list.append(mod.running_mean)
            tensor_list.append(mod.running_var)
    if len(tensor_list) > 0:
        misc.all_reduce_mean(tensor_list)
