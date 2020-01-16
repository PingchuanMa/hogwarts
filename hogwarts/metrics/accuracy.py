__all__ = ['topk_accuracies', 'topk_accuracy', 'TopkAccuracy']


import torch


def topk_accuracies(output, label, ks=(1,)):
    assert output.dim() == 2
    assert label.dim() == 1
    assert output.size(0) == label.size(0)

    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    label = label.unsqueeze(1).expand_as(pred)
    correct = pred.eq(label).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1).mean()
        accu_list.append(accu.item())
    return accu_list


def topk_accuracy(output, label, k):
    return topk_accuracies(output, label, (k,))[0]


class TopkAccuracy(torch.nn.Module):

    def __init__(self, k):
        super(TopkAccuracy, self).__init__()
        self.k = k

    def forward(self, output, label):
        return topk_accuracy(output, label, self.k)
