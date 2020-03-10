
from torch.nn import functional as F


def flatten_cross_entropy(pred, target):
    target = target.flatten()
    pred = pred.view(len(target), -1)
    return F.cross_entropy(pred, target)
