
from torch.nn import functional as F

from .mask import fill_masked


def flatten_cross_entropy(pred, target, ignore_id=None):
    target = target.flatten()
    pred = pred.view(len(target), -1)
    return F.cross_entropy(pred, target, ignore_index=ignore_id)


def masked_flatten_cross_entropy(pred, target, mask, ignore_id=-100):
    target = fill_masked(target, ~mask, ignore_id)
    return flatten_cross_entropy(pred, target, ignore_id)
