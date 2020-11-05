
from torch.nn import functional as F

from .mask import fill_masked


def flatten_cross_entropy(pred, target, ignore_id=None):
    """
    Flatten the entropy of a target.

    Args:
        pred: (todo): write your description
        target: (todo): write your description
        ignore_id: (str): write your description
    """
    target = target.flatten()
    pred = pred.view(len(target), -1)
    return F.cross_entropy(pred, target, ignore_index=ignore_id)


def masked_flatten_cross_entropy(pred, target, mask, ignore_id=-100):
    """
    Flatten masked mask.

    Args:
        pred: (todo): write your description
        target: (todo): write your description
        mask: (array): write your description
        ignore_id: (str): write your description
    """
    target = fill_masked(target, ~mask, ignore_id)
    return flatten_cross_entropy(pred, target, ignore_id)
