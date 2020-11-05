
import torch

from .record import Record
from .pad import pad_sequence


class Masked(Record):
    __attributes__ = ['value', 'mask']


def mask_like(input):
    """
    Mask the given array of elements.

    Args:
        input: (array): write your description
    """
    return torch.ones_like(input, dtype=torch.bool)


def split_masked(input, mask):
    """
    Split the given mask into two - 1dict.

    Args:
        input: (array): write your description
        mask: (array): write your description
    """
    sizes = mask.sum(dim=-1).tolist()
    return input[mask].split(sizes)


def pad_masked(input, mask, fill=0):
    """
    Pad a mask with padding.

    Args:
        input: (array): write your description
        mask: (array): write your description
        fill: (str): write your description
    """
    seqs = split_masked(input, mask)
    return pad_sequence(seqs, fill)


def fill_masked(input, mask, fill=0):
    """
    Fill nan.

    Args:
        input: (array): write your description
        mask: (array): write your description
        fill: (str): write your description
    """
    return fill * mask + input * ~mask
