
from .record import Record
from .pad import pad_sequence


class Masked(Record):
    __attributes__ = ['value', 'mask']


def split_masked(input, mask):
    sizes = mask.sum(dim=-1).tolist()
    return input[mask].split(sizes)


def pad_masked(input, mask, fill=0):
    seqs = split_masked(input, mask)
    return pad_sequence(seqs, fill)
