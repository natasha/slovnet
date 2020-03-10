
from .record import Record
from .pad import pad_sequence


class Masked(Record):
    __attributes__ = ['value', 'mask']

    def __init__(self, value, mask):
        self.value = value
        self.mask = mask

    def to(self, device):
        return Masked(
            self.value.to(device),
            self.mask.to(device)
        )


def split_masked(input, mask):
    sizes = mask.sum(dim=-1).tolist()
    return input[mask].split(sizes)


def pad_masked(input, mask, fill=0):
    seqs = split_masked(input, mask)
    return pad_sequence(seqs, fill)
