
from itertools import islice

from nerus.load import load_norm

from .record import Record
from .markup import SpanMarkup


class Dataset(Record):
    __attributes__ = ['path']

    def __init__(self, path):
        self.path = path

    def slice(self, start, stop):
        return SliceDataset(self, start, stop)

    def __iter__(self):
        raise NotImplementedError


class NerusDataset(Dataset):
    def __iter__(self):
        for record in load_norm(self.path):
            yield SpanMarkup(record.text, record.spans)


class SliceDataset(Dataset):
    __attributes__ = ['records', 'start', 'stop']

    def __init__(self, records, start, stop):
        self.records = records
        self.start = start
        self.stop = stop

    def __iter__(self):
        return islice(self.records, self.start, self.stop)
