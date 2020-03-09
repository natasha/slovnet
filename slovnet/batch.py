
from .record import Record


class Batch(Record):
    __attributes__ = ['input', 'target']

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def to(self, device):
        return Batch(
            self.input.to(device),
            self.target.to(device)
        )

    def processed(self, loss, pred):
        return ProcessedBatch(
            self.input, self.target,
            loss, pred
        )


class ProcessedBatch(Record):
    __attributes__ = ['input', 'target', 'loss', 'pred']

    def __init__(self, input, target, loss, pred):
        self.input = input
        self.target = target
        self.loss = loss
        self.pred = pred
