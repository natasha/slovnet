
from .record import Record


class Batch(Record):
    __attributes__ = ['input', 'target']

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
