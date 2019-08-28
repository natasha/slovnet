
import torch

from .record import Record


class Batch(Record):
    __attributes__ = ['input', 'target']

    def __init__(self, input, target):
        self.input = input
        self.target = target

    def processed(self, loss, pred):
        return ProcessedBatch(
            self.input, self.target,
            loss, pred
        )

    def to(self, device):
        input = [_.to(device) for _ in self.input]
        target = self.target
        if target is not None:
            target = target.to(device)
        return Batch(input, target)

    @classmethod
    def from_markup_encoder(cls, ids):
        # ids  [(words, tags), (words, tags), ...]
        inputs, targets = zip(*ids)
        # inputs [(feat1, feat2, ...), (feat1, feat2, ...), ...]
        inputs = zip(*inputs)

        input = [torch.LongTensor(_) for _ in inputs]
        target = torch.LongTensor(targets)
        return cls(input, target)

    @classmethod
    def from_token_encoder(cls, ids):
        # ids  [feats1, feats2, ...]
        input = [torch.LongTensor([_]) for _ in ids]
        return cls(input, target=None)


class ProcessedBatch(Batch):
    __attributes__ = ['input', 'target', 'loss', 'pred']

    def __init__(self, input, target, loss, pred):
        Batch.__init__(self, input, target)
        self.loss = loss
        self.pred = pred
