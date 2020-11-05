
from .record import Record


class Batch(Record):
    __attributes__ = ['input', 'target']

    def processed(self, loss, pred):
        """
        Perform a single loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            pred: (todo): write your description
        """
        return ProcessedBatch(
            self.input, self.target,
            loss, pred
        )


class ProcessedBatch(Record):
    __attributes__ = ['input', 'target', 'loss', 'pred']
