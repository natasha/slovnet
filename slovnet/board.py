
from os.path import join as join_path
from collections import Counter

from torch.utils.tensorboard import SummaryWriter

from .record import Record
from .bio import PER, LOC, ORG


class Board(Record):
    __attributes__ = ['dir', 'root']

    def __init__(self, dir, root, flush_secs=1):
        self.dir = dir
        self.root = root
        self.writer = SummaryWriter(
            join_path(root, dir),
            flush_secs=flush_secs
        )
        self.steps = Counter()

    def prefixed(self, prefix):
        return PrefixedBoard(prefix, self)

    def step(self, key):
        self.steps[key] += 1
        return self.steps[key]

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value, self.step(key))

    def add_hist(self, key, values):
        self.writer.add_histogram(key, values, self.step(key))

    def add_batch_score(self, score, types=[PER, LOC, ORG]):
        self.add_scalar('01_loss', score.loss)
        for index, type in enumerate(types, 2):
            key = '%02d_%s' % (index, type)
            value = score.get(type)
            if value:
                self.add_scalar(key, value.f1)

    def add_batch_scores(self, scores):
        for score in scores:
            self.add_batch_score(score)


class PrefixedBoard(Board):
    __attributes__ = ['prefix', 'board']

    def __init__(self, prefix, board):
        self.prefix = prefix
        self.board = board

    def prefixed(self, key):
        return '%s/%s' % (self.prefix, key)

    def add_scalar(self, key, value):
        self.board.add_scalar(self.prefixed(key), value)

    def add_hist(self, key, values):
        self.board.add_hist(self.prefixed(key), values)
