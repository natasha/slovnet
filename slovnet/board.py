
from os.path import join as join_path

from torch.utils.tensorboard import SummaryWriter

from .record import Record


class Board(Record):
    __attributes__ = ['dir', 'root']

    def __init__(self, dir, root, steps=0, flush_secs=1):
        self.dir = dir
        self.root = root
        self.writer = SummaryWriter(
            join_path(root, dir),
            flush_secs=flush_secs
        )
        self.steps = steps

    def section(self, name):
        return BoardSection(name, self)

    def step(self):
        self.steps += 1

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value, self.steps)


class BoardSection(Record):
    __attributes__ = ['name', 'board']

    def __init__(self, name, board):
        self.name = name
        self.board = board

    def prefixed(self, key):
        return '%s/%s' % (self.name, key)

    def add_scalar(self, key, value):
        self.board.add_scalar(self.prefixed(key), value)
