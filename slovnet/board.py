
from os.path import join as join_path

from torch.utils.tensorboard import SummaryWriter

from .record import Record
from .log import log


class Board(Record):
    __attributes__ = ['steps']

    def __init__(self, steps=0):
        self.steps = steps

    def section(self, name):
        return BoardSection(name, self)

    def step(self):
        self.steps += 1


class TensorBoard(Board):
    __attributes__ = ['dir', 'root', 'steps']

    def __init__(self, dir, root, steps=0, flush_secs=1):
        self.dir = dir
        self.root = root
        self.writer = SummaryWriter(
            join_path(root, dir),
            flush_secs=flush_secs
        )
        super(TensorBoard, self).__init__(steps)

    def add_scalar(self, key, value):
        self.writer.add_scalar(key, value, self.steps)


class LogBoard(Board):
    def add_scalar(self, key, value):
        log('{:>4} {:.4f} {}'.format(self.steps, value, key))


class MultiBoard(Board):
    __attributes__ = ['boards']

    def __init__(self, boards):
        self.boards = boards

    def step(self):
        for board in self.boards:
            board.step()

    def add_scalar(self, key, value):
        for board in self.boards:
            board.add_scalar(key, value)


class BoardSection(Record):
    __attributes__ = ['name', 'board']

    def prefixed(self, key):
        return '%s/%s' % (self.name, key)

    def add_scalar(self, key, value):
        self.board.add_scalar(self.prefixed(key), value)
