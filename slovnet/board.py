
from os.path import join as join_path

from torch.utils.tensorboard import SummaryWriter

from .record import Record
from .log import log


class Board(Record):
    __attributes__ = ['steps']

    def __init__(self, steps=0):
        """
        Initialize steps.

        Args:
            self: (todo): write your description
            steps: (int): write your description
        """
        self.steps = steps

    def section(self, name):
        """
        Get a section by name.

        Args:
            self: (todo): write your description
            name: (str): write your description
        """
        return BoardSection(name, self)

    def step(self):
        """
        Perform one step.

        Args:
            self: (todo): write your description
        """
        self.steps += 1


class TensorBoard(Board):
    __attributes__ = ['dir', 'root', 'steps']

    def __init__(self, dir, root, steps=0, flush_secs=1):
        """
        Initialize the game.

        Args:
            self: (todo): write your description
            dir: (todo): write your description
            root: (str): write your description
            steps: (int): write your description
            flush_secs: (todo): write your description
        """
        self.dir = dir
        self.root = root
        self.writer = SummaryWriter(
            join_path(root, dir),
            flush_secs=flush_secs
        )
        super(TensorBoard, self).__init__(steps)

    def add_scalar(self, key, value):
        """
        Adds an scalar value.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        self.writer.add_scalar(key, value, self.steps)


class LogBoard(Board):
    def add_scalar(self, key, value):
        """
        Add scalarith value.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        log('{:>4} {:.4f} {}'.format(self.steps, value, key))


class MultiBoard(Board):
    __attributes__ = ['boards']

    def __init__(self, boards):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
            boards: (todo): write your description
        """
        self.boards = boards

    def step(self):
        """
        Step the board.

        Args:
            self: (todo): write your description
        """
        for board in self.boards:
            board.step()

    def add_scalar(self, key, value):
        """
        Add an scalar value

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        for board in self.boards:
            board.add_scalar(key, value)


class BoardSection(Record):
    __attributes__ = ['name', 'board']

    def prefixed(self, key):
        """
        Returns a pre - readable string for the given key.

        Args:
            self: (todo): write your description
            key: (str): write your description
        """
        return '%s/%s' % (self.name, key)

    def add_scalar(self, key, value):
        """
        Add a scalar value.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        self.board.add_scalar(self.prefixed(key), value)
