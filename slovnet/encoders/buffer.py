
from random import shuffle


class Buffer:
    def __init__(self, size):
        """
        Reset the state.

        Args:
            self: (todo): write your description
            size: (int): write your description
        """
        self.size = size
        self.reset()

    def append(self, item):
        """
        Add item to the end of the buffer.

        Args:
            self: (todo): write your description
            item: (array): write your description
        """
        self.buffer.append(item)

    def reset(self):
        """
        Reset the buffer.

        Args:
            self: (todo): write your description
        """
        self.buffer = []

    @property
    def is_full(self):
        """
        Returns true if the buffer is full.

        Args:
            self: (todo): write your description
        """
        return len(self.buffer) >= self.size

    def __call__(self, items):
        """
        Iterate over items.

        Args:
            self: (todo): write your description
            items: (todo): write your description
        """
        for item in items:
            self.append(item)
            if self.is_full:
                for item in self.flush():
                    yield item
        for item in self.flush():
            yield item


class ShuffleBuffer(Buffer):
    def flush(self):
        """
        Flush all the buffers.

        Args:
            self: (todo): write your description
        """
        shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.reset()


class SortBuffer(Buffer):
    def __init__(self, size, key):
        """
        Initialize the buffer.

        Args:
            self: (todo): write your description
            size: (int): write your description
            key: (str): write your description
        """
        self.key = key
        Buffer.__init__(self, size)

    def flush(self):
        """
        Flush all remaining items from the queue.

        Args:
            self: (todo): write your description
        """
        self.buffer.sort(key=self.key)
        for item in self.buffer:
            yield item
        self.reset()
