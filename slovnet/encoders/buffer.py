
from random import shuffle


class Buffer:
    def __init__(self, size):
        self.size = size
        self.reset()

    def append(self, item):
        self.buffer.append(item)

    def reset(self):
        self.buffer = []

    @property
    def is_full(self):
        return len(self.buffer) >= self.size

    def __call__(self, items):
        for item in items:
            self.append(item)
            if self.is_full:
                for item in self.flush():
                    yield item
        for item in self.flush():
            yield item


class ShuffleBuffer(Buffer):
    def flush(self):
        shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.reset()


class SortBuffer(Buffer):
    def __init__(self, size, key):
        self.key = key
        Buffer.__init__(self, size)

    def flush(self):
        self.buffer.sort(key=self.key)
        for item in self.buffer:
            yield item
        self.reset()
