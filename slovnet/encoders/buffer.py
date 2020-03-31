
from collections import defaultdict
from random import shuffle


class Buffer:
    def __call__(self, items):
        for item in items:
            self.append(item)
            if self.is_full:
                for item in self.flush():
                    yield item
        for item in self.flush():
            yield item


class ShuffleBuffer(Buffer):
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

    def flush(self):
        shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.reset()


class LenBuffer(Buffer):
    def __init__(self, size):
        self.size = size
        self.reset()

    def append(self, item):
        self.count += 1
        self.buffer[len(item)].append(item)

    def reset(self):
        self.count = 0
        self.buffer = defaultdict(list)

    @property
    def is_full(self):
        return self.count >= self.size

    def flush(self):
        for len in sorted(self.buffer):
            yield self.buffer[len]
        self.reset()
