
from collections import defaultdict
from random import shuffle


class Buffer(object):
    def __call__(self, items):
        for item in items:
            self.append(item)
            if self.is_full:
                for item in self.flush():
                    yield item
        for item in self.flush():
            yield item


class ShuffleBuffer(Buffer):
    def __init__(self, cap):
        self.cap = cap
        self.reset()

    def append(self, item):
        self.buffer.append(item)

    def reset(self):
        self.buffer = []

    @property
    def is_full(self):
        return len(self.buffer) >= self.cap

    def flush(self):
        shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.reset()


class SizeBuffer(Buffer):
    def __init__(self, cap):
        self.cap = cap
        self.reset()

    def append(self, item):
        self.count += 1
        self.buffer[item.size].append(item)

    def reset(self):
        self.count = 0
        self.buffer = defaultdict(list)

    @property
    def is_full(self):
        return self.count >= self.cap

    def flush(self):
        for size in sorted(self.buffer):
            yield self.buffer[size]
        self.reset()
