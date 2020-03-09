
from random import shuffle


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
