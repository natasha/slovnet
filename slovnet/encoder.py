
from random import (
    seed,
    shuffle
)

from .record import Record
from .shape import get_shape
from .chop import (
    chop,
    chop_equal
)
from .markup import TagMarkup


class Encoder(Record):
    def __call__(self, item):
        raise NotImplementedError

    def map(self, items):
        return [
            self(_)
            for _ in items
        ]


class StackEncoder(Encoder):
    __attributes__ = ['encoders']

    def __init__(self, encoders):
        self.encoders = encoders

    def __call__(self, item):
        return [
            _(item)
            for _ in self.encoders
        ]

    def map(self, items):
        ids = [[] for _ in self.encoders]
        for item in items:
            for index, encoder in enumerate(self.encoders):
                id = encoder(item)
                ids[index].append(id)
        return ids


class VocabEncoder(Encoder):
    __attributes__ = ['vocab']

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, item):
        return self.vocab.encode(item)


class WordEncoder(VocabEncoder):
    def __call__(self, token):
        return self.vocab.encode(token.text)


class ShapeEncoder(VocabEncoder):
    def __call__(self, token):
        shape = get_shape(token)
        return self.vocab.encode(shape)


class TagEncoder(VocabEncoder):
    pass


class MarkupEncoder(Encoder):
    __attributes__ = ['token_encoder', 'tag_encoder']

    def __init__(self, token_encoder, tag_encoder):
        self.token_encoder = token_encoder
        self.tag_encoder = tag_encoder

    def __call__(self, markup):
        return (
            self.token_encoder.map(markup.tokens),
            self.tag_encoder.map(markup.tags)
        )


class ShuffleBuffer(object):
    def __init__(self, size, seed=1):
        self.size = size
        self.seed = seed
        self.reset()

    def append(self, item):
        self.buffer.append(item)

    def reset(self):
        self.buffer = []

    @property
    def is_full(self):
        return len(self.buffer) >= self.size

    def flush(self):
        seed(self.seed)
        shuffle(self.buffer)
        for item in self.buffer:
            yield item
        self.reset()


class BatchEncoder(Encoder):
    __attributes__ = [
        'tokenizer',
        'markup_encoder',
        'seq_len',
        'batch_size',
        'shuffle_buffer_size'
    ]

    def __init__(self, tokenizer, markup_encoder,
                 seq_len=100, batch_size=1, shuffle_buffer_size=1):
        self.tokenizer = tokenizer
        self.markup_encoder = markup_encoder
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle_buffer_size = shuffle_buffer_size
        self.buffer = ShuffleBuffer(self.shuffle_buffer_size)

    def tokenize(self, markups):
        for markup in markups:
            markup = markup.to_tag(self.tokenizer)
            for pair in markup.pairs:
                yield pair

    def chop(self, pairs):
        chunks = chop_equal(pairs, self.seq_len)
        for chunk in chunks:
            yield TagMarkup.from_pairs(chunk)

    def shuffle(self, items):
        for item in items:
            self.buffer.append(item)
            if self.buffer.is_full:
                for item in self.buffer.flush():
                    yield item
        for item in self.buffer.flush():
            yield item

    def encode(self, groups):
        # uses torch, not needed in infer
        from .batch import Batch

        for group in groups:
            ids = self.markup_encoder.map(group)
            yield Batch.from_markup_encoder(ids)

    def map(self, markups):
        pairs = self.tokenize(markups)
        markups = self.chop(pairs)
        markups = self.shuffle(markups)
        groups = chop(markups, self.batch_size)
        return self.encode(groups)
