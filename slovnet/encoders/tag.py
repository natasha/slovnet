
import torch

from slovnet.record import Record
from slovnet.pad import pad_sequence
from slovnet.chop import chop, chop_drop
from slovnet.shape import word_shape
from slovnet.batch import Batch

from .buffer import ShuffleBuffer


class TagTrainInput(Record):
    __attributes__ = ['word_id', 'shape_id']


class TagInferInput(Record):
    __attributes__ = ['word_id', 'shape_id', 'mask']


class TagTrainEncoder:
    def __init__(self, words_vocab, shapes_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.shapes_vocab = shapes_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def items(self, markups):
        for markup in markups:
            for token in markup.tokens:
                shape = word_shape(token.text)
                word = token.text.lower()
                yield (
                    self.words_vocab.encode(word),
                    self.shapes_vocab.encode(shape),
                    self.tags_vocab.encode(token.tag)
                )

    def batch(self, chunk):
        chunk = torch.tensor(chunk, dtype=torch.long)  # batch x seq x (word, shp, tag)
        word_id, shape_id, tag_id = chunk.unbind(-1)

        input = TagTrainInput(word_id, shape_id)
        return Batch(input, tag_id)

    def __call__(self, markups):
        items = self.items(markups)
        seqs = chop_drop(items, self.seq_len)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


class TagInferEncoder:
    def __init__(self, words_vocab, shapes_vocab,
                 batch_size=8):
        self.words_vocab = words_vocab
        self.shapes_vocab = shapes_vocab

        self.batch_size = batch_size

    def item(self, markup):
        word_ids, shape_ids = [], []
        for token in markup.tokens:
            shape = word_shape(token.text)
            word = token.text.lower()
            word_id = self.words_vocab.encode(word)
            shape_id = self.shapes_vocab.encode(shape)
            word_ids.append(word_id)
            shape_ids.append(shape_id)
        return word_ids, shape_ids

    def input(self, items):
        word_id, shape_id = [], []
        for word_ids, shape_ids in items:
            word_id.append(torch.tensor(word_ids, dtype=torch.long))
            shape_id.append(torch.tensor(shape_ids, dtype=torch.long))
        word_id = pad_sequence(word_id, self.words_vocab.pad_id)
        shape_id = pad_sequence(shape_id, self.shapes_vocab.pad_id)
        mask = word_id == self.words_vocab.pad_id
        return TagInferInput(word_id, shape_id, mask)

    def __call__(self, items):
        items = (self.item(_) for _ in items)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.input(chunk)
