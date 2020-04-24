
import torch

from slovnet.record import Record
from slovnet.pad import pad_sequence
from slovnet.chop import chop
from slovnet.shape import word_shape


class WordShapeInferInput(Record):
    __attributes__ = ['word_id', 'shape_id', 'pad_mask']


class WordShapeInferEncoder:
    def __init__(self, words_vocab, shapes_vocab,
                 batch_size=8):
        self.words_vocab = words_vocab
        self.shapes_vocab = shapes_vocab

        self.batch_size = batch_size

    def item(self, words):
        word_ids, shape_ids = [], []
        for word in words:
            shape = word_shape(word)
            word_id = self.words_vocab.encode(word.lower())
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
        pad_mask = word_id == self.words_vocab.pad_id
        return WordShapeInferInput(word_id, shape_id, pad_mask)

    def __call__(self, items):
        items = (self.item(_) for _ in items)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.input(chunk)
