
import torch

from slovnet.record import Record
from slovnet.pad import pad_sequence
from slovnet.chop import chop
from slovnet.shape import word_shape
from slovnet.batch import Batch

from .buffer import SortBuffer
from .common import WordShapeInferEncoder


ROOT_ID = '0'


class SyntaxTrainItem(Record):
    __attributes__ = ['word_ids', 'shape_ids', 'head_ids', 'rel_ids']


class SyntaxInput(Record):
    __attributes__ = ['word_id', 'shape_id', 'pad_mask']


class SyntaxTarget(Record):
    __attributes__ = ['head_id', 'rel_id', 'mask']


class SyntaxTrainEncoder:
    def __init__(self, words_vocab, shapes_vocab, rels_vocab,
                 batch_size=8, sort_size=1):
        self.words_vocab = words_vocab
        self.shapes_vocab = shapes_vocab
        self.rels_vocab = rels_vocab

        self.batch_size = batch_size
        self.sort = SortBuffer(sort_size, key=lambda _: len(_.tokens))

    def item(self, markup):
        word_ids, shape_ids, head_ids, rel_ids = [], [], [], []
        ids = {ROOT_ID: 0}

        for index, token in enumerate(markup.tokens, 1):
            ids[token.id] = index
            head_ids.append(token.head_id)

            rel_id = self.rels_vocab.encode(token.rel)
            rel_ids.append(rel_id)

            shape = word_shape(token.text)
            shape_id = self.shapes_vocab.encode(shape)
            shape_ids.append(shape_id)

            word = token.text.lower()
            word_id = self.words_vocab.encode(word)
            word_ids.append(word_id)

        head_ids = [ids[_] for _ in head_ids]
        return SyntaxTrainItem(word_ids, shape_ids, head_ids, rel_ids)

    def batch(self, chunk):
        word_id, shape_id, head_id, rel_id = [], [], [], []
        for item in chunk:
            word_id.append(torch.tensor(item.word_ids, dtype=torch.long))
            shape_id.append(torch.tensor(item.shape_ids, dtype=torch.long))
            head_id.append(torch.tensor(item.head_ids, dtype=torch.long))
            rel_id.append(torch.tensor(item.rel_ids, dtype=torch.long))

        word_id = pad_sequence(word_id, fill=self.words_vocab.pad_id)
        shape_id = pad_sequence(shape_id, fill=self.shapes_vocab.pad_id)
        pad_mask = word_id == self.words_vocab.pad_id
        input = SyntaxInput(word_id, shape_id, pad_mask)

        head_id = pad_sequence(head_id)
        rel_id = pad_sequence(rel_id, fill=self.rels_vocab.pad_id)
        mask = rel_id != self.rels_vocab.pad_id
        target = SyntaxTarget(head_id, rel_id, mask)

        return Batch(input, target)

    def __call__(self, markups):
        markups = self.sort(markups)
        items = (self.item(_) for _ in markups)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


class SyntaxInferEncoder(WordShapeInferEncoder):
    pass
