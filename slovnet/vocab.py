# coding: utf8

from .record import Record
from .bio import (
    B, I, O,
    format_bio
)


UNK = '<unk>'
PAD = '<pad>'
CLS = '<cls>'
SEP = '<sep>'
MASK = '<mask>'


class Vocab(Record):
    __attributes__ = ['items']

    def __init__(self, items):
        self.items = items
        self.item_ids = {
            item: id
            for id, item in enumerate(self.items)
        }
        self.unk_id = self.item_ids.get(UNK)
        self.pad_id = self.item_ids.get(PAD)

    def encode(self, item):
        return self.item_ids.get(item, self.unk_id)

    def decode(self, id):
        return self.items[id]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return '%s(items=[...])' % self.__class__.__name__

    def _repr_pretty_(self, printer, cycle):
        printer.text(repr(self))


class BERTVocab(Vocab):
    def __init__(self, items):
        super(BERTVocab, self).__init__(items)
        self.sep_id = self.item_ids[SEP]
        self.cls_id = self.item_ids[CLS]
        self.mask_id = self.item_ids[MASK]


class WordsVocab(Vocab):
    def encode(self, word):
        word = word.lower()
        return self.item_ids.get(word, self.unk_id)


class ShapesVocab(Vocab):
    pass


def type_tags(types):
    yield O
    for type in types:
        for part in [B, I]:
            yield format_bio(part, type)


class TagsVocab(Vocab):
    __hide_repr__ = False

    def __init__(self, types):
        self.types = types
        items = list(type_tags(types))
        super(TagsVocab, self).__init__(items)
