# coding: utf8

from .record import Record
from .bio import (
    B, I, O,
    format_bio
)


UNK = '<unk>'
PAD = '<pad>'


class Vocab(Record):
    __attributes__ = ['items']
    __hide_repr__ = True

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
        if self.__hide_repr__:
            name = self.__class__.__name__
            return '%s(items=[...])' % name
        return super(Vocab, self).__repr__()

    def _repr_pretty_(self, printer, cycle):
        if self.__hide_repr__:
            printer.text(repr(self))
        else:
            super(Vocab, self)._repr_pretty_(printer, cycle)


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
