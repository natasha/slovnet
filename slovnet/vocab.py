
from .record import Record
from .io import (
    load_lines,
    dump_lines
)
from .const import (
    B, I, O,

    UNK, PAD,
    CLS, SEP,
    MASK,
)
from .bio import format_bio


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

    @classmethod
    def load(cls, path):
        items = list(load_lines(path))
        return cls(items)

    def dump(self, path):
        dump_lines(self.items, path)


class BERTVocab(Vocab):
    def __init__(self, items):
        super(BERTVocab, self).__init__(items)
        self.sep_id = self.item_ids[SEP]
        self.cls_id = self.item_ids[CLS]
        self.mask_id = self.item_ids[MASK]


class BIOTagsVocab(Vocab):
    def __init__(self, types):
        self.types = types

        items = [PAD, O]
        for type in types:
            for part in [B, I]:
                items.append(format_bio(part, type))

        super(BIOTagsVocab, self).__init__(items)
