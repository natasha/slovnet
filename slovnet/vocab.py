
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
        """
        Initialize an iterable.

        Args:
            self: (todo): write your description
            items: (todo): write your description
        """
        self.items = items
        self.item_ids = {
            item: id
            for id, item in enumerate(self.items)
        }
        self.unk_id = self.item_ids.get(UNK)
        self.pad_id = self.item_ids.get(PAD)

    def encode(self, item):
        """
        Encode the given item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return self.item_ids.get(item, self.unk_id)

    def decode(self, id):
        """
        Decode a single item.

        Args:
            self: (todo): write your description
            id: (int): write your description
        """
        return self.items[id]

    def __len__(self):
        """
        Returns the number of items in bytes.

        Args:
            self: (todo): write your description
        """
        return len(self.items)

    def __repr__(self):
        """
        Return a human - readable representation of this object.

        Args:
            self: (todo): write your description
        """
        return '%s(items=[...])' % self.__class__.__name__

    def _repr_pretty_(self, printer, cycle):
        """
        Print a pretty printed representation of this object.

        Args:
            self: (todo): write your description
            printer: (todo): write your description
            cycle: (todo): write your description
        """
        printer.text(repr(self))

    @classmethod
    def load(cls, path):
        """
        Load a list of items in path.

        Args:
            cls: (todo): write your description
            path: (str): write your description
        """
        items = list(load_lines(path))
        return cls(items)

    def dump(self, path):
        """
        Dump the given file path.

        Args:
            self: (todo): write your description
            path: (str): write your description
        """
        dump_lines(self.items, path)


class BERTVocab(Vocab):
    def __init__(self, items):
        """
        Initialize item ids.

        Args:
            self: (todo): write your description
            items: (todo): write your description
        """
        super(BERTVocab, self).__init__(items)
        self.sep_id = self.item_ids[SEP]
        self.cls_id = self.item_ids[CLS]
        self.mask_id = self.item_ids[MASK]


class BIOTagsVocab(Vocab):
    def __init__(self, types):
        """
        Initialize the types.

        Args:
            self: (todo): write your description
            types: (todo): write your description
        """
        self.types = types

        items = [PAD, O]
        for type in types:
            for part in [B, I]:
                items.append(format_bio(part, type))

        super(BIOTagsVocab, self).__init__(items)
