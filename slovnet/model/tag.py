
from torch import nn

from .base import Module
from .crf import CRF
from .emb import WordShapeEmbedding
from .cnn import CNNEncoder


class NERHead(Module):
    def __init__(self, emb_dim, tags_num):
        """
        Initialize the iterator.

        Args:
            self: (todo): write your description
            emb_dim: (int): write your description
            tags_num: (int): write your description
        """
        super(NERHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)
        self.crf = CRF(tags_num)

    def forward(self, input):
        """
        Evaluate forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return self.proj(input)


class MorphHead(Module):
    def __init__(self, emb_dim, tags_num):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            emb_dim: (int): write your description
            tags_num: (int): write your description
        """
        super(MorphHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)

    def decode(self, pred):
        """
        Decode the given predicate.

        Args:
            self: (todo): write your description
            pred: (array): write your description
        """
        return pred.argmax(-1)

    def forward(self, input):
        """
        Evaluate forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        return self.proj(input)


class TagEmbedding(WordShapeEmbedding):
    pass


class TagEncoder(CNNEncoder):
    pass


class Tag(Module):
    def __init__(self, emb, encoder, head):
        """
        Initialize the encoder.

        Args:
            self: (todo): write your description
            emb: (todo): write your description
            encoder: (todo): write your description
            head: (todo): write your description
        """
        super(Tag, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, word_id, shape_id, pad_mask=None):
        """
        Parameters ---------- word_id : string.

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
            pad_mask: (todo): write your description
        """
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, pad_mask)
        return self.head(x)


class NER(Tag):
    pass


class Morph(Tag):
    pass
