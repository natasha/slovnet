
from torch import nn

from .base import Module
from .crf import CRF
from .emb import WordShapeEmbedding
from .cnn import CNNEncoder


class NERHead(Module):
    def __init__(self, emb_dim, tags_num):
        super(NERHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)
        self.crf = CRF(tags_num)

    def forward(self, input):
        return self.proj(input)


class MorphHead(Module):
    def __init__(self, emb_dim, tags_num):
        super(MorphHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)

    def decode(self, pred):
        return pred.argmax(-1)

    def forward(self, input):
        return self.proj(input)


class TagEmbedding(WordShapeEmbedding):
    pass


class TagEncoder(CNNEncoder):
    pass


class Tag(Module):
    def __init__(self, emb, encoder, head):
        super(Tag, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, word_id, shape_id, pad_mask=None):
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, pad_mask)
        return self.head(x)


class NER(Tag):
    pass


class Morph(Tag):
    pass
