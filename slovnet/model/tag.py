
from torch import nn

from .base import Module
from .crf import CRF
from .emb import WordShapeEmbedding as TagEmbedding
from .cnn import CNNEncoder as TagEncoder


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


class Tag(Module):
    def __init__(self, emb, encoder, head):
        super(Tag, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, word_id, shape_id, mask=None):
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, mask)
        return self.head(x)


NER = Tag
Morph = Tag
