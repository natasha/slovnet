
from collections import OrderedDict

import torch
from torch import nn

from .base import Module
from .crf import CRF


class TagEmbedding(Module):
    def __init__(self, word, shape):
        super(TagEmbedding, self).__init__()
        self.word = word
        self.shape = shape
        self.dim = word.dim + shape.dim

    def forward(self, word_id, shape_id):
        word = self.word(word_id)
        shape = self.shape(shape_id)
        return torch.cat([word, shape], dim=-1)


def TagEncoderLayer(in_dim, out_dim, kernel_size):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv1d(
            in_dim, out_dim, kernel_size,
            padding=padding
        )),
        ('relu', nn.ReLU()),
        ('norm', nn.BatchNorm1d(out_dim))
    ]))


def gen_tag_encoder_layers(input_dim, layer_dims, kernel_size):
    dims = [input_dim] + layer_dims
    for index in range(1, len(dims)):
        in_dim = dims[index - 1]
        out_dim = dims[index]
        yield TagEncoderLayer(in_dim, out_dim, kernel_size)


class TagEncoder(Module):
    def __init__(self, input_dim, layer_dims, kernel_size):
        super(TagEncoder, self).__init__()

        layers = gen_tag_encoder_layers(input_dim, layer_dims, kernel_size)
        self.layers = nn.ModuleList(layers)
        self.dim = layer_dims[-1]

    def forward(self, input, mask=None):  # batch x seq x emb
        input = input.transpose(2, 1)  # batch x emb x seq

        if mask is not None:
            mask = mask.unsqueeze(1)  # batch x 1 x seq

        for layer in self.layers:
            input = layer(input)  # batch x dim x seq

            if mask is not None:
                input[mask.expand_as(input)] = 0

        return input.transpose(2, 1)  # batch x seq x dim


#######
#
#  NER
#
######


class NERHead(Module):
    def __init__(self, emb_dim, tags_num):
        super(NERHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)
        self.crf = CRF(tags_num)

    def forward(self, input):
        return self.proj(input)


class NER(Module):
    def __init__(self, emb, encoder, ner):
        super(NER, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.ner = ner

    def forward(self, *input, mask=None):
        x = self.emb(*input)
        x = self.encoder(x, mask)
        return self.ner(x)


########
#
#  MORPH
#
########


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


class Morph(Module):
    def __init__(self, emb, encoder, morph):
        super(Morph, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.morph = morph

    def forward(self, *input, mask=None):
        x = self.emb(*input)
        x = self.encoder(x, mask)
        return self.morph(x)
