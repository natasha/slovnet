
import torch
from torch import nn


class ShapeEmbedding(nn.Embedding):
    def __init__(self, vocab_size, dim, pad_id):
        super(ShapeEmbedding, self).__init__(vocab_size, dim, pad_id)
        self.dim = dim


class WordModel(nn.Module):
    def __init__(self, word_emb, shape_emb):
        super(WordModel, self).__init__()
        self.word_emb = word_emb
        self.shape_emb = shape_emb
        self.dim = word_emb.dim + shape_emb.dim

    def forward(self, input):
        word_id, shape_id = input
        word_emb = self.word_emb(word_id)
        shape_emb = self.shape_emb(shape_id)
        return torch.cat([word_emb, shape_emb], dim=-1)
