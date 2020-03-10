
import torch
from torch import nn
from torch.nn import functional as F

from slovnet.record import Record


class BERTConfig(Record):
    __attributes__ = [
        'vocab_size', 'seq_len', 'emb_dim',
        'layers_num', 'heads_num', 'hidden_dim',
        'dropout', 'norm_eps'
    ]

    def __init__(self, vocab_size, seq_len, emb_dim,
                 layers_num, heads_num, hidden_dim,
                 dropout, norm_eps):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.layers_num = layers_num
        self.heads_num = heads_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.norm_eps = norm_eps


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, dropout=0.1, norm_eps=1e-12):
        super(BERTEmbedding, self).__init__()
        self.word = nn.Embedding(vocab_size, emb_dim)
        self.position = nn.Embedding(seq_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.drop = nn.Dropout(dropout)

    def __call__(self, word_id, position_id):
        emb = self.word(word_id) + self.position(position_id)
        emb = self.norm(emb)
        return self.drop(emb)


def BERTLayer(emb_dim, heads_num, hidden_dim, dropout=0.1, norm_eps=1e-12):
    layer = nn.TransformerEncoderLayer(
        d_model=emb_dim,
        nhead=heads_num,
        dim_feedforward=hidden_dim,
        dropout=dropout,
        activation='gelu'
    )
    layer.norm1.eps = norm_eps
    layer.norm2.eps = norm_eps
    return layer


class BERTEncoder(nn.Module):
    def __init__(self, layers_num, emb_dim, heads_num, hidden_dim,
                 dropout=0.1, norm_eps=1e-12):
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            BERTLayer(
                emb_dim, heads_num, hidden_dim,
                dropout, norm_eps
            )
            for _ in range(layers_num)
        ])

    def forward(self, input):
        input = input.transpose(0, 1)  # torch expects seq x batch x emb
        for layer in self.layers:
            input = layer(input)
        return input.transpose(0, 1)  # restore


class BERTMLMHead(nn.Module):
    def __init__(self, emb_dim, vocab_size, norm_eps=1e-12):
        super(BERTMLMHead, self).__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.linear2 = nn.Linear(emb_dim, vocab_size)

    def forward(self, input):
        x = self.linear1(input)
        x = F.gelu(x)
        x = self.norm(x)
        return self.linear2(x)


class BERTMLM(nn.Module):
    def __init__(self, emb, encoder, mlm):
        super(BERTMLM, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.mlm = mlm

    def forward(self, input):
        batch_size, seq_len = input.shape
        position = torch.arange(seq_len).expand_as(input).to(input.device)

        x = self.emb(input, position)
        x = self.encoder(x)
        return self.mlm(x)
