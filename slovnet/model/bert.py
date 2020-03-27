
import torch
from torch import nn
from torch.nn import functional as F

from slovnet.record import Record

from .crf import CRF


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


class RuBERTConfig(BERTConfig):
    def __init__(self,
                 vocab_size=50106,
                 seq_len=512,
                 emb_dim=768,
                 layers_num=12,
                 heads_num=12,
                 hidden_dim=3072,
                 dropout=0.1,
                 norm_eps=1e-12):
        super(RuBERTConfig, self).__init__(
            vocab_size, seq_len, emb_dim,
            layers_num, heads_num, hidden_dim,
            dropout, norm_eps
        )


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, seq_len, emb_dim, dropout=0.1, norm_eps=1e-12):
        super(BERTEmbedding, self).__init__()
        self.word = nn.Embedding(vocab_size, emb_dim)
        self.position = nn.Embedding(seq_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.drop = nn.Dropout(dropout)

    def __call__(self, input):
        batch_size, seq_len = input.shape
        position = torch.arange(seq_len).expand_as(input).to(input.device)

        emb = self.word(input) + self.position(position)
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


#########
#
#   MLM
#
#########


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
        x = self.emb(input)
        x = self.encoder(x)
        return self.mlm(x)


#########
#
#   NER
#
######


class BERTNERHead(nn.Module):
    def __init__(self, emb_dim, tags_num):
        super(BERTNERHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)
        self.crf = CRF(tags_num)

    def forward(self, input):
        return self.proj(input)


class BERTNER(nn.Module):
    def __init__(self, emb, encoder, ner):
        super(BERTNER, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.ner = ner

    def forward(self, input):
        x = self.emb(input)
        x = self.encoder(x)
        return self.ner(x)


#########
#
#   MORPH
#
######


class BERTMorphHead(nn.Module):
    def __init__(self, emb_dim, tags_num):
        super(BERTMorphHead, self).__init__()
        self.emb_dim = emb_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(emb_dim, tags_num)

    def forward(self, input):
        return self.proj(input)


class BERTMorph(nn.Module):
    def __init__(self, emb, encoder, morph):
        super(BERTMorph, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.morph = morph

    def forward(self, input):
        x = self.emb(input)
        x = self.encoder(x)
        return self.morph(x)


#######
#
#   SYNTAX
#
#######


class FF(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(FF, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        x = self.proj(input)
        x = self.relu(x)
        return self.drop(x)


def append_root(input, root):
    batch_size, seq_size, emb_dim = input.shape
    root = root.repeat(batch_size).view(batch_size, 1, emb_dim)
    return torch.cat((root, input), dim=1)


def strip_root(input):
    input = input[:, 1:, :]
    return input.contiguous()


class BERTSyntaxHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(BERTSyntaxHead, self).__init__()
        self.head = FF(input_dim, hidden_dim, dropout)
        self.tail = FF(input_dim, hidden_dim, dropout)

        self.root = nn.Parameter(torch.empty(input_dim))
        self.kernel = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.root, -0.1, 0.1)
        nn.init.eye_(self.kernel)

    def forward(self, input):
        input = append_root(input, self.root)
        head = self.head(input)
        tail = self.tail(input)

        x = head.matmul(self.kernel)
        x = x.matmul(tail.transpose(-2, -1))
        return strip_root(x)


def select_head(input, root, index):
    batch_size, seq_size, emb_dim = input.shape
    input = append_root(input, root)  # batch x seq + 1 x emb

    # for root select root
    zero = torch.zeros(batch_size, 1).long().to(input.device)
    index = torch.cat((zero, index), dim=-1)  # batch x seq + 1 x emb

    # prep for gather
    index = index.view(batch_size, seq_size + 1, 1)
    index = index.expand(batch_size, seq_size + 1, emb_dim)

    input = torch.gather(input, dim=-2, index=index)
    return strip_root(input)  # batch x seq x emb


class BERTSyntaxRel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rel_dim, dropout=0.1):
        super(BERTSyntaxRel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rel_dim = rel_dim

        self.head = FF(input_dim, hidden_dim, dropout)
        self.tail = FF(input_dim, hidden_dim, dropout)

        self.root = nn.Parameter(torch.empty(input_dim))
        self.kernel = nn.Parameter(torch.empty(hidden_dim, hidden_dim * rel_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.root, -0.1, 0.1)
        nn.init.xavier_uniform_(self.kernel)

    def forward(self, input, head_id):
        head = self.head(select_head(input, self.root, head_id))
        tail = self.tail(input)

        batch_size, seq_size, _ = input.shape
        x = head.matmul(self.kernel)  # batch x seq x hidden * rel
        x = x.view(batch_size, seq_size, self.rel_dim, self.hidden_dim)
        x = x.matmul(tail.view(batch_size, seq_size, self.hidden_dim, 1))
        return x.view(batch_size, seq_size, self.rel_dim)


def select_words(input, mask):
    batch_size, seq_size, emb_dim = input.shape
    # assert mask.sum(-1) all the same
    select_size = mask.sum().item() // batch_size
    return input[mask].view(batch_size, select_size, emb_dim)


class SyntaxPred(Record):
    __attributes__ = ['head', 'rel']

    def __init__(self, head, rel):
        self.head = head
        self.rel = rel


class BERTSyntax(nn.Module):
    def __init__(self, emb, encoder, head, rel):
        super(BERTSyntax, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head
        self.rel = rel

    def forward(self, input, word_mask, pad_mask, head_id):
        x = self.emb(input)
        x = self.encoder(x, pad_mask)
        x = select_words(x, word_mask)
        return SyntaxPred(
            head=self.head(x),
            rel=self.rel(x, head_id)
        )
