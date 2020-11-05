
import torch
from torch import nn
from torch.nn import functional as F

from slovnet.record import Record
from slovnet.mask import pad_masked

from .base import Module
from .tag import (
    NERHead,
    MorphHead
)
from .syntax import (
    SyntaxHead,
    SyntaxRel,
    SyntaxPred
)


class BERTConfig(Record):
    __attributes__ = [
        'vocab_size', 'seq_len', 'emb_dim',
        'layers_num', 'heads_num', 'hidden_dim',
        'dropout', 'norm_eps'
    ]


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
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            vocab_size: (int): write your description
            seq_len: (int): write your description
            emb_dim: (int): write your description
            layers_num: (int): write your description
            heads_num: (int): write your description
            hidden_dim: (int): write your description
            dropout: (str): write your description
            norm_eps: (int): write your description
        """
        super(RuBERTConfig, self).__init__(
            vocab_size, seq_len, emb_dim,
            layers_num, heads_num, hidden_dim,
            dropout, norm_eps
        )


class BERTEmbedding(Module):
    def __init__(self, vocab_size, seq_len, emb_dim, dropout=0.1, norm_eps=1e-12):
        """
        Initialize embedding.

        Args:
            self: (todo): write your description
            vocab_size: (int): write your description
            seq_len: (int): write your description
            emb_dim: (int): write your description
            dropout: (str): write your description
            norm_eps: (int): write your description
        """
        super(BERTEmbedding, self).__init__()
        self.word = nn.Embedding(vocab_size, emb_dim)
        self.position = nn.Embedding(seq_len, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.drop = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, config):
        """
        Initialize a vocab from a configuration.

        Args:
            cls: (todo): write your description
            config: (todo): write your description
        """
        return cls(
            config.vocab_size, config.seq_len, config.emb_dim,
            config.dropout, config.norm_eps
        )

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        batch_size, seq_len = input.shape
        position = torch.arange(seq_len).expand_as(input).to(input.device)

        emb = self.word(input) + self.position(position)
        emb = self.norm(emb)
        return self.drop(emb)


def BERTLayer(emb_dim, heads_num, hidden_dim, dropout=0.1, norm_eps=1e-12):
    """
    Return the relative todo.

    Args:
        emb_dim: (int): write your description
        heads_num: (int): write your description
        hidden_dim: (int): write your description
        dropout: (bool): write your description
        norm_eps: (float): write your description
    """
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


class BERTEncoder(Module):
    def __init__(self, layers_num, emb_dim, heads_num, hidden_dim,
                 dropout=0.1, norm_eps=1e-12):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            layers_num: (int): write your description
            emb_dim: (int): write your description
            heads_num: (int): write your description
            hidden_dim: (int): write your description
            dropout: (str): write your description
            norm_eps: (int): write your description
        """
        super(BERTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            BERTLayer(
                emb_dim, heads_num, hidden_dim,
                dropout, norm_eps
            )
            for _ in range(layers_num)
        ])

    @classmethod
    def from_config(cls, config):
        """
        Initialize a layer from a configuration.

        Args:
            cls: (todo): write your description
            config: (todo): write your description
        """
        return cls(
            config.layers_num, config.emb_dim, config.heads_num, config.hidden_dim,
            config.dropout, config.norm_eps
        )

    def forward(self, input, pad_mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            pad_mask: (todo): write your description
        """
        input = input.transpose(0, 1)  # torch expects seq x batch x emb
        for layer in self.layers:
            input = layer(input, src_key_padding_mask=pad_mask)
        return input.transpose(0, 1)  # restore


#########
#
#   MLM
#
#########


class BERTMLMHead(Module):
    def __init__(self, emb_dim, vocab_size, norm_eps=1e-12):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            emb_dim: (int): write your description
            vocab_size: (int): write your description
            norm_eps: (int): write your description
        """
        super(BERTMLMHead, self).__init__()
        self.linear1 = nn.Linear(emb_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim, eps=norm_eps)
        self.linear2 = nn.Linear(emb_dim, vocab_size)

    def forward(self, input):
        """
        R forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        x = self.linear1(input)
        x = F.gelu(x)
        x = self.norm(x)
        return self.linear2(x)


class BERTMLM(Module):
    def __init__(self, emb, encoder, head):
        """
        Initialize the encoder.

        Args:
            self: (todo): write your description
            emb: (todo): write your description
            encoder: (todo): write your description
            head: (todo): write your description
        """
        super(BERTMLM, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, input):
        """
        Return the forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        x = self.emb(input)
        x = self.encoder(x)
        return self.head(x)


#########
#
#   TAG
#
######


class BERTNERHead(NERHead):
    pass


class BERTMorphHead(MorphHead):
    pass


class BERTTag(Module):
    def __init__(self, emb, encoder, head):
        """
        Initialize the encoder.

        Args:
            self: (todo): write your description
            emb: (todo): write your description
            encoder: (todo): write your description
            head: (todo): write your description
        """
        super(BERTTag, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head

    def forward(self, input, pad_mask=None):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            pad_mask: (todo): write your description
        """
        x = self.emb(input)
        x = self.encoder(x, pad_mask)
        return self.head(x)


class BERTNER(BERTTag):
    pass


class BERTMorph(BERTTag):
    pass


#######
#
#   SYNTAX
#
#######


class BERTSyntaxHead(SyntaxHead):
    pass


class BERTSyntaxRel(SyntaxRel):
    pass


class BERTSyntax(Module):
    def __init__(self, emb, encoder, head, rel):
        """
        Initialize the encoder.

        Args:
            self: (todo): write your description
            emb: (todo): write your description
            encoder: (todo): write your description
            head: (todo): write your description
            rel: (todo): write your description
        """
        super(BERTSyntax, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head
        self.rel = rel

    def forward(self, input, word_mask, pad_mask,
                target_mask, target_head_id=None):
        """
        Parameters ---------- input : str.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            word_mask: (todo): write your description
            pad_mask: (todo): write your description
            target_mask: (todo): write your description
            target_head_id: (str): write your description
        """
        x = self.emb(input)
        x = self.encoder(x, pad_mask)
        x = pad_masked(x, word_mask)

        head_id = self.head(x)
        if target_head_id is None:
            target_head_id = self.head.decode(head_id, target_mask)

        return SyntaxPred(
            head_id=head_id,
            rel_id=self.rel(x, target_head_id)
        )
