
import torch
from torch import nn
from torch.nn import functional as F

from slovnet.record import Record
from slovnet.mask import fill_masked

from .base import Module
from .cnn import CNNEncoder
from .emb import WordShapeEmbedding


class FF(Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        """
        Initialize the graph.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            hidden_dim: (int): write your description
            dropout: (str): write your description
        """
        super(FF, self).__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, input):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        x = self.proj(input)
        x = self.relu(x)
        return self.drop(x)


def append_root(input, root):
    """
    Append a root element.

    Args:
        input: (todo): write your description
        root: (todo): write your description
    """
    batch_size, seq_len, emb_dim = input.shape
    root = root.repeat(batch_size).view(batch_size, 1, emb_dim)
    return torch.cat((root, input), dim=1)


def strip_root(input):
    """
    Strips the root of the string.

    Args:
        input: (todo): write your description
    """
    input = input[:, 1:, :]
    return input.contiguous()


def append_root_mask(mask):
    """
    Append mask to the root element.

    Args:
        mask: (array): write your description
    """
    return F.pad(mask, (1, 0), 'constant', True)


def matmul_mask(mask):
    """
    Matmul_mask mask.

    Args:
        mask: (todo): write your description
    """
    # 1 1 1 0 0
    # ->
    # 1 1 1 0 0
    # 1 1 1 0 0
    # 1 1 1 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0

    mask = mask.float()  # matmul not supported for bool
    mask = mask.unsqueeze(-2)  # batch x 1 x seq
    mask = mask.transpose(-2, -1).matmul(mask)
    return mask.bool()


class SyntaxHead(Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            hidden_dim: (int): write your description
            dropout: (str): write your description
        """
        super(SyntaxHead, self).__init__()
        self.head = FF(input_dim, hidden_dim, dropout)
        self.tail = FF(input_dim, hidden_dim, dropout)

        self.root = nn.Parameter(torch.empty(input_dim))
        self.kernel = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters.

        Args:
            self: (todo): write your description
        """
        nn.init.uniform_(self.root)
        nn.init.eye_(self.kernel)

    def decode(self, pred, mask):
        """
        Decode a mask.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            mask: (array): write your description
        """
        # multiple roots
        # loops, nonprojective
        # ~10% sents

        mask = append_root_mask(mask)
        mask = matmul_mask(mask)
        mask = strip_root(mask)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def forward(self, input):
        """
        Forward the forward of the graph.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        input = append_root(input, self.root)
        head = self.head(input)
        tail = self.tail(input)

        x = head.matmul(self.kernel)
        x = x.matmul(tail.transpose(-2, -1))
        return strip_root(x)


def gather_head(input, root, index):
    """
    Gather the first element in the sequence.

    Args:
        input: (array): write your description
        root: (todo): write your description
        index: (int): write your description
    """
    batch_size, seq_len, emb_dim = input.shape
    input = append_root(input, root)  # batch x seq + 1 x emb

    # for root select root
    zero = torch.zeros(batch_size, 1, dtype=torch.long, device=input.device)
    index = torch.cat((zero, index), dim=-1)  # batch x seq + 1 x emb

    # prep for gather
    index = index.view(batch_size, seq_len + 1, 1)
    index = index.expand(batch_size, seq_len + 1, emb_dim)

    input = torch.gather(input, dim=-2, index=index)
    return strip_root(input)  # batch x seq x emb


class SyntaxRel(Module):
    def __init__(self, input_dim, hidden_dim, rel_dim, dropout=0.1):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            input_dim: (int): write your description
            hidden_dim: (int): write your description
            rel_dim: (int): write your description
            dropout: (str): write your description
        """
        super(SyntaxRel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rel_dim = rel_dim

        self.head = FF(input_dim, hidden_dim, dropout)
        self.tail = FF(input_dim, hidden_dim, dropout)

        self.root = nn.Parameter(torch.empty(input_dim))
        self.kernel = nn.Parameter(torch.empty(hidden_dim, hidden_dim * rel_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the kernel parameters.

        Args:
            self: (todo): write your description
        """
        nn.init.uniform_(self.root)
        nn.init.xavier_uniform_(self.kernel)

    def decode(self, pred, mask):
        """
        Decode a boolean mask.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            mask: (todo): write your description
        """
        mask = mask.unsqueeze(-1)  # batch x seq x 1
        mask = mask.expand_as(pred)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def forward(self, input, head_id):
        """
        Forward computation

        Args:
            self: (todo): write your description
            input: (todo): write your description
            head_id: (str): write your description
        """
        head = self.head(gather_head(input, self.root, head_id))
        tail = self.tail(input)

        batch_size, seq_len, _ = input.shape
        x = head.matmul(self.kernel)  # batch x seq x hidden * rel
        x = x.view(batch_size, seq_len, self.rel_dim, self.hidden_dim)
        x = x.matmul(tail.view(batch_size, seq_len, self.hidden_dim, 1))
        return x.view(batch_size, seq_len, self.rel_dim)


class SyntaxPred(Record):
    __attributes__ = ['head_id', 'rel_id']


class SyntaxEmbedding(WordShapeEmbedding):
    pass


class SyntaxEncoder(CNNEncoder):
    pass


class Syntax(Module):
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
        super(Syntax, self).__init__()
        self.emb = emb
        self.encoder = encoder
        self.head = head
        self.rel = rel

    def forward(self, word_id, shape_id, pad_mask, target_head_id=None):
        """
        Parameters ---------- word_id : string.

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
            pad_mask: (todo): write your description
            target_head_id: (str): write your description
        """
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, pad_mask)

        head_id = self.head(x)
        if target_head_id is None:
            target_head_id = self.head.decode(head_id, ~pad_mask)

        return SyntaxPred(
            head_id=head_id,
            rel_id=self.rel(x, target_head_id)
        )
