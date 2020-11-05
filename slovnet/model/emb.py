
import torch
from torch import nn

from .base import Module


class Embedding(nn.Embedding, Module):
    def __init__(self, vocab_size, dim, pad_id):
        """
        Initialize the iterator.

        Args:
            self: (todo): write your description
            vocab_size: (int): write your description
            dim: (int): write your description
            pad_id: (int): write your description
        """
        super(Embedding, self).__init__(vocab_size, dim, pad_id)
        self.dim = dim


class PQEmbedding(Module):
    def __init__(self, indexes, codes):
        """
        Initialize the subdim.

        Args:
            self: (todo): write your description
            indexes: (str): write your description
            codes: (array): write your description
        """
        super(PQEmbedding, self).__init__()

        qdim, centroids, subdim = codes.shape
        self.subdim = subdim
        self.dim = qdim * subdim

        codes = codes.transpose(1, 0)  # for gather, centroids x qdim x subdim
        self.codes = nn.Parameter(codes, requires_grad=False)
        self.indexes = nn.Parameter(indexes, requires_grad=False)

    def extra_repr(self):
        """
        Return a string representation of this object.

        Args:
            self: (todo): write your description
        """
        return 'indexes=[...], codes=[...]'

    def forward(self, input):
        """
        Compute forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
        shape = input.shape
        input = input.flatten()  # reshape in return

        indexes = self.indexes[input]
        vectors, qdim = indexes.shape

        indexes = indexes.view(vectors, qdim, 1)
        indexes = indexes.expand(vectors, qdim, self.subdim)
        output = self.codes.gather(0, indexes.long())  # vectors x qdim x subdim
        return output.view(*shape, self.dim)


class NavecEmbedding(PQEmbedding):
    def __init__(self, navec):
        """
        Initialize metadata

        Args:
            self: (todo): write your description
            navec: (todo): write your description
        """
        self.id = navec.meta.id
        super(NavecEmbedding, self).__init__(
            torch.from_numpy(navec.pq.indexes),
            torch.from_numpy(navec.pq.codes)
        )

    def extra_repr(self):
        """
        Return a human - readable string.

        Args:
            self: (todo): write your description
        """
        return 'id=%r, indexes=[...], codes=[...]' % self.id


class WordShapeEmbedding(Module):
    def __init__(self, word, shape):
        """
        Initialize a word.

        Args:
            self: (todo): write your description
            word: (str): write your description
            shape: (int): write your description
        """
        super(WordShapeEmbedding, self).__init__()
        self.word = word
        self.shape = shape
        self.dim = word.dim + shape.dim

    def forward(self, word_id, shape_id):
        """
        Parameters ---------- word_id : str

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
        """
        word = self.word(word_id)
        shape = self.shape(shape_id)
        return torch.cat([word, shape], dim=-1)
