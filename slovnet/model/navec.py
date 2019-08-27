
import torch
from torch import nn


class PQEmbedding(nn.Module):
    def __init__(self, indexes, codes):
        super(PQEmbedding, self).__init__()

        vectors, qdim = indexes.shape
        qdim, centroids, chunk = codes.shape
        self.chunk = chunk
        self.dim = qdim * chunk

        self.pad_id = vectors
        pad_indexes = indexes.new_full((1, qdim), centroids)
        pad_codes = codes.new_zeros((qdim, 1, chunk))
        indexes = torch.cat([indexes, pad_indexes], dim=0)
        codes = torch.cat([codes, pad_codes], dim=1)
        codes = codes.transpose(1, 0)  # for gather, centroids x qdim x chunk

        self.codes = nn.Parameter(codes, requires_grad=False)
        self.indexes = nn.Parameter(indexes, requires_grad=False)

    def extra_repr(self):
        return 'indexes=[...], codes=[...]'

    def forward(self, input):
        shape = input.shape
        input = input.flatten()  # reshape in return

        indexes = self.indexes[input]
        vectors, qdim = indexes.shape

        indexes = indexes.view(vectors, qdim, 1)
        indexes = indexes.expand(vectors, qdim, self.chunk)
        output = self.codes.gather(0, indexes.long())  # vectors x qdim x chunk

        shape = shape + (self.dim,)  # for py2
        return output.view(*shape)


class NavecEmbedding(PQEmbedding):
    def __init__(self, id, indexes, codes):
        self.id = id
        super(NavecEmbedding, self).__init__(indexes, codes)

    def extra_repr(self):
        return 'id=%r, indexes=[...], codes=[...]' % self.id

    @classmethod
    def from_navec(cls, navec):
        return cls(
            navec.meta.id,
            torch.from_numpy(navec.pq.indexes),
            torch.from_numpy(navec.pq.codes)
        )
