
import numpy as np

from slovnet.record import Record


class Weight(np.ndarray):
    def __repr__(self):
        return '{name}(..., shape={shape!r})'.format(
            name=self.__class__.__name__,
            shape=self.shape
        )

    def _repr_pretty_(self, printer, cycle):
        printer.text(self.__repr__())

    @property
    def as_bytes(self):
        return self.tobytes()

    @classmethod
    def from_file(cls, file):
        buffer = file.read()
        return np.frombuffer(buffer, np.float32).view(cls)


class Module(Record):
    @property
    def as_scheme(self):
        from .scheme import SchemeVisitor

        visitor = SchemeVisitor()
        scheme = visitor(self)
        return scheme, visitor.context

    def pack(self, id):
        from .pack import Meta, Pack

        scheme, context = self.as_scheme
        meta = Meta(id)
        return Pack(meta, scheme, context)


class Transpose(Module):
    __attributes__ = ['axis1', 'axis2']

    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def __call__(self, input):
        return np.swapaxes(input, self.axis1, self.axis2)


class Sequential(Module):
    __attributes__ = ['modules']

    def __init__(self, modules):
        self.modules = modules

    def __call__(self, input):
        output = input
        for module in self.modules:
            output = module(output)
        return output


class Linear(Module):
    __attributes__ = ['weight', 'bias']

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.in_dim, self.out_dim = self.weight.shape

    def __call__(self, input):
        shape = input.shape
        input = input.reshape(-1, self.in_dim)
        output = np.matmul(input, self.weight) + self.bias

        shape = shape[:-1] + (self.out_dim,)
        return output.reshape(*shape)


class Conv1d(Module):
    __attributes__ = ['weight', 'bias', 'padding']

    def __init__(self, weight, bias, padding):
        self.weight = weight  # filters x in x kernel
        self.bias = bias  # filters
        self.padding = padding

    def __call__(self, input):
        input = np.pad(
            input,
            # batch no pad, in no pad, pad seq
            [(0, 0), (0, 0), (self.padding, self.padding)],
            mode='constant', constant_values=0
        )
        input = np.ascontiguousarray(input)

        batch_size, in_dim, seq_len = input.shape
        batch_stride, in_stride, seq_stride = input.strides
        unit_stride = input.data.itemsize
        filters_count, in_dim, kernel_size = self.weight.shape
        windows_count = seq_len - kernel_size + 1

        # populate conv windows
        windows = np.ndarray(
            [batch_size, windows_count, in_dim, kernel_size],
            dtype=input.dtype,
            buffer=input.data,
            strides=[batch_stride, unit_stride, in_stride, seq_stride]
        )

        # conv
        windows = windows.reshape(batch_size * windows_count, in_dim * kernel_size)
        weight = self.weight.reshape(filters_count, in_dim * kernel_size)
        output = np.matmul(windows, weight.T) + self.bias
        output = output.reshape(batch_size, windows_count, filters_count)

        # as in torch
        return output.swapaxes(2, 1)  # batch x filters x windows


class ReLU(Module):
    def __call__(self, input):
        return input.clip(0)


class BatchNorm1d(Module):
    __attributes__ = ['weight', 'bias', 'mean', 'std']

    def __init__(self, weight, bias, mean, std):
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std

    def __call__(self, input):
        # input is N x C x L, do ops on C
        input = input.swapaxes(2, 1)
        output = (input - self.mean) / self.std * self.weight + self.bias
        return output.swapaxes(2, 1)  # recover shape


class Embedding(Module):
    __attributes__ = ['weight']

    def __init__(self, weight):
        self.weight = weight
        _, self.dim = self.weight.shape

    def __call__(self, input):
        shape = input.shape
        input = input.flatten()
        weight = self.weight[input]
        shape = shape + (self.dim,)
        return weight.reshape(*shape)


class StackEmbedding(Embedding):
    __attributes__ = ['embs']

    def __init__(self, embs):
        self.embs = embs

    def __call__(self, input):
        ids = [
            emb(id)
            for emb, id
            in zip(self.embs, input)
        ]
        return np.concatenate(ids, axis=-1)


class NavecEmbedding(Embedding):
    __attributes__ = ['id', 'indexes', 'codes']

    def __init__(self, id, indexes, codes):
        self.id = id
        self.indexes = indexes
        self.codes = codes

        qdim, centroids, chunk = codes.shape
        self.dim = qdim * chunk
        self.qdims = np.arange(qdim)

    @classmethod
    def from_navec(cls, navec):
        return cls(
            navec.meta.id,
            navec.pq.indexes,
            navec.pq.codes
        )

    def __call__(self, input):
        shape = input.shape
        input = input.flatten()
        indexes = self.indexes[input]
        output = self.codes[self.qdims, indexes]
        shape = shape + (self.dim,)
        return output.reshape(*shape)


class CRF(Module):
    __attributes__ = ['transitions']

    def __init__(self, transitions):
        self.transitions = transitions

    def __call__(self, emissions):
        emissions = emissions.swapaxes(1, 0)
        seq_len, batch_size, tags_num = emissions.shape

        history = []
        score = emissions[0]  # batch x tags
        for index in range(1, seq_len):
            score_ = score.reshape(batch_size, tags_num, 1)
            emissions_ = emissions[index].reshape(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            indexes = score_.argmax(-2)
            # https://stackoverflow.com/questions/20128837/get-indices-of-numpy-argmax-elements-over-an-axis
            batch_indexes, tags_indexes = np.indices(indexes.shape)
            score = score_[batch_indexes, indexes, tags_indexes]
            history.append(indexes)

        batch = []
        for index in range(batch_size):
            best = score[index].argmax()
            tags = [best]
            for indexes in reversed(history):
                best = indexes[index][best]
                tags.append(best)
            tags.reverse()
            batch.append(tags)

        return np.array(batch)
