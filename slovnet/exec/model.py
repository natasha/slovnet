
import numpy as np

from slovnet.record import Record, parse_annotation
from slovnet.visitor import Visitor

from .mask import fill_masked


class Weight(Record):
    __attributes__ = ['shape', 'dtype', 'array']

    def empty(self):
        return self.replace(array=None)

    @property
    def is_empty(self):
        return self.array is None

    @property
    def is_id(self):
        return type(self.array) is int


class Module(Record):
    def separate_arrays(self):
        visitor = SeparateArraysVisitor()
        scheme = visitor(self)
        return visitor.arrays, scheme

    def inject_arrays(self, arrays):
        visitor = InjectArraysVisitor(arrays)
        return visitor(self)

    def strip_navec(self):
        visitor = StripNavecVisitor()
        return visitor(self)

    def inject_navec(self, navec):
        visitor = InjectNavecVisitor(navec)
        return visitor(self)

    @property
    def weights(self):
        visitor = WeightsVisitor()
        visitor(self)
        return visitor.weights


class Linear(Module):
    __attributes__ = ['weight', 'bias']
    __annotations__ = {
        'weight': Weight,
        'bias': Weight
    }

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.in_dim, self.out_dim = self.weight.shape

    def __call__(self, input):
        shape = input.shape
        input = input.reshape(-1, self.in_dim)
        output = np.matmul(input, self.weight.array) + self.bias.array

        shape = shape[:-1] + (self.out_dim,)
        return output.reshape(*shape)


class Conv1d(Module):
    __attributes__ = [
        'weight',  # filters x in x kernel
        'bias',  # filters
        'padding'
    ]
    __annotations__ = {
        'weight': Weight,
        'bias': Weight
    }

    def __call__(self, input):
        input = np.pad(
            input,
            # batch no pad, in no pad, pad seq
            ((0, 0), (0, 0), (self.padding, self.padding)),
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
            (batch_size, windows_count, in_dim, kernel_size),
            dtype=input.dtype,
            buffer=input.data,
            strides=(batch_stride, unit_stride, in_stride, seq_stride)
        )

        # conv
        windows = windows.reshape(batch_size * windows_count, in_dim * kernel_size)
        weight = self.weight.array.reshape(filters_count, in_dim * kernel_size)
        output = np.matmul(windows, weight.T) + self.bias.array
        output = output.reshape(batch_size, windows_count, filters_count)

        # as in torch
        return output.swapaxes(2, 1)  # batch x filters x windows


class ReLU(Module):
    def __call__(self, input):
        return input.clip(0)


class BatchNorm1d(Module):
    __attributes__ = ['weight', 'bias', 'mean', 'std']
    __annotations__ = {
        'weight': Weight,
        'bias': Weight,
        'mean': Weight,
        'std': Weight
    }

    def __call__(self, input):
        # input is N x C x L, do ops on C
        input = input.swapaxes(2, 1)
        output = (
            (input - self.mean.array)
            / self.std.array
            * self.weight.array
            + self.bias.array
        )
        return output.swapaxes(2, 1)  # recover shape


#######
#
#   CRF
#
#######


class CRF(Module):
    __attributes__ = ['transitions']
    __annotations__ = {
        'transitions': Weight
    }

    def decode(self, emissions, mask):
        batch_size, seq_len, tags_num = emissions.shape
        emissions = emissions.swapaxes(1, 0)
        mask = mask.swapaxes(1, 0)

        history = []
        score = emissions[0]  # batch x tags
        for index in range(1, seq_len):
            score_ = score.reshape(batch_size, tags_num, 1)
            emissions_ = emissions[index].reshape(batch_size, 1, tags_num)
            score_ = score_ + self.transitions.array + emissions_  # batch x tags x tags
            indexes = score_.argmax(-2)
            # https://stackoverflow.com/questions/20128837/get-indices-of-numpy-argmax-elements-over-an-axis
            batch_indexes, tags_indexes = np.indices(indexes.shape)
            score_ = score_[batch_indexes, indexes, tags_indexes]

            mask_ = mask[index].reshape(batch_size, 1)
            score = np.where(mask_, score_, score)
            history.append(indexes)

        sizes = mask.sum(0) - 1
        batch = []
        for index in range(batch_size):
            best = score[index].argmax()
            tags = [best]
            size = sizes[index]
            for indexes in reversed(history[:size]):
                best = indexes[index][best]
                tags.append(best)
            tags.reverse()
            batch.append(np.array(tags))

        return batch


########
#
#   EMB
#
#######


class Embedding(Module):
    __attributes__ = ['weight']
    __annotations__ = {
        'weight': Weight
    }

    def __init__(self, weight):
        self.weight = weight
        _, self.dim = self.weight.shape

    def __call__(self, input):
        shape = input.shape
        input = input.flatten()
        weight = self.weight.array[input]
        return weight.reshape(*shape, self.dim)


class NavecEmbedding(Embedding):
    __attributes__ = ['id', 'indexes', 'codes']
    __annotations__ = {
        'indexes': Weight,
        'codes': Weight
    }

    def __init__(self, id, indexes, codes):
        self.id = id
        self.indexes = indexes
        self.codes = codes

        qdim, centroids, chunk = codes.shape
        self.dim = qdim * chunk
        self.qdims = np.arange(qdim)

    def __call__(self, input):
        shape = input.shape
        input = input.flatten()
        indexes = self.indexes.array[input]
        output = self.codes.array[self.qdims, indexes]
        return output.reshape(*shape, self.dim)


class WordShapeEmbedding(Module):
    __attributes__ = ['word', 'shape']
    __annotations__ = {
        'word': NavecEmbedding,
        'shape': Embedding
    }

    def __call__(self, word_id, shape_id):
        word = self.word(word_id)
        shape = self.shape(shape_id)
        return np.concatenate((word, shape), axis=-1)


######
#
#   CNN
#
#######


class CNNEncoderLayer(Module):
    __attributes__ = ['conv', 'relu', 'norm']
    __annotations__ = {
        'conv': Conv1d,
        'relu': ReLU,
        'norm': BatchNorm1d
    }

    def __call__(self, input):
        x = self.conv(input)
        x = self.relu(x)
        return self.norm(x)


class CNNEncoder(Module):
    __attributes__ = ['layers']
    __annotations__ = {
        'layers': [CNNEncoderLayer]
    }

    def __call__(self, input, mask):
        input = np.swapaxes(input, 2, 1)
        mask = np.expand_dims(mask, axis=1)

        for layer in self.layers:
            input = layer(input)
            size = input.shape[1]
            input[mask.repeat(size, axis=1)] = 0

        return np.swapaxes(input, 2, 1)


#######
#
#   TAG
#
######


class NERHead(Module):
    __attributes__ = ['proj', 'crf']
    __annotations__ = {
        'proj': Linear,
        'crf': CRF
    }

    def __call__(self, input):
        return self.proj(input)


class MorphHead(Module):
    __attributes__ = ['proj']
    __annotations__ = {
        'proj': Linear
    }

    def __call__(self, input):
        return self.proj(input)

    def decode(self, pred):
        return pred.argmax(-1)


class Tag(Module):
    __attributes__ = ['emb', 'encoder', 'head']

    def __call__(self, word_id, shape_id, pad_mask):
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, pad_mask)
        return self.head(x)


class NER(Tag):
    __annotations__ = {
        'emb': WordShapeEmbedding,
        'encoder': CNNEncoder,
        'head': NERHead
    }


class Morph(Tag):
    __annotations__ = {
        'emb': WordShapeEmbedding,
        'encoder': CNNEncoder,
        'head': MorphHead
    }


########
#
#  SYNTAX
#
######


class FF(Module):
    __attributes__ = ['proj', 'relu']
    __annotations__ = {
        'proj': Linear,
        'relu': ReLU
    }

    def __call__(self, input):
        x = self.proj(input)
        return self.relu(x)


def append_root(input, root):
    batch_size, _, emb_dim = input.shape
    root = np.tile(root, batch_size)
    root = root.reshape(batch_size, 1, emb_dim)
    return np.concatenate((root, input), axis=1)


def strip_root(input):
    return input[:, 1:, :]


def append_root_mask(mask):
    return np.pad(
        mask,
        [(0, 0), (1, 0)],  # no pad for batch, pad left seq
        mode='constant', constant_values=True
    )


def matmul_mask(mask):
    mask = np.expand_dims(mask, axis=-2)
    return np.matmul(mask.swapaxes(-2, -1), mask)


class SyntaxHead(Module):
    __attributes__ = ['head', 'tail', 'root', 'kernel']
    __annotations__ = {
        'head': FF,
        'tail': FF,
        'root': Weight,
        'kernel': Weight
    }

    def decode(self, pred, mask):
        mask = append_root_mask(mask)
        mask = matmul_mask(mask)
        mask = strip_root(mask)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def __call__(self, input):
        input = append_root(input, self.root.array)
        head = self.head(input)
        tail = self.tail(input)

        x = np.matmul(head, self.kernel.array)
        x = np.matmul(x, tail.swapaxes(-2, -1))
        return strip_root(x)


def gather_head(input, root, index):
    batch_size, seq_len, emb_dim = input.shape
    input = append_root(input, root)

    zero = np.zeros((batch_size, 1), dtype=np.int_)
    index = np.concatenate((zero, index), axis=-1)
    # flatten input, absolute indexing
    input = input.reshape(-1, emb_dim)  # batch * seq x dim
    offset = np.arange(batch_size) * (seq_len + 1)
    index = offset[np.newaxis].T + index
    input = input[index]

    return strip_root(input)


class SyntaxRel(Module):
    __attributes__ = ['head', 'tail', 'root', 'kernel']
    __annotations__ = {
        'head': FF,
        'tail': FF,
        'root': Weight,
        'kernel': Weight
    }

    def decode(self, pred, mask):
        _, _, rel_dim = pred.shape
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, rel_dim, axis=-1)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def __call__(self, input, head_id):
        head = self.head(gather_head(input, self.root.array, head_id))
        tail = self.tail(input)

        batch_size, seq_len, _ = input.shape
        hidden_dim, dim = self.kernel.shape
        rel_dim = dim // hidden_dim

        x = np.matmul(head, self.kernel.array)  # batch x seq x hidden * rel
        x = x.reshape(batch_size, seq_len, rel_dim, hidden_dim)
        x = np.matmul(x, tail.reshape(batch_size, seq_len, hidden_dim, 1))
        return x.reshape(batch_size, seq_len, rel_dim)


class SyntaxPred(Record):
    __attributes__ = ['head_id', 'rel_id']


class Syntax(Module):
    __attributes__ = ['emb', 'encoder', 'head', 'rel']
    __annotations__ = {
        'emb': WordShapeEmbedding,
        'encoder': CNNEncoder,
        'head': SyntaxHead,
        'rel': SyntaxRel
    }

    def __call__(self, word_id, shape_id, pad_mask):
        x = self.emb(word_id, shape_id)
        x = self.encoder(x, pad_mask)

        head_id = self.head(x)
        target_head_id = self.head.decode(head_id, ~pad_mask)
        rel_id = self.rel(x, target_head_id)
        return SyntaxPred(head_id, rel_id)


#######
#
#   VISITOR
#
######


class ModuleVisitor(Visitor):
    def visit_Weight(self, item):
        return item

    def visit_Module(self, item):
        args = []
        for key in item.__attributes__:
            value = getattr(item, key)
            annotation = item.__annotations__.get(key)
            if annotation and value is not None:
                _, repeatable, _ = parse_annotation(annotation)
                if repeatable:
                    value = [self.visit(_) for _ in value]
                else:
                    value = self.visit(value)
            args.append(value)
        return type(item)(*args)


class SeparateArraysVisitor(ModuleVisitor):
    def __init__(self):
        self.arrays = {}

    def visit_Weight(self, item):
        if item.is_empty:
            return item

        id = len(self.arrays)
        self.arrays[id] = item.array
        return item.replace(array=id)


class InjectArraysVisitor(ModuleVisitor):
    def __init__(self, arrays):
        self.arrays = arrays

    def visit_Weight(self, item):
        if not item.is_id:
            return item

        return item.replace(
            array=self.arrays[item.array]
        )


class StripNavecVisitor(ModuleVisitor):
    def visit_NavecEmbedding(self, item):
        return item.replace(
            indexes=item.indexes.empty(),
            codes=item.codes.empty()
        )


class InjectNavecVisitor(ModuleVisitor):
    def __init__(self, navec):
        self.navec = navec

    def visit_NavecEmbedding(self, item):
        id = self.navec.meta.id
        if item.id != id:
            raise ValueError('Expected id=%r, got %r' % (item.id, id))

        pq = self.navec.pq
        return item.replace(
            indexes=item.indexes.replace(array=pq.indexes),
            codes=item.codes.replace(array=pq.codes)
        )


class WeightsVisitor(ModuleVisitor):
    def __init__(self):
        self.weights = []

    def visit_Weight(self, item):
        self.weights.append(item)
        return item
