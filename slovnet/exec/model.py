
import numpy as np

from slovnet.record import Record, parse_annotation
from slovnet.visitor import Visitor

from .mask import fill_masked


class Weight(Record):
    __attributes__ = ['shape', 'dtype', 'array']

    def empty(self):
        """
        Return an empty array with all empty values.

        Args:
            self: (todo): write your description
        """
        return self.replace(array=None)

    @property
    def is_empty(self):
        """
        Returns true if all empty array is empty.

        Args:
            self: (todo): write your description
        """
        return self.array is None

    @property
    def is_id(self):
        """
        Return the id of the type : int.

        Args:
            self: (todo): write your description
        """
        return type(self.array) is int


class Module(Record):
    def separate_arrays(self):
        """
        Return a new arrays.

        Args:
            self: (todo): write your description
        """
        visitor = SeparateArraysVisitor()
        scheme = visitor(self)
        return visitor.arrays, scheme

    def inject_arrays(self, arrays):
        """
        Inject arrays injects.

        Args:
            self: (todo): write your description
            arrays: (array): write your description
        """
        visitor = InjectArraysVisitor(arrays)
        return visitor(self)

    def strip_navec(self):
        """
        Strips the nave.

        Args:
            self: (todo): write your description
        """
        visitor = StripNavecVisitor()
        return visitor(self)

    def inject_navec(self, navec):
        """
        Injects a nave.

        Args:
            self: (todo): write your description
            navec: (str): write your description
        """
        visitor = InjectNavecVisitor(navec)
        return visitor(self)

    @property
    def weights(self):
        """
        The weights of weights.

        Args:
            self: (todo): write your description
        """
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
        """
        Initialize weights.

        Args:
            self: (todo): write your description
            weight: (int): write your description
            bias: (float): write your description
        """
        self.weight = weight
        self.bias = bias
        self.in_dim, self.out_dim = self.weight.shape

    def __call__(self, input):
        """
        Implement self ( self input ).

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
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
        """
        Private function.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
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
        """
        Calls self. call on self.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
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
        """
        Return the mean value ).

        Args:
            self: (todo): write your description
            input: (todo): write your description
        """
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
        """
        Decode a batch of sequences from a batch of sentences.

        Args:
            self: (todo): write your description
            emissions: (todo): write your description
            mask: (todo): write your description
        """
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
        """
        Initialize weights.

        Args:
            self: (todo): write your description
            weight: (int): write your description
        """
        self.weight = weight
        _, self.dim = self.weight.shape

    def __call__(self, input):
        """
        Todo : py : meth : ndarray.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
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
        """
        Initialize a qdims.

        Args:
            self: (todo): write your description
            id: (str): write your description
            indexes: (str): write your description
            codes: (array): write your description
        """
        self.id = id
        self.indexes = indexes
        self.codes = codes

        qdim, centroids, chunk = codes.shape
        self.dim = qdim * chunk
        self.qdims = np.arange(qdim)

    def __call__(self, input):
        """
        Call the callable.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
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
        """
        Returns the shape_id

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
        """
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
        """
        Implement operator.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        x = self.conv(input)
        x = self.relu(x)
        return self.norm(x)


class CNNEncoder(Module):
    __attributes__ = ['layers']
    __annotations__ = {
        'layers': [CNNEncoderLayer]
    }

    def __call__(self, input, mask):
        """
        Call the network.

        Args:
            self: (todo): write your description
            input: (array): write your description
            mask: (array): write your description
        """
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
        """
        Calls the method.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        return self.proj(input)


class MorphHead(Module):
    __attributes__ = ['proj']
    __annotations__ = {
        'proj': Linear
    }

    def __call__(self, input):
        """
        Calls the method.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        return self.proj(input)

    def decode(self, pred):
        """
        Decode the given predicate.

        Args:
            self: (todo): write your description
            pred: (array): write your description
        """
        return pred.argmax(-1)


class Tag(Module):
    __attributes__ = ['emb', 'encoder', 'head']

    def __call__(self, word_id, shape_id, pad_mask):
        """
        Parameters ---------- word_id : string.

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
            pad_mask: (array): write your description
        """
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
        """
        Implement self.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        x = self.proj(input)
        return self.relu(x)


def append_root(input, root):
    """
    Append a root element.

    Args:
        input: (todo): write your description
        root: (array): write your description
    """
    batch_size, _, emb_dim = input.shape
    root = np.tile(root, batch_size)
    root = root.reshape(batch_size, 1, emb_dim)
    return np.concatenate((root, input), axis=1)


def strip_root(input):
    """
    Strips the root of a string.

    Args:
        input: (todo): write your description
    """
    return input[:, 1:, :]


def append_root_mask(mask):
    """
    Append a mask.

    Args:
        mask: (array): write your description
    """
    return np.pad(
        mask,
        [(0, 0), (1, 0)],  # no pad for batch, pad left seq
        mode='constant', constant_values=True
    )


def matmul_mask(mask):
    """
    Swmulax mask.

    Args:
        mask: (array): write your description
    """
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
        """
        Decode a mask.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            mask: (array): write your description
        """
        mask = append_root_mask(mask)
        mask = matmul_mask(mask)
        mask = strip_root(mask)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def __call__(self, input):
        """
        Call the kernel matrix.

        Args:
            self: (todo): write your description
            input: (array): write your description
        """
        input = append_root(input, self.root.array)
        head = self.head(input)
        tail = self.tail(input)

        x = np.matmul(head, self.kernel.array)
        x = np.matmul(x, tail.swapaxes(-2, -1))
        return strip_root(x)


def gather_head(input, root, index):
    """
    Gather the first n elements in the array.

    Args:
        input: (array): write your description
        root: (todo): write your description
        index: (int): write your description
    """
    batch_size, seq_len, emb_dim = input.shape
    input = append_root(input, root)

    zero = np.zeros((batch_size, 1), dtype=np.long)
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
        """
        Decode the masked mask.

        Args:
            self: (todo): write your description
            pred: (todo): write your description
            mask: (array): write your description
        """
        _, _, rel_dim = pred.shape
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, rel_dim, axis=-1)

        pred = fill_masked(pred, ~mask, pred.min())
        return pred.argmax(-1)

    def __call__(self, input, head_id):
        """
        Call a batch.

        Args:
            self: (todo): write your description
            input: (array): write your description
            head_id: (str): write your description
        """
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
        """
        Parameters ---------- word_id : string.

        Args:
            self: (todo): write your description
            word_id: (str): write your description
            shape_id: (str): write your description
            pad_mask: (array): write your description
        """
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
        """
        Determine if item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return item

    def visit_Module(self, item):
        """
        Handles an ast.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
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
        """
        Initialize the internal state.

        Args:
            self: (todo): write your description
        """
        self.arrays = {}

    def visit_Weight(self, item):
        """
        Return an array corresponding to item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        if item.is_empty:
            return item

        id = len(self.arrays)
        self.arrays[id] = item.array
        return item.replace(array=id)


class InjectArraysVisitor(ModuleVisitor):
    def __init__(self, arrays):
        """
        Initialize arrays.

        Args:
            self: (todo): write your description
            arrays: (array): write your description
        """
        self.arrays = arrays

    def visit_Weight(self, item):
        """
        Convert an array as an array.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        if not item.is_id:
            return item

        return item.replace(
            array=self.arrays[item.array]
        )


class StripNavecVisitor(ModuleVisitor):
    def visit_NavecEmbedding(self, item):
        """
        Return the nave of a nave.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return item.replace(
            indexes=item.indexes.empty(),
            codes=item.codes.empty()
        )


class InjectNavecVisitor(ModuleVisitor):
    def __init__(self, navec):
        """
        Åīľ´æĸ°

        Args:
            self: (todo): write your description
            navec: (todo): write your description
        """
        self.navec = navec

    def visit_NavecEmbedding(self, item):
        """
        Create a naveingcEmbing item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
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
        """
        Initialize weights

        Args:
            self: (todo): write your description
        """
        self.weights = []

    def visit_Weight(self, item):
        """
        Add a new item to the list.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        self.weights.append(item)
        return item
