
from collections import OrderedDict

from slovnet.record import Record, JsonMixin
from slovnet.visitor import Visitor

from . import impl


def walk_subclasses(cls):
    for child in cls.__subclasses__():
        yield child
        for item in walk_subclasses(child):
            yield item


def parse_annotation(type):
    repeatable = False
    if isinstance(type, list):  # [ModuleScheme]
        repeatable = True
        type = type[0]
    return repeatable, type


def as_json(item):
    if isinstance(item, SchemeRecord):
        return item.as_json
    elif isinstance(item, list):
        return [as_json(_) for _ in item]
    else:
        return item


class SchemeRecord(Record, JsonMixin):
    __annotations__ = {}
    name = None

    @property
    def as_json(self):
        data = OrderedDict()
        if self.name:
            data['name'] = self.name

        for key in self.__attributes__:
            value = getattr(self, key)
            data[key] = as_json(value)

        return data

    @classmethod
    def from_json(cls, data):
        name = data.get('name')
        if name != cls.name:
            for child in walk_subclasses(cls):
                if name == child.name:
                    return child.from_json(data)
            raise ValueError(name)

        args = []
        for key in cls.__attributes__:
            annotation = cls.__annotations__.get(key)
            value = data[key]
            if annotation:
                repeatable, type = parse_annotation(annotation)
                if repeatable:
                    value = [type.from_json(_) for _ in value]
                else:
                    value = type.from_json(value)
            args.append(value)

        return cls(*args)


class WeightScheme(SchemeRecord):
    __attributes__ = ['id', 'shape']

    def __init__(self, id, shape):
        self.id = id
        self.shape = shape


class ModuleScheme(SchemeRecord):
    def to_impl(self, context):
        visitor = ImplVisitor(context)
        return visitor(self)


class LinearScheme(ModuleScheme):
    __attributes__ = ['weight', 'bias']
    __annotations__ = {
        'weight': WeightScheme,
        'bias': WeightScheme
    }
    name = 'linear'

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias


class Conv1dScheme(ModuleScheme):
    __attributes__ = ['weight', 'bias', 'padding']
    __annotations__ = {
        'weight': WeightScheme,
        'bias': WeightScheme
    }
    name = 'conv1d'

    def __init__(self, weight, bias, padding):
        self.weight = weight
        self.bias = bias
        self.padding = padding


class ReLUScheme(ModuleScheme):
    name = 'relu'


class BatchNorm1dScheme(ModuleScheme):
    __attributes__ = ['weight', 'bias', 'mean', 'std']
    __annotations__ = {
        'weight': WeightScheme,
        'bias': WeightScheme,
        'mean': WeightScheme,
        'std': WeightScheme
    }
    name = 'batch_norm1d'

    def __init__(self, weight, bias, mean, std):
        self.weight = weight
        self.bias = bias
        self.mean = mean
        self.std = std


class EmbeddingScheme(ModuleScheme):
    __attributes__ = ['weight']
    __annotations__ = {
        'weight': WeightScheme
    }
    name = 'embedding'

    def __init__(self, weight):
        self.weight = weight


class StackEmbeddingScheme(EmbeddingScheme):
    __attributes__ = ['embs']
    __annotations__ = {
        'embs': [EmbeddingScheme]
    }
    name = 'stack_embedding'

    def __init__(self, embs):
        self.embs = embs


class NavecEmbeddingScheme(EmbeddingScheme):
    __attributes__ = ['id']
    name = 'navec_embedding'

    def __init__(self, id):
        self.id = id


class TransposeScheme(ModuleScheme):
    __attributes__ = ['axis1', 'axis2']
    name = 'transpose'

    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2


class SequentialScheme(ModuleScheme):
    __attributes__ = ['modules']
    __annotations__ = {
        'modules': [ModuleScheme]
    }
    name = 'sequential'

    def __init__(self, modules):
        self.modules = modules


class CRFScheme(ModuleScheme):
    __attributes__ = ['transitions']
    __annotations__ = {
        'transitions': WeightScheme
    }
    name = 'crf'

    def __init__(self, transitions):
        self.transitions = transitions


class Context(Record):
    __attributes__ = ['navec', 'weights']

    def __init__(self, navec=None, weights=None):
        self.navec = navec
        if not weights:
            weights = {}
        self.weights = weights


class SchemeVisitor(Visitor):
    def __init__(self):
        self.context = Context()

    def visit_Weight(self, item):
        id = len(self.context.weights)
        self.context.weights[id] = item.flatten()
        return WeightScheme(id, item.shape)

    def visit_Transpose(self, item):
        return TransposeScheme(
            item.axis1,
            item.axis2
        )

    def visit_Sequential(self, item):
        return SequentialScheme([
            self.visit(_)
            for _ in item.modules
        ])

    def visit_Linear(self, item):
        return LinearScheme(
            self.visit(item.weight),
            self.visit(item.bias)
        )

    def visit_Conv1d(self, item):
        return Conv1dScheme(
            self.visit(item.weight),
            self.visit(item.bias),
            item.padding
        )

    def visit_ReLU(self, item):
        return ReLUScheme()

    def visit_BatchNorm1d(self, item):
        return BatchNorm1dScheme(
            self.visit(item.weight),
            self.visit(item.bias),
            self.visit(item.mean),
            self.visit(item.std),
        )

    def visit_Embedding(self, item):
        return EmbeddingScheme(
            self.visit(item.weight)
        )

    def visit_StackEmbedding(self, item):
        return StackEmbeddingScheme([
            self.visit(_)
            for _ in item.embs
        ])

    def visit_NavecEmbedding(self, item):
        self.context.navec = item
        return NavecEmbeddingScheme(item.id)

    def visit_CRF(self, item):
        return CRFScheme(
            self.visit(item.transitions)
        )


class ImplVisitor(Visitor):
    def __init__(self, context):
        self.context = context

    def visit_WeightScheme(self, item):
        weight = self.context.weights.get(item.id)
        if weight is None:
            raise KeyError('weight %r not found in context' % item.id)
        return weight.reshape(item.shape)

    def visit_TransposeScheme(self, item):
        return impl.Transpose(
            item.axis1,
            item.axis2
        )

    def visit_SequentialScheme(self, item):
        return impl.Sequential([
            self.visit(_)
            for _ in item.modules
        ])

    def visit_LinearScheme(self, item):
        return impl.Linear(
            self.visit(item.weight),
            self.visit(item.bias)
        )

    def visit_Conv1dScheme(self, item):
        return impl.Conv1d(
            self.visit(item.weight),
            self.visit(item.bias),
            item.padding
        )

    def visit_ReLUScheme(self, item):
        return impl.ReLU()

    def visit_BatchNorm1dScheme(self, item):
        return impl.BatchNorm1d(
            self.visit(item.weight),
            self.visit(item.bias),
            self.visit(item.mean),
            self.visit(item.std),
        )

    def visit_EmbeddingScheme(self, item):
        return impl.Embedding(
            self.visit(item.weight)
        )

    def visit_StackEmbeddingScheme(self, item):
        return impl.StackEmbedding([
            self.visit(_)
            for _ in item.embs
        ])

    def visit_NavecEmbeddingScheme(self, item):
        if not self.context.navec:
            raise ValueError('no navec in context')

        if item.id != self.context.navec.id:
            raise ValueError('expected navec id {expected!r}, got {got!r}'.format(
                expected=item.id,
                got=self.context.navec.id
            ))

        return self.context.navec

    def visit_CRFScheme(self, item):
        return impl.CRF(
            self.visit(item.transitions)
        )
