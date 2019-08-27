
import re
from collections import OrderedDict

from slovnet.record import Record, JsonMixin
from slovnet.tar import Tar, DumpTar

from .impl import Weight
from .scheme import (
    ModuleScheme,
    Context
)


PROTOCOL = 1

META = 'meta.json'
SCHEME = 'scheme.json'
WEIGHTS = 'weights'


def weight_path(id):
    return 'weights/%d.bin' % id


def parse_weight_path(path):
    match = re.match('^weights/(\d+).bin$', path)
    return int(match.group(1))


class Meta(Record, JsonMixin):
    __attributes__ = ['id', 'protocol']

    def __init__(self, id, protocol=PROTOCOL):
        self.id = id
        self.protocol = protocol

    def check_protocol(self):
        if self.protocol != PROTOCOL:
            raise ValueError('Expected protocol=%d, got %d' % (PROTOCOL, self.protocol))

    @property
    def as_json(self):
        data = OrderedDict()
        data['id'] = self.id
        data['protocol'] = self.protocol
        return data

    @classmethod
    def from_json(cls, data):
        return cls(
            data['id'],
            data['protocol']
        )


class Pack(Record):
    __attributes__ = ['meta', 'scheme', 'context']

    def __init__(self, meta, scheme, context):
        self.meta = meta
        self.scheme = scheme
        self.context = context

    def dump(self, path):
        with DumpTar(path) as tar:
            tar.write(self.meta.as_bytes, META)
            tar.write(self.scheme.as_bytes, SCHEME)
            for id, weight in self.context.weights.items():
                tar.write(weight.as_bytes, weight_path(id))

    @classmethod
    def load(cls, path):
        with Tar(path) as tar:
            file = tar.open(META)
            meta = Meta.from_file(file)
            meta.check_protocol()

            file = tar.open(SCHEME)
            scheme = ModuleScheme.from_file(file)

            context = Context()
            for path in tar.list(WEIGHTS):
                id = parse_weight_path(path)
                file = tar.open(path)
                weight = Weight.from_file(file)
                context.weights[id] = weight

            return cls(meta, scheme, context)
