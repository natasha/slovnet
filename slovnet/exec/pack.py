
import json
from gzip import (
    compress,
    decompress
)

import numpy as np

from slovnet.record import Record
from slovnet.tar import Tar, DumpTar
from slovnet.vocab import Vocab


PROTOCOL = 1

META = 'meta.json'
MODEL = 'model.json'


class Meta(Record):
    __attributes__ = ['id', 'protocol']

    def __init__(self, id, protocol=PROTOCOL):
        self.id = id
        self.protocol = protocol

    def check_protocol(self):
        if self.protocol != PROTOCOL:
            raise ValueError('Expected protocol=%r, got %r' % (PROTOCOL, self.protocol))


#######
#
#  ARRAY
#
#######


def array_name(id):
    return 'arrays/%d.bin' % id


def array_bytes(array):
    return array.tobytes()


def bytes_array(bytes, shape, dtype):
    return np.frombuffer(bytes, dtype).reshape(shape)


######
#
#  VOCAB
#
#######


def vocab_name(id):
    return 'vocabs/%s.gz' % id


def vocab_bytes(vocab):
    content = '\n'.join(vocab.items)
    bytes = content.encode('utf8')
    return compress(bytes)


def bytes_vocab(bytes):
    content = decompress(bytes).decode('utf8')
    items = content.splitlines()
    return Vocab(items)


######
#
#  PACK
#
########


def json_bytes(data):
    content = json.dumps(data, ensure_ascii=False, indent=2)
    return content.encode('utf8')


def bytes_json(bytes):
    return json.loads(bytes.decode('utf8'))


class Pack(Tar):
    def load_record(self, name, Record):
        bytes = self.read(name)
        data = bytes_json(bytes)
        return Record.from_json(data)

    def load_meta(self):
        return self.load_record(META, Meta)

    def load_model(self, Model):
        return self.load_record(MODEL, Model)

    def load_arrays(self, weights):
        for weight in weights:
            if not weight.is_id:
                continue

            shape, dtype, id = weight
            name = array_name(id)
            bytes = self.read(name)
            yield id, bytes_array(bytes, shape, dtype)

    def load_vocab(self, id):
        name = vocab_name(id)
        bytes = self.read(name)
        return bytes_vocab(bytes)


class DumpPack(DumpTar):
    def dump_record(self, record, name):
        bytes = json_bytes(record.as_json)
        self.write(bytes, name)

    def dump_meta(self, meta):
        self.dump_record(meta, META)

    def dump_model(self, model):
        self.dump_record(model, MODEL)

    def dump_arrays(self, arrays):
        for id, array in arrays.items():
            name = array_name(id)
            bytes = array_bytes(array)
            self.write(bytes, name)

    def dump_vocab(self, vocab, id):
        name = vocab_name(id)
        bytes = vocab_bytes(vocab)
        self.write(bytes, name)
