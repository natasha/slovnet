
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
        """
        Initialize the protocol.

        Args:
            self: (todo): write your description
            id: (str): write your description
            protocol: (todo): write your description
            PROTOCOL: (todo): write your description
        """
        self.id = id
        self.protocol = protocol

    def check_protocol(self):
        """
        Check if the protocol.

        Args:
            self: (todo): write your description
        """
        if self.protocol != PROTOCOL:
            raise ValueError('Expected protocol=%r, got %r' % (PROTOCOL, self.protocol))


#######
#
#  ARRAY
#
#######


def array_name(id):
    """
    Return an array name of an array.

    Args:
        id: (array): write your description
    """
    return 'arrays/%d.bin' % id


def array_bytes(array):
    """
    Return an array array as bytes.

    Args:
        array: (array): write your description
    """
    return array.tobytes()


def bytes_array(bytes, shape, dtype):
    """
    Convert a numpy array tobuffer.

    Args:
        bytes: (todo): write your description
        shape: (int): write your description
        dtype: (todo): write your description
    """
    return np.frombuffer(bytes, dtype).reshape(shape)


######
#
#  VOCAB
#
#######


def vocab_name(id):
    """
    Returns the name of a vocabulary.

    Args:
        id: (int): write your description
    """
    return 'vocabs/%s.gz' % id


def vocab_bytes(vocab):
    """
    Convert the vocab to vocab.

    Args:
        vocab: (todo): write your description
    """
    content = '\n'.join(vocab.items)
    bytes = content.encode('utf8')
    return compress(bytes)


def bytes_vocab(bytes):
    """
    Decompress a byte string.

    Args:
        bytes: (todo): write your description
    """
    content = decompress(bytes).decode('utf8')
    items = content.splitlines()
    return Vocab(items)


######
#
#  PACK
#
########


def json_bytes(data):
    """
    Encode the given data to bytes.

    Args:
        data: (array): write your description
    """
    content = json.dumps(data, ensure_ascii=False, indent=2)
    return content.encode('utf8')


def bytes_json(bytes):
    """
    Convert bytes to bytes

    Args:
        bytes: (todo): write your description
    """
    return json.loads(bytes.decode('utf8'))


class Pack(Tar):
    def load_record(self, name, Record):
        """
        Load a record.

        Args:
            self: (todo): write your description
            name: (str): write your description
            Record: (todo): write your description
        """
        bytes = self.read(name)
        data = bytes_json(bytes)
        return Record.from_json(data)

    def load_meta(self):
        """
        Load meta data.

        Args:
            self: (todo): write your description
        """
        return self.load_record(META, Meta)

    def load_model(self, Model):
        """
        Load a model.

        Args:
            self: (todo): write your description
            Model: (todo): write your description
        """
        return self.load_record(MODEL, Model)

    def load_arrays(self, weights):
        """
        Loads all the arrays.

        Args:
            self: (todo): write your description
            weights: (array): write your description
        """
        for weight in weights:
            if not weight.is_id:
                continue

            shape, dtype, id = weight
            name = array_name(id)
            bytes = self.read(name)
            yield id, bytes_array(bytes, shape, dtype)

    def load_vocab(self, id):
        """
        Load a vocabulary from the given id.

        Args:
            self: (todo): write your description
            id: (str): write your description
        """
        name = vocab_name(id)
        bytes = self.read(name)
        return bytes_vocab(bytes)


class DumpPack(DumpTar):
    def dump_record(self, record, name):
        """
        Write a record to file.

        Args:
            self: (todo): write your description
            record: (todo): write your description
            name: (str): write your description
        """
        bytes = json_bytes(record.as_json)
        self.write(bytes, name)

    def dump_meta(self, meta):
        """
        Dump meta data.

        Args:
            self: (todo): write your description
            meta: (str): write your description
        """
        self.dump_record(meta, META)

    def dump_model(self, model):
        """
        Dump a model.

        Args:
            self: (todo): write your description
            model: (todo): write your description
        """
        self.dump_record(model, MODEL)

    def dump_arrays(self, arrays):
        """
        Writes the arrays to the given arrays.

        Args:
            self: (todo): write your description
            arrays: (dict): write your description
        """
        for id, array in arrays.items():
            name = array_name(id)
            bytes = array_bytes(array)
            self.write(bytes, name)

    def dump_vocab(self, vocab, id):
        """
        Write vocab to a file.

        Args:
            self: (todo): write your description
            vocab: (todo): write your description
            id: (int): write your description
        """
        name = vocab_name(id)
        bytes = vocab_bytes(vocab)
        self.write(bytes, name)
