
from .record import Record
from .const import WORD, SHAPE, TAG, REL
from .chop import chop

from .exec.pack import Pack
from .exec.model import (
    Morph as MorphModel,
    NER as NERModel,
    Syntax as SyntaxModel
)
from .exec.encoders import (
    TagEncoder,
    SyntaxEncoder
)
from .exec.infer import (
    TagDecoder,
    MorphInfer,
    NERInfer,

    SyntaxDecoder,
    SyntaxInfer
)


class API(Record):
    __attributes__ = ['infer', 'batch_size']

    def navec(self, navec):
        self.infer.model = self.infer.model.inject_navec(navec)
        return self

    def map(self, items):
        for chunk in chop(items, self.batch_size):
            yield from self.infer(chunk)

    def __call__(self, item):
        return next(self.map([item]))


class NER(API):
    @classmethod
    def load(cls, path, batch_size=8):
        with Pack(path) as pack:
            meta = pack.load_meta()
            meta.check_protocol()

            model = pack.load_model(NERModel)
            arrays = dict(pack.load_arrays(model.weights))

            words_vocab = pack.load_vocab(WORD)
            shapes_vocab = pack.load_vocab(SHAPE)
            tags_vocab = pack.load_vocab(TAG)

        model = model.inject_arrays(arrays)
        encoder = TagEncoder(
            words_vocab, shapes_vocab,
            batch_size
        )
        decoder = TagDecoder(tags_vocab)
        infer = NERInfer(model, encoder, decoder)

        return cls(infer, batch_size)


class Morph(API):
    @classmethod
    def load(cls, path, batch_size=8):
        with Pack(path) as pack:
            meta = pack.load_meta()
            meta.check_protocol()

            model = pack.load_model(MorphModel)
            arrays = dict(pack.load_arrays(model.weights))

            words_vocab = pack.load_vocab(WORD)
            shapes_vocab = pack.load_vocab(SHAPE)
            tags_vocab = pack.load_vocab(TAG)

        model = model.inject_arrays(arrays)
        encoder = TagEncoder(
            words_vocab, shapes_vocab,
            batch_size
        )
        decoder = TagDecoder(tags_vocab)
        infer = MorphInfer(model, encoder, decoder)

        return cls(infer, batch_size)


class Syntax(API):
    @classmethod
    def load(cls, path, batch_size=8):
        with Pack(path) as pack:
            meta = pack.load_meta()
            meta.check_protocol()

            model = pack.load_model(SyntaxModel)
            arrays = dict(pack.load_arrays(model.weights))

            words_vocab = pack.load_vocab(WORD)
            shapes_vocab = pack.load_vocab(SHAPE)
            rels_vocab = pack.load_vocab(REL)

        model = model.inject_arrays(arrays)
        encoder = SyntaxEncoder(
            words_vocab, shapes_vocab,
            batch_size
        )
        decoder = SyntaxDecoder(rels_vocab)
        infer = SyntaxInfer(model, encoder, decoder)

        return cls(infer, batch_size)
