
from .const import WORD, SHAPE, TAG, REL

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


def NER(path, batch_size=8):
    with Pack(path) as pack:
        meta = pack.load_meta()
        meta.check_protocol()

        model = pack.load_model(NERModel)
        arrays = dict(pack.load_arrays(model.weights))

        words_vocab = pack.load_vocab(WORD)
        shapes_vocab = pack.load_vocab(SHAPE)
        tags_vocab = pack.load_vocab(TAG)

    model = model.impl(arrays)
    encoder = TagEncoder(
        words_vocab, shapes_vocab,
        batch_size
    )
    decoder = TagDecoder(tags_vocab)

    return NERInfer(model, encoder, decoder)


def Morph(path, batch_size=8):
    with Pack(path) as pack:
        meta = pack.load_meta()
        meta.check_protocol()

        model = pack.load_model(MorphModel)
        arrays = dict(pack.load_arrays(model.weights))

        words_vocab = pack.load_vocab(WORD)
        shapes_vocab = pack.load_vocab(SHAPE)
        tags_vocab = pack.load_vocab(TAG)

    model = model.impl(arrays)
    encoder = TagEncoder(
        words_vocab, shapes_vocab,
        batch_size
    )
    decoder = TagDecoder(tags_vocab)

    return MorphInfer(model, encoder, decoder)


def Syntax(path, batch_size=8):
    with Pack(path) as pack:
        meta = pack.load_meta()
        meta.check_protocol()

        model = pack.load_model(SyntaxModel)
        arrays = dict(pack.load_arrays(model.weights))

        words_vocab = pack.load_vocab(WORD)
        shapes_vocab = pack.load_vocab(SHAPE)
        rels_vocab = pack.load_vocab(REL)

    model = model.impl(arrays)
    encoder = SyntaxEncoder(
        words_vocab, shapes_vocab,
        batch_size
    )
    decoder = SyntaxDecoder(rels_vocab)

    return SyntaxInfer(model, encoder, decoder)
