
from .const import WORD, SHAPE, TAG

from .exec.pack import Pack
from .exec.model import (
    Morph as MorphModel,
    NER as NERModel
)
from .exec.encoders import TagEncoder
from .exec.infer import (
    TagDecoder,
    MorphInfer,
    NERInfer
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
