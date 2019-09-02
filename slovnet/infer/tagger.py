
import numpy as np

from slovnet.record import Record
from slovnet.bio import (
    PER, LOC, ORG,
    io_spans
)
from slovnet.shape import SHAPES
from slovnet.markup import SpanMarkup
from slovnet.tokenizer import Tokenizer
from slovnet.encoder import (
    StackEncoder,
    WordEncoder,
    ShapeEncoder
)
from slovnet.vocab import (
    WordsVocab,
    ShapesVocab,
    TagsVocab
)

from .impl import NavecEmbedding
from .pack import Pack


class NERTagger(Record):
    __attributes__ = ['tokenizer', 'token_encoder', 'tags_vocab', 'model']

    def __init__(self, tokenizer, token_encoder, tags_vocab, model):
        self.tokenizer = tokenizer
        self.token_encoder = token_encoder
        self.tags_vocab = tags_vocab
        self.model = model

    def __call__(self, text):
        tokens = list(self.tokenizer(text))

        ids = self.token_encoder.map(tokens)  # [feats1, feats2, ...]
        input = [np.array([_]) for _ in ids]

        pred = self.model(input)
        pred = pred.squeeze(0)
        pred = pred.tolist()

        tags = [self.tags_vocab.decode(_) for _ in pred]
        spans = list(io_spans(tokens, tags))  # in case of broken bio
        return SpanMarkup(text, spans)

    @classmethod
    def load(cls, path, navec):
        tokenizer = Tokenizer()

        words_vocab = WordsVocab(navec.vocab.words)
        shapes_vocab = ShapesVocab(SHAPES)
        token_encoder = StackEncoder([
            WordEncoder(words_vocab),
            ShapeEncoder(shapes_vocab)
        ])

        tags_vocab = TagsVocab([PER, LOC, ORG])

        pack = Pack.load(path)
        pack.context.navec = NavecEmbedding.from_navec(navec)
        model = pack.scheme.to_impl(pack.context)

        return cls(
            tokenizer,
            token_encoder,
            tags_vocab,
            model
        )
