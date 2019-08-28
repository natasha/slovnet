
import numpy as np

from slovnet.record import Record
from slovnet.bio import io_spans
from slovnet.markup import SpanMarkup


class Tagger(Record):
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
        pred = pred.squeeze()
        pred = pred.tolist()

        tags = [self.tags_vocab.decode(_) for _ in pred]
        spans = list(io_spans(tokens, tags))  # in case of broken bio
        return SpanMarkup(text, spans)
