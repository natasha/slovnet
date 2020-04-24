
from slovnet.mask import split_masked
from slovnet.token import tokenize
from slovnet.markup import (
    BIOMarkup,
    MorphMarkup
)

from .base import Infer


class TagDecoder:
    def __init__(self, tags_vocab):
        self.tags_vocab = tags_vocab

    def __call__(self, preds):
        for pred in preds:
            yield [self.tags_vocab.decode(_) for _ in pred]


def text_words(text):
    return [_.text for _ in tokenize(text)]


class NERInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            yield from self.model.ner.crf.decode(pred, ~input.pad_mask)

    def __call__(self, texts):
        items = [text_words(_) for _ in texts]
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for text, item, pred in zip(texts, items, preds):
            tuples = zip(item, pred)
            markup = BIOMarkup.from_tuples(tuples)
            yield markup.to_span(text)


class MorphInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            pred = self.model.morph.decode(pred)
            yield from split_masked(pred, ~input.pad_mask)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            tuples = zip(item, pred)
            yield MorphMarkup.from_tuples(tuples)
