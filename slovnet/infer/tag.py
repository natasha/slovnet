
from slovnet.mask import split_masked
from slovnet.token import tokenize
from slovnet.markup import (
    BIOMarkup,
    MorphMarkup
)

from .base import Infer


class TagDecoder:
    def __init__(self, tags_vocab):
        """
        Initialize vocab.

        Args:
            self: (todo): write your description
            tags_vocab: (todo): write your description
        """
        self.tags_vocab = tags_vocab

    def __call__(self, preds):
        """
        Yields all preds.

        Args:
            self: (todo): write your description
            preds: (array): write your description
        """
        for pred in preds:
            yield [self.tags_vocab.decode(_) for _ in pred]


def text_words(text):
    """
    Returns a list of words.

    Args:
        text: (str): write your description
    """
    return [_.text for _ in tokenize(text)]


class NERInfer(Infer):
    def process(self, inputs):
        """
        Iterate through - process.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            yield from self.model.ner.crf.decode(pred, ~input.pad_mask)

    def __call__(self, texts):
        """
        Yields tokens.

        Args:
            self: (todo): write your description
            texts: (str): write your description
        """
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
        """
        Process the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            pred = self.model.morph.decode(pred)
            yield from split_masked(pred, ~input.pad_mask)

    def __call__(self, items):
        """
        Iterate on items.

        Args:
            self: (todo): write your description
            items: (todo): write your description
        """
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            tuples = zip(item, pred)
            yield MorphMarkup.from_tuples(tuples)
