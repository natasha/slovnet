
from slovnet.record import Record
from slovnet.token import tokenize
from slovnet.markup import (
    BIOMarkup,
    MorphMarkup,
    SyntaxMarkup
)

from .mask import split_masked


class Infer(Record):
    __attributes__ = ['model', 'encoder', 'decoder']


######
#
#   TAG
#
#####


class TagDecoder(Record):
    __attributes__ = ['tags_vocab']

    def __call__(self, preds):
        for pred in preds:
            yield [self.tags_vocab.decode(_) for _ in pred]


def text_words(text):
    return [_.text for _ in tokenize(text)]


class NERInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            yield from self.model.head.crf.decode(pred, ~input.pad_mask)

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
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            pred = self.model.head.decode(pred)
            yield from split_masked(pred, ~input.pad_mask)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            tuples = zip(item, pred)
            yield MorphMarkup.from_tuples(tuples)


########
#
#   SYNTAX
#
######


class SyntaxDecoder(Record):
    __attributes__ = ['rels_vocab']

    def __call__(self, preds):
        for pred in preds:
            head_ids, rel_ids = pred
            ids = [str(_ + 1) for _ in range(len(head_ids))]
            head_ids = [str(_) for _ in head_ids.tolist()]
            rels = [self.rels_vocab.decode(_) for _ in rel_ids]
            yield ids, head_ids, rels


class SyntaxInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            mask = ~input.pad_mask

            head_id = self.model.head.decode(pred.head_id, mask)
            head_id = split_masked(head_id, mask)

            rel_id = self.model.rel.decode(pred.rel_id, mask)
            rel_id = split_masked(rel_id, mask)

            yield from zip(head_id, rel_id)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            ids, head_ids, rels = pred
            tuples = zip(ids, item, head_ids, rels)
            yield SyntaxMarkup.from_tuples(tuples)
