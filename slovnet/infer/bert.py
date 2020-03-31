
from itertools import groupby

import torch

from slovnet.record import Record
from slovnet.token import tokenize
from slovnet.bert import bert_subs
from slovnet.chop import chop_weighted
from slovnet.mask import pad_masked
from slovnet.markup import BIOMarkup


class SubsToken(Record):
    __attributes__ = ['text', 'subs']


class BERTInferItem(Record):
    __attributes__ = ['id', 'tokens', 'pred']

    def __init__(self, id, tokens, pred=None):
        self.id = id
        self.tokens = tokens
        self.pred = pred

    @property
    def words(self):
        return [_.text for _ in self.tokens]


def substoken(token, vocab):
    subs = bert_subs(token.text, vocab)
    return SubsToken(token.text, subs)


def text_items(texts, vocab):
    for id, text in enumerate(texts):
        tokens = [
            substoken(_, vocab)
            for _ in tokenize(text)
        ]
        yield BERTInferItem(id, tokens)


def segment_items(items, seq_len):
    for item in items:
        chunks = chop_weighted(
            item.tokens,
            seq_len - 2,  # consider <cls>, <sep> spec tokens
            weight=lambda _: len(_.subs)
        )
        for chunk in chunks:
            yield BERTInferItem(item.id, chunk)


def flatten(seqs):
    return [
        item
        for seq in seqs
        for item in seq
    ]


def join_items(items):
    for id, group in groupby(items, key=lambda _: _.id):
        group = list(group)
        tokens = flatten(_.tokens for _ in group)
        pred = flatten(_.pred for _ in group)
        yield BERTInferItem(id, tokens, pred)


class BERTNERInfer:
    def __init__(self, model, encoder):
        self.model = model
        self.encoder = encoder

    def process(self, inputs):
        with torch.no_grad():
            for input in inputs:
                pred = self.model(input.word_id, input.pad_mask)
                pred = pad_masked(pred, input.word_mask)
                mask = pad_masked(input.word_mask, input.word_mask)
                yield from self.model.ner.crf.decode(pred, mask)

    def __call__(self, texts):
        items = text_items(texts, self.encoder.words_vocab)
        items = list(segment_items(items, self.encoder.seq_len))
        inputs = self.encoder.encode(items)
        preds = self.process(inputs)
        preds = self.encoder.decode(preds)

        for item, pred in zip(items, preds):
            item.pred = pred

        items = join_items(items)

        for text, item in zip(texts, items):
            pairs = zip(item.words, item.pred)
            markup = BIOMarkup.from_pairs(pairs)
            yield markup.to_span(text)
