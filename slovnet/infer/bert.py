
from itertools import groupby

from slovnet.record import Record
from slovnet.token import tokenize
from slovnet.bert import bert_subs
from slovnet.chop import chop_weighted
from slovnet.mask import pad_masked, split_masked
from slovnet.markup import (
    BIOMarkup,
    MorphMarkup,
    SyntaxMarkup
)


##########
#
#   SEGMENT
#
#########


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


def substoken(text, vocab):
    subs = bert_subs(text, vocab)
    return SubsToken(text, subs)


def text_items(texts, vocab):
    for id, text in enumerate(texts):
        tokens = [
            substoken(_.text, vocab)
            for _ in tokenize(text)
        ]
        yield BERTInferItem(id, tokens)


def word_items(items, vocab):
    for id, words in enumerate(items):
        tokens = [
            substoken(_, vocab)
            for _ in words
        ]
        yield BERTInferItem(id, tokens)


def segment_items(items, seq_len):
    for item in items:
        chunks = chop_weighted(
            item.tokens,
            seq_len,
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


#######
#
#  DECODER
#
######


class BERTTagsDecoder:
    def __init__(self, tags_vocab):
        self.tags_vocab = tags_vocab

    def __call__(self, preds):
        for pred in preds:
            yield [self.tags_vocab.decode(_) for _ in pred]


class BERTSyntaxDecoder:
    def __init__(self, rels_vocab):
        self.rels_vocab = rels_vocab

    def __call__(self, preds):
        for pred in preds:
            head_ids, rel_ids = pred
            ids = [str(_ + 1) for _ in range(len(head_ids))]
            head_ids = [str(_) for _ in head_ids.tolist()]
            rels = [self.rels_vocab.decode(_) for _ in rel_ids]
            yield ids, head_ids, rels


#######
#
#   INFER
#
######


class BERTInfer:
    def __init__(self, model, encoder, decoder):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder


class BERTNERInfer(BERTInfer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.pad_mask)
            pred = pad_masked(pred, input.word_mask)
            mask = pad_masked(input.word_mask, input.word_mask)
            yield from self.model.ner.crf.decode(pred, mask)

    def __call__(self, chunk):
        items = text_items(chunk, self.encoder.words_vocab)
        # consider <cls>, <sep> spec tokens
        items = list(segment_items(items, self.encoder.seq_len - 2))
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            item.pred = pred

        items = join_items(items)

        for text, item in zip(chunk, items):
            tuples = zip(item.words, item.pred)
            markup = BIOMarkup.from_tuples(tuples)
            yield markup.to_span(text)


class BERTMorphInfer(BERTInfer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.pad_mask)
            pred = self.model.morph.decode(pred)
            yield from split_masked(pred, input.word_mask)

    def __call__(self, chunk):
        items = word_items(chunk, self.encoder.words_vocab)
        items = list(segment_items(items, self.encoder.seq_len - 2))
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            item.pred = pred

        items = join_items(items)
        for item in items:
            tuples = zip(item.words, item.pred)
            yield MorphMarkup.from_tuples(tuples)


def check_syntax_items(items, seq_len):
    sizes = {len(_.tokens) for _ in items}
    if len(sizes) != 1:
        raise ValueError('expected same size, got %r' % sorted(sizes))

    for item in items:
        size = sum(len(_.subs) for _ in item.tokens)
        if size > seq_len:
            raise ValueError('expected size <= %r, got' % (seq_len, size))


class BERTSyntaxInfer(BERTInfer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.word_mask, input.pad_mask)
            head_id = self.model.head.decode(pred.head_id)
            rel_id = self.model.rel.decode(pred.rel_id)
            yield from zip(head_id, rel_id)

    def __call__(self, chunk):
        items = list(word_items(chunk, self.encoder.words_vocab))
        check_syntax_items(items, self.encoder.seq_len - 2)
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            item.pred = pred

        items = join_items(items)
        for item in items:
            ids, head_ids, rels = item.pred
            tuples = zip(ids, item.words, head_ids, rels)
            yield SyntaxMarkup.from_tuples(tuples)
