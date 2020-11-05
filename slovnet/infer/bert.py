
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

from .base import Infer
from .tag import TagDecoder
from .syntax import SyntaxDecoder


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
        """
        Initialize a new token.

        Args:
            self: (todo): write your description
            id: (str): write your description
            tokens: (int): write your description
            pred: (todo): write your description
        """
        self.id = id
        self.tokens = tokens
        self.pred = pred

    @property
    def words(self):
        """
        Returns a list of words.

        Args:
            self: (todo): write your description
        """
        return [_.text for _ in self.tokens]


def substoken(text, vocab):
    """
    Substoken a subs. substoken.

    Args:
        text: (str): write your description
        vocab: (todo): write your description
    """
    subs = bert_subs(text, vocab)
    return SubsToken(text, subs)


def text_items(texts, vocab):
    """
    Generate : param texts from text.

    Args:
        texts: (str): write your description
        vocab: (todo): write your description
    """
    for id, text in enumerate(texts):
        tokens = [
            substoken(_.text, vocab)
            for _ in tokenize(text)
        ]
        yield BERTInferItem(id, tokens)


def word_items(items, vocab):
    """
    Yields a list.

    Args:
        items: (todo): write your description
        vocab: (str): write your description
    """
    for id, words in enumerate(items):
        tokens = [
            substoken(_, vocab)
            for _ in words
        ]
        yield BERTInferItem(id, tokens)


def segment_items(items, seq_len):
    """
    Yields chunks from seq_len.

    Args:
        items: (todo): write your description
        seq_len: (int): write your description
    """
    for item in items:
        chunks = chop_weighted(
            item.tokens,
            seq_len,
            weight=lambda _: len(_.subs)
        )
        for chunk in chunks:
            yield BERTInferItem(item.id, chunk)


def flatten(seqs):
    """
    Flatten a sequence of sequences.

    Args:
        seqs: (todo): write your description
    """
    return [
        item
        for seq in seqs
        for item in seq
    ]


def join_items(items):
    """
    Join items in items.

    Args:
        items: (todo): write your description
    """
    for id, group in groupby(items, key=lambda _: _.id):
        group = list(group)
        tokens = flatten(_.tokens for _ in group)
        pred = flatten(_.pred for _ in group)
        yield BERTInferItem(id, tokens, pred)


#######
#
#  DECODE
#
#######


class BERTTagDecoder(TagDecoder):
    pass


class BERTSyntaxDecoder(SyntaxDecoder):
    pass


#######
#
#   INFER
#
######


class BERTNERInfer(Infer):
    def process(self, inputs):
        """
        Process the given inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.pad_mask)
            pred = pad_masked(pred, input.word_mask)
            mask = pad_masked(input.word_mask, input.word_mask)
            yield from self.model.head.crf.decode(pred, mask)

    def __call__(self, chunk):
        """
        Yields - based on - batch.

        Args:
            self: (todo): write your description
            chunk: (array): write your description
        """
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


class BERTMorphInfer(Infer):
    def process(self, inputs):
        """
        Process the model.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.pad_mask)
            pred = self.model.head.decode(pred)
            yield from split_masked(pred, input.word_mask)

    def __call__(self, chunk):
        """
        Generator that yields a batch.

        Args:
            self: (todo): write your description
            chunk: (array): write your description
        """
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


class BERTSyntaxInfer(Infer):
    def process(self, inputs):
        """
        Processes a list of the inputs.

        Args:
            self: (todo): write your description
            inputs: (todo): write your description
        """
        for input in inputs:
            input = input.to(self.model.device)
            pred = self.model(input.word_id, input.word_mask, input.pad_mask)
            head_id = self.model.head.decode(pred.head_id)
            rel_id = self.model.rel.decode(pred.rel_id)
            yield from zip(head_id, rel_id)

    def __call__(self, chunk):
        """
        Parameters ---------- chunk : a batch of words.

        Args:
            self: (todo): write your description
            chunk: (array): write your description
        """
        items = list(word_items(chunk, self.encoder.words_vocab))
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
