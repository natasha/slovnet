
from ipymarkup import show_span_box_markup as show_span_markup  # noqa

from .record import Record
from .bio import (
    spans_bio,
    bio_spans
)
from .token import find_tokens
from .sent import sentenize
from .span import (
    Span,
    envelop_spans,
    offset_spans
)


########
#
#   SPAN
#
#######


def sent_spans(sent, spans):
    spans = envelop_spans(sent, spans)
    return offset_spans(spans, -sent.start)


class SpanMarkup(Record):
    __attributes__ = ['text', 'spans']
    __annotations__ = {
        'spans': [Span]
    }

    def __init__(self, text, spans):
        self.text = text
        self.spans = spans

    @property
    def sents(self):
        for sent in sentenize(self.text):
            yield SpanMarkup(
                sent.text,
                list(sent_spans(sent, self.spans))
            )

    def to_bio(self, tokens):
        tags = spans_bio(tokens, self.spans)
        words = [_.text for _ in tokens]
        return BIOMarkup.from_pairs(zip(words, tags))


########
#
#   TAG
#
######


class TagToken(Record):
    __attributes__ = ['text', 'tag']

    def __init__(self, text, tag):
        self.text = text
        self.tag = tag


class TagMarkup(Record):
    __attributes__ = ['tokens']
    __annotations__ = {
        'tokens': [TagToken]
    }

    def __init__(self, tokens):
        self.tokens = tokens

    @property
    def words(self):
        return [_.text for _ in self.tokens]

    @property
    def tags(self):
        return [_.tag for _ in self.tokens]

    @classmethod
    def from_pairs(cls, pairs):
        return cls([
            TagToken(word, tag)
            for word, tag in pairs
        ])


class BIOMarkup(TagMarkup):
    def to_span(self, text):
        tokens = find_tokens(text, self.words)
        spans = list(bio_spans(tokens, self.tags))
        return SpanMarkup(text, spans)


#######
#
#   SYNTAX
#
#######


class SyntaxToken(Record):
    __attributes__ = ['word', 'head_id', 'rel']

    def __init__(self, word, head_id, rel):
        self.word = word
        self.head_id = head_id
        self.rel = rel


class SyntaxMarkup(Record):
    __attributes__ = ['tokens']
    __annotations__ = {
        'tokens': [SyntaxToken]
    }

    def __init__(self, tokens):
        self.tokens = tokens
