
from .record import Record
from .bio import (
    spans_bio,
    bio_spans
)


class Markup(Record):
    pass


class SpanMarkup(Markup):
    __attributes__ = ['text', 'spans']

    def __init__(self, text, spans):
        self.text = text
        self.spans = spans

    def to_bio(self, tokenizer):
        tokens = list(tokenizer(self.text))
        tags = list(spans_bio(tokens, self.spans))
        return BIOMarkup(tokens, tags)


class TagMarkup(Markup):
    __attributes__ = ['tokens', 'tags']

    def __init__(self, tokens, tags):
        self.tokens = tokens
        self.tags = tags

    @property
    def pairs(self):
        return zip(self.tokens, self.tags)

    @classmethod
    def from_pairs(cls, pairs):
        tokens, tags = [], []
        for token, tag in pairs:
            tokens.append(token)
            tags.append(tag)
        return cls(tokens, tags)


def tokens_text(tokens, fill=' '):
    previous = None
    parts = []
    for token in tokens:
        if previous:
            parts.append(fill * (token.start - previous.stop))
        parts.append(token.text)
        previous = token
    return ''.join(parts)


class BIOMarkup(TagMarkup):
    def to_span(self):
        text = tokens_text(self.tokens)
        spans = list(bio_spans(self.tokens, self.tags))
        return SpanMarkup(text, spans)
