
from .record import Record
from .bio import spans_bio
from .sent import (
    sentenize,
    sent_spans
)


class Markup(Record):
    pass


class SpanMarkup(Markup):
    __attributes__ = ['text', 'spans']

    def __init__(self, text, spans):
        self.text = text
        self.spans = spans

    @property
    def sents(self):
        for sent in sentenize(self.text):
            spans = sent_spans(sent, self.spans)
            yield SpanMarkup(
                sent.text,
                list(spans)
            )

    def to_tag(self, tokenizer):
        tokens = list(tokenizer(self.text))
        tags = list(spans_bio(tokens, self.spans))
        return TagMarkup(tokens, tags)


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
        tokens = []
        tags = []
        for token, tag in pairs:
            tokens.append(token)
            tags.append(tag)
        return cls(tokens, tags)
