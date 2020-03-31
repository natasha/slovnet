
from razdel import tokenize as tokenize_

from .record import Record


class Token(Record):
    __attributes__ = ['start', 'stop', 'text']


def tokenize(text):
    for token in tokenize_(text):
        yield Token(
            token.start,
            token.stop,
            token.text
        )


def find_tokens(text, words):
    pass
