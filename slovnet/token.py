
from razdel import tokenize as tokenize_

from .record import Record


class Token(Record):
    __attributes__ = ['start', 'stop', 'text']


def tokenize(text):
    """
    Tokenize a generator.

    Args:
        text: (str): write your description
    """
    for token in tokenize_(text):
        yield Token(
            token.start,
            token.stop,
            token.text
        )


def find_tokens(text, chunks):
    """
    Yields chunks of text string.

    Args:
        text: (str): write your description
        chunks: (int): write your description
    """
    offset = 0
    for chunk in chunks:
        start = text.find(chunk, offset)
        stop = start + len(chunk)
        yield Token(start, stop, chunk)
        offset = stop
