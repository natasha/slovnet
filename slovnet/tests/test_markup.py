
import re

from slovnet.bio import B, I, O
from slovnet.span import Span
from slovnet.tokenizer import Tokenizer
from slovnet.markup import (
    SpanMarkup,
    BIOMarkup
)


def find_spans(text, pattern):
    matches = re.finditer(pattern, text)
    for match in matches:
        start = match.start()
        stop = match.end()
        yield Span(start, stop)


def test_bio():
    text = '1 мая в Спб...'
    spans = find_spans(text, r'1 мая|Спб')
    markup = SpanMarkup(text, list(spans))

    tokenizer = Tokenizer()
    tokens = list(tokenizer(text))
    tags = [B, I, O, B, O, O, O]
    guess = markup.to_bio(tokenizer)
    etalon = BIOMarkup(tokens, tags)
    assert guess == etalon
    assert guess == BIOMarkup.from_pairs(guess.pairs)
