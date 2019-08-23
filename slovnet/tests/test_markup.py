
import re

from slovnet.bio import B, I, O
from slovnet.span import Span
from slovnet.tokenizer import Tokenizer
from slovnet.markup import (
    SpanMarkup,
    TagMarkup
)


def find_spans(text, pattern):
    matches = re.finditer(pattern, text)
    for match in matches:
        start = match.start()
        stop = match.end()
        yield Span(start, stop)


def test_sents():
    text = 'И. А. Ильяхов. Путь через Пирении.'
    spans = find_spans(text, r'И\. А\. Ильяхов|Пирении')
    markup = SpanMarkup(text, list(spans))
    guess = list(markup.sents)
    etalon = [
        SpanMarkup(text='И. А. Ильяхов.', spans=[Span(start=0, stop=13)]),
        SpanMarkup(text='Путь через Пирении.', spans=[Span(start=11, stop=18)])
    ]
    assert guess == etalon


def test_tag():
    text = '1 мая в Спб...'
    spans = find_spans(text, r'1 мая|Спб')
    markup = SpanMarkup(text, list(spans))

    tokenizer = Tokenizer()
    tokens = list(tokenizer(text))
    tags = [B, I, O, B, O, O, O]
    guess = markup.to_tag(tokenizer)
    etalon = TagMarkup(tokens, tags)
    assert guess == etalon
    assert guess == TagMarkup.from_pairs(guess.pairs)
