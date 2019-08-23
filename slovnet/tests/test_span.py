
from slovnet.span import (
    Span,
    offset_spans,
    envelop_spans,
    select_type_spans
)


def test_offset_spans():
    spans = [Span(1, 2), Span(3, 4)]
    guess = offset_spans(spans, -1)
    etalon = [Span(0, 1), Span(2, 3)]
    assert etalon == list(guess)


def test_envelop_spans():
    spans = [Span(1, 2), Span(3, 4), Span(4, 5)]
    guess = envelop_spans(Span(3, 5), spans)
    etalon = [Span(3, 4), Span(4, 5)]
    assert etalon == list(guess)


def test_select_type():
    spans = [Span(1, 2, 'A'), Span(3, 4, 'B')]
    guess = select_type_spans(spans, ['B'])
    etalon = [Span(3, 4, 'B')]
    assert etalon == list(guess)
