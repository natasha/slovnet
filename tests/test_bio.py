
import pytest

from slovnet.token import tokenize
from slovnet.const import (
    B, I, O,
    PER, LOC
)
from slovnet.bio import (
    format_bio,

    io_spans,
    spans_io,

    bio_spans,
    spans_bio,

    bio_io,
    select_type_tags
)


T1, T2, T3, T4, T5 = tokenize('1 2 3 4 5')

B_PER = format_bio(B, PER)
I_PER = format_bio(I, PER)
B_LOC = format_bio(B, LOC)
I_LOC = format_bio(I, LOC)


TESTS = [
    [
        [T1, T2, T3],
        [O, O, O],
    ],
    [
        [],
        [],
    ]
]

IO_TESTS = [
    [
        [T1, T2, T3],
        [I_PER, O, O]
    ],
    [
        [T1, T2, T3],
        [I_PER, I_PER, O]
    ],
    [
        [T1, T2, T3],
        [I_PER, I_LOC, O]
    ],
    [
        [T1, T2],
        [I_PER, I_PER]
    ],
]

BIO_TESTS = [
    [
        [T1, T2, T3],
        [B_PER, O, O],
    ],
    [
        [T1, T2, T3],
        [B_PER, I_PER, O],
    ],
    [
        [T1, T2],
        [B_PER, I_PER],
    ],
    [
        [T1, T2, T3],
        [B_PER, B_LOC, O],
    ],
    [
        [T1, T2, T3],
        [B_PER, B_PER, O],
    ],
]

CONVERT_TESTS = [
    [
        [B_PER, I_PER],
        [I]
    ]
]


@pytest.mark.parametrize('test', TESTS + IO_TESTS)
def test_io(test):
    tokens, tags = test
    spans = io_spans(tokens, tags)
    guess = spans_io(tokens, spans)
    assert tags == list(guess)


@pytest.mark.parametrize('test', TESTS + BIO_TESTS)
def test_bio(test):
    tokens, tags = test
    spans = bio_spans(tokens, tags)
    guess = spans_bio(tokens, spans)
    assert tags == list(guess)


def test_convert():
    guess = bio_io([B_PER, I_PER, I_LOC])
    etalon = [I_PER, I_PER, I_LOC]
    assert etalon == list(guess)


def test_select():
    guess = select_type_tags([B_PER, I_LOC], PER)
    etalon = [B_PER, O]
    assert etalon == list(guess)
