
import pytest

from slovnet.shape import (
    X, x, xx, XX, Xx, Xx_Xx, UNK,
    RU, LATIN, INT, PUNCT,

    get_shape,
    format_shape as s
)
from slovnet.tokenizer import Tokenizer


@pytest.fixture(scope='module')
def tokenizer():
    return Tokenizer()


TESTS = [
    [
        'В',
        [s(RU, X)],
    ],
    [
        'ИЛ-2',
        [s(RU, XX), s(PUNCT, '-'), INT],
    ],
    [
        '105г.',
        [INT, s(RU, x), s(PUNCT, '.')]
    ],
    [
        'Pal-Yz',
        [s(LATIN, Xx_Xx)]
    ],
    [
        'и Я-ДаА',
        [s(RU, x), s(RU, UNK)]
    ],
    [
        'Прибыл на I@',
        [s(RU, Xx), s(RU, xx), s(LATIN, X), s(PUNCT, '@')]
    ],
]


@pytest.mark.parametrize('test', TESTS)
def test_shape(tokenizer, test):
    text, etalon = test
    tokens = tokenizer(text)
    guess = [get_shape(_) for _ in tokens]
    assert guess == etalon
