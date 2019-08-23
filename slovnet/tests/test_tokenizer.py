
import pytest

from slovnet.tokenizer import (
    RU, LATIN, PUNCT, INT,

    Token,
    Tokenizer
)


@pytest.fixture(scope='module')
def tokenizer():
    return Tokenizer()


TESTS = [
    [
        'Москва-3',
        [
            Token(text='Москва', start=0, stop=6, type=RU),
            Token(text='-', start=6, stop=7, type=PUNCT),
            Token(text='3', start=7, stop=8, type=INT)
        ]
    ],
    [
        'New-York',
        [
            Token(text='New-York', start=0, stop=8, type=LATIN)
        ]
    ],
    [
        'сине-бело-голубой',
        [
            Token(text='сине-бело-голубой', start=0, stop=17, type=RU)
        ]
    ],
    [
        'Yahoo!',
        [
            Token(text='Yahoo', start=0, stop=5, type=LATIN),
            Token(text='!', start=5, stop=6, type=PUNCT)
        ]
    ],
]


@pytest.mark.parametrize('test', TESTS)
def test_tokenizer(tokenizer, test):
    text, etalon = test
    guess = tokenizer(text)
    assert etalon == list(guess)
