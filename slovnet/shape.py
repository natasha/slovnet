
from .tokenizer import (
    RU, LATIN,
    INT, OTHER,
    PUNCT, PUNCTS
)


X = 'X'
x = 'x'
XX = 'XX'
xx = 'xx'
Xx = 'Xx'
Xx_Xx = 'Xx-Xx'
UNK = '<unk>'


def is_title(word):
    return len(word) > 1 and word[0].isupper() and word[1:].islower()


def is_dash_title(word):
    if '-' in word:
        left, right = word.split('-', 1)
        return is_title(left) and is_title(right)


def get_word_shape(word):
    if len(word) == 1:
        if word.isupper():
            return X
        else:
            return x
    else:
        if word.isupper():
            return XX
        elif word.islower():
            return xx
        elif is_title(word):
            return Xx
        elif is_dash_title(word):
            return Xx_Xx
        else:
            return UNK


def format_shape(type, value):
    return '%s_%s' % (type, value)


def get_shape(token):
    text = token.text
    type = token.type
    if type in (RU, LATIN):
        return format_shape(type, get_word_shape(text))
    elif type == PUNCT:
        return format_shape(PUNCT, text)
    elif type in (INT, OTHER):
        return type


WORD_SHAPES = [X, x, XX, xx, Xx, Xx_Xx, UNK]
SHAPES = (
    [format_shape(RU, _) for _ in WORD_SHAPES]
    + [format_shape(LATIN, _) for _ in WORD_SHAPES]
    + [format_shape(PUNCT, _) for _ in PUNCTS]
    + [INT, OTHER]
)
