
import re

RU = 'RU'
EN = 'EN'
NUM = 'NUM'
PUNCT = 'PUNCT'
OTHER = 'OTHER'

PUNCTS = (
    '!#$%&()[]\\/*+,.:;<=>?@^_{|}~'  # string.punctuation
    '-‐−‒⁃–—―'  # https://habr.com/ru/post/20588/
    '`"\'«»„“ʼʻ”'
    '№…'
)
TYPE = re.compile(
    r'''
    (?P<RU>[а-яё]+)
    |(?P<EN>[a-z]+)
    |(?P<NUM>[+-]?\d+)
    |(?P<PUNCT>[%s]+)
    ''' % re.escape(PUNCTS),
    re.X | re.IGNORECASE
)

X = 'X'
x = 'x'
XX = 'XX'
xx = 'xx'
Xx = 'Xx'
Xx_Xx = 'Xx-Xx'


def is_title(word):
    """
    Returns true if word is a title.

    Args:
        word: (str): write your description
    """
    return len(word) > 1 and word[0].isupper() and word[1:].islower()


def is_dash_title(word):
    """
    Returns true if the given word is a valid title.

    Args:
        word: (str): write your description
    """
    if '-' in word:
        left, right = word.split('-', 1)
        return is_title(left) and is_title(right)


def word_outline(word):
    """
    Return the outline of a word.

    Args:
        word: (str): write your description
    """
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
            return OTHER


def word_type(word):
    """
    Determine type of a word.

    Args:
        word: (str): write your description
    """
    # СИЗО-6 -> RU
    # 2011-2020 -> NUM
    match = TYPE.match(word)
    if match:
        return match.lastgroup
    return OTHER


def format_shape(type, value):
    """
    Format a shape for a type.

    Args:
        type: (str): write your description
        value: (todo): write your description
    """
    return '%s_%s' % (type, value)


def word_shape(word):
    """
    Returns a shape string.

    Args:
        word: (str): write your description
    """
    type = word_type(word)
    if type in (RU, EN):
        return format_shape(type, word_outline(word))
    elif type == PUNCT:
        if len(word) > 1 or word not in PUNCTS:
            # ..., ?!, ****
            word = OTHER
        return format_shape(PUNCT, word)
    elif type in (NUM, OTHER):
        return type


OUTLINES = [X, x, XX, xx, Xx, Xx_Xx, OTHER]
SHAPES = (
    [format_shape(RU, _) for _ in OUTLINES]
    + [format_shape(EN, _) for _ in OUTLINES]
    + [format_shape(PUNCT, _) for _ in PUNCTS]
    + [format_shape(PUNCT, OTHER), NUM, OTHER]
)
