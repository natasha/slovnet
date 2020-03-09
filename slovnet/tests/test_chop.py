
from slovnet.chop import (
    chop,
    chop_fill
)


def test_chop():
    guess = chop(range(10), 3)
    etalon = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert etalon == list(guess)


def test_chop_fill():
    guess = chop_fill(range(10), 3)
    etalon = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [7, 8, 9]]
    assert etalon == list(guess)

    guess = chop_fill(range(3), 5)
    etalon = [[0, 1, 2, 0, 1]]
    assert etalon == list(guess)
