
from slovnet.chop import (
    chop,
    chop_equal
)


def test_chop():
    guess = chop(range(10), 3)
    etalon = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert etalon == list(guess)


def test_chop_equal():
    guess = chop_equal(range(10), 3)
    etalon = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [7, 8, 9]]
    assert etalon == list(guess)

    guess = chop_equal(range(3), 5)
    etalon = [[0, 1, 2, 0, 1]]
    assert etalon == list(guess)
