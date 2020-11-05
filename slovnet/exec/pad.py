
import numpy as np


def pad_sequence(sequences, fill=0):
    """
    Pad a sequence of sequences into a single array.

    Args:
        sequences: (list): write your description
        fill: (str): write your description
    """
    # assert all sequences are 1d
    size = max(_.size for _ in sequences)
    array = np.full((len(sequences), size), fill)
    for index, sequence in enumerate(sequences):
        array[index, :sequence.size] = sequence
    return array
