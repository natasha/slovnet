
import numpy as np


def pad_sequence(sequences, fill=0):
    # assert all sequences are 1d
    size = max(_.size for _ in sequences)
    array = np.full((len(sequences), size), fill)
    for index, sequence in enumerate(sequences):
        array[index, :sequence.size] = sequence
    return array
