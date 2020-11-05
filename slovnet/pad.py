
from torch.nn.utils.rnn import pad_sequence as pad_sequence_


def pad_sequence(seqs, fill=0):
    """
    Pad a sequence with padding padding sequence.

    Args:
        seqs: (list): write your description
        fill: (str): write your description
    """
    return pad_sequence_(
        seqs,
        batch_first=True,
        padding_value=fill
    )
