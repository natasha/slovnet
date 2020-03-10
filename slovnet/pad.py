
from torch.nn.utils.rnn import pad_sequence as pad_sequence_


def pad_sequence(seqs, fill=0):
    return pad_sequence_(
        seqs,
        batch_first=True,
        padding_value=fill
    )
