
import torch

from slovnet.chop import chop_drop
from slovnet.batch import Batch

from .shuffle import ShuffleBuffer


def wordpiece(token, vocab, prefix='##'):
    start = 0
    stop = size = len(token)
    parts = []
    while start < size:
        part = token[start:stop]
        if start > 0:
            part = prefix + part
        if part in vocab.item_ids:
            parts.append(part)
            start = stop
            stop = size
        else:
            stop -= 1
            if stop < start:
                return
    return parts


def texts_ids(texts, vocab):
    for text in texts:
        tokens = re_tokenize(text)
        for token in tokens:
            parts = wordpiece(token, vocab)
            if not parts:
                yield vocab.unk_id
            else:
                for part in parts:
                    yield vocab.encode(part)


def ids_seqs(ids, vocab, size):
    for chunk in chop_drop(ids, size - 2):
        yield [vocab.cls_id] + chunk + [vocab.sep_id]


def mlm_mask(input, vocab, prob=0.15):
    prob = torch.full(input.shape, prob)

    spec = (input == vocab.cls_id) | (input == vocab.sep_id)
    prob.masked_fill_(spec, 0)  # do not mask cls, sep

    return torch.bernoulli(prob).bool()


class BERTMLMEncoder:
    def __init__(self, vocab,
                 seq_size=512, batch_size=8, shuffle_size=1,
                 mask_prob=0.15, ignore_id=-100):
        self.vocab = vocab
        self.seq_size = seq_size
        self.batch_size = batch_size
        
        self.shuffle = ShuffleBuffer(shuffle_size)

        self.mask_prob = mask_prob
        self.ignore_id = ignore_id

    def map(self, texts):
        vocab, seq_size, batch_size = self.vocab, self.seq_size, self.batch_size

        ids = texts_ids(texts, vocab)
        seqs = ids_seqs(ids, vocab, seq_size)
        seqs = self.shuffle.map(seqs)
        inputs = chop_drop(seqs, batch_size)

        for input in inputs:
            input = torch.LongTensor(input)
            target = input.clone()

            mask = mlm_mask(input, vocab, self.mask_prob)
            input[mask] = vocab.mask_id
            target[~mask] = self.ignore_id

            yield Batch(input, target)
