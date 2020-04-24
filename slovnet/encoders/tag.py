
import torch

from slovnet.record import Record
from slovnet.chop import chop, chop_drop
from slovnet.shape import word_shape
from slovnet.batch import Batch

from .buffer import ShuffleBuffer
from .common import WordShapeInferEncoder


class TagTrainInput(Record):
    __attributes__ = ['word_id', 'shape_id']


class TagTrainEncoder:
    def __init__(self, words_vocab, shapes_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.shapes_vocab = shapes_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def items(self, markups):
        for markup in markups:
            for token in markup.tokens:
                shape = word_shape(token.text)
                word = token.text.lower()
                yield (
                    self.words_vocab.encode(word),
                    self.shapes_vocab.encode(shape),
                    self.tags_vocab.encode(token.tag)
                )

    def batch(self, chunk):
        chunk = torch.tensor(chunk, dtype=torch.long)  # batch x seq x (word, shp, tag)
        word_id, shape_id, tag_id = chunk.unbind(-1)

        input = TagTrainInput(word_id, shape_id)
        return Batch(input, tag_id)

    def __call__(self, markups):
        items = self.items(markups)
        seqs = chop_drop(items, self.seq_len)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


class TagInferEncoder(WordShapeInferEncoder):
    pass
