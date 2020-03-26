
import re

import torch

from slovnet.record import Record
from slovnet.pad import pad_sequence
from slovnet.chop import chop_drop
from slovnet.batch import Batch
from slovnet.mask import Masked, pad_masked

from .buffer import ShuffleBuffer, SizeBuffer


def wordpiece(text, vocab, prefix='##'):
    start = 0
    stop = size = len(text)
    parts = []
    while start < size:
        part = text[start:stop]
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


##########
#
#   MLM
#
########


def mlm_split(text):
    # diff with bert tokenizer 28 / 10000 ~0.3%
    # школа №3 -> школа, №3
    # @diet_prada -> @, diet, _, prada
    return re.findall(r'\w+|[^\w\s]', text)


def mlm_ids(texts, vocab):
    for text in texts:
        chunks = mlm_split(text)
        for chunk in chunks:
            parts = wordpiece(chunk, vocab)
            if not parts:
                yield vocab.unk_id
            else:
                for part in parts:
                    yield vocab.encode(part)


def mlm_seqs(ids, vocab, size):
    for chunk in chop_drop(ids, size - 2):
        yield [vocab.cls_id] + chunk + [vocab.sep_id]


def mlm_mask(input, vocab, prob=0.15):
    prob = torch.full(input.shape, prob)

    spec = (input == vocab.cls_id) | (input == vocab.sep_id)
    prob.masked_fill_(spec, 0)  # do not mask cls, sep

    return torch.bernoulli(prob).bool()


class BERTMLMEncoder:
    def __init__(self, vocab,
                 seq_len=512, batch_size=8, shuffle_size=1,
                 mask_prob=0.15, ignore_id=-100):
        self.vocab = vocab
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

        self.mask_prob = mask_prob
        self.ignore_id = ignore_id

    def __call__(self, texts):
        ids = mlm_ids(texts, self.vocab)
        seqs = mlm_seqs(ids, self.vocab, self.seq_len)
        seqs = self.shuffle(seqs)
        inputs = chop_drop(seqs, self.batch_size)

        for input in inputs:
            input = torch.tensor(input).long()
            target = input.clone()

            mask = mlm_mask(input, self.vocab, self.mask_prob)
            input[mask] = self.vocab.mask_id
            target[~mask] = self.ignore_id

            yield Batch(input, target)


#########
#
#   NER
#
######


def ner_items(markups, words_vocab, tags_vocab):
    for markup in markups:
        for token in markup.tokens:
            parts = wordpiece(token.text, words_vocab)
            tag_id = tags_vocab.encode(token.tag)
            if not parts:
                yield (words_vocab.unk_id, tag_id, True)
            else:
                for index, part in enumerate(parts):
                    yield (
                        words_vocab.encode(part),
                        tag_id,
                        index == 0  # use first subtoken
                    )


def ner_seqs(items, words_vocab, tags_vocab, size):
    cls = (words_vocab.cls_id, tags_vocab.pad_id, False)
    sep = (words_vocab.sep_id, tags_vocab.pad_id, False)
    for chunk in chop_drop(items, size - 2):
        yield [cls] + chunk + [sep]


class BERTNEREncoder:
    def __init__(self, words_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def __call__(self, markups):
        items = ner_items(markups, self.words_vocab, self.tags_vocab)
        seqs = ner_seqs(items, self.words_vocab, self.tags_vocab, self.seq_len)
        seqs = self.shuffle(seqs)
        chunks = chop_drop(seqs, self.batch_size)

        for chunk in chunks:
            chunk = torch.tensor(chunk)  # batch x seq x (word, mask, tag)
            word_id, tag_id, mask = chunk.unbind(-1)
            word_id, tag_id, mask = word_id.long(), tag_id.long(), mask.bool()

            input = Masked(word_id, mask)
            target = Masked(
                pad_masked(tag_id, input.mask),
                pad_masked(input.mask, input.mask)
            )

            yield Batch(input, target)


###########
#
#   MORPH
#
#######


def morph_items(markups, words_vocab, tags_vocab):
    for markup in markups:
        for token in markup.tokens:
            parts = wordpiece(token.text, words_vocab)
            tag_id = tags_vocab.encode(token.tag)
            if not parts:
                yield (words_vocab.unk_id, tag_id, True)
            else:
                for index, part in enumerate(parts):
                    yield (
                        words_vocab.encode(part),
                        tag_id,
                        index == 0
                    )


def morph_seqs(items, words_vocab, tags_vocab, size):
    cls = (words_vocab.cls_id, tags_vocab.pad_id, False)
    sep = (words_vocab.sep_id, tags_vocab.pad_id, False)
    for chunk in chop_drop(items, size - 2):
        yield [cls] + chunk + [sep]


class BERTMorphEncoder:
    def __init__(self, words_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def __call__(self, markups):
        items = morph_items(markups, self.words_vocab, self.tags_vocab)
        seqs = morph_seqs(items, self.words_vocab, self.tags_vocab, self.seq_len)
        seqs = self.shuffle(seqs)
        chunks = chop_drop(seqs, self.batch_size)

        for chunk in chunks:
            chunk = torch.tensor(chunk)  # batch x seq x (word, mask, tag)
            word_id, tag_id, mask = chunk.unbind(-1)
            word_id, tag_id, mask = word_id.long(), tag_id.long(), mask.bool()

            input = Masked(word_id, mask)
            target = Masked(
                pad_masked(tag_id, input.mask),
                pad_masked(input.mask, input.mask)
            )

            yield Batch(input, target)


########
#
#   SYNTAX
#
####


class SyntaxItem(Record):
    __attributes__ = ['size', 'word_ids', 'mask', 'head_ids', 'rel_ids']

    def __init__(self, size, word_ids, mask, head_ids, rel_ids):
        self.size = size
        self.word_ids = word_ids
        self.mask = mask
        self.head_ids = head_ids
        self.rel_ids = rel_ids


def syntax_item(markup, words_vocab, rels_vocab):
    word_ids, mask, head_ids, rel_ids = [], [], [], []
    for token in markup.tokens:
        parts = wordpiece(token.text, words_vocab)
        if not parts:
            word_ids.append(words_vocab.unk_id)
            mask.append(True)
        else:
            for index, part in enumerate(parts):
                id = words_vocab.encode(part)
                word_ids.append(id)
                mask.append(index == 0)

        head_ids.append(token.head_id)

        id = rels_vocab.encode(token.rel)
        rel_ids.append(id)

    size = len(markup.tokens)
    word_ids = [words_vocab.cls_id] + word_ids + [words_vocab.sep_id]
    mask = [False] + mask + [False]
    return SyntaxItem(size, word_ids, mask, head_ids, rel_ids)


def syntax_items(markups, words_vocab, rels_vocab):
    for markup in markups:
        yield syntax_item(markup, words_vocab, rels_vocab)


def syntax_chop(items, max_seq_len, max_items):
    buffer = []
    accum = 0
    for item in items:
        size = len(item.word_ids)

        if size > max_seq_len:  # 0.02% sents longer then 128
            continue

        buffer.append(item)
        accum += size

        if accum >= max_items:
            yield buffer
            buffer = []
            accum = 0

    if buffer:
        yield buffer


class SyntaxInput(Record):
    __attributes__ = ['word_id', 'word_mask', 'pad_mask']

    def __init__(self, word_id, word_mask, pad_mask):
        self.word_id = word_id
        self.word_mask = word_mask
        self.pad_mask = pad_mask

    def to(self, device):
        return SyntaxInput(
            self.word_id.to(device),
            self.word_mask.to(device),
            self.pad_mask.to(device)
        )


class SyntaxTarget(Record):
    __attributes__ = ['head_id', 'rel_id', 'mask']

    def __init__(self, head_id, rel_id, mask):
        self.head_id = head_id
        self.rel_id = rel_id
        self.mask = mask

    def to(self, device):
        return SyntaxTarget(
            self.head_id.to(device),
            self.rel_id.to(device),
            self.mask.to(device)
        )


def syntax_batch(items, words_vocab, rels_vocab):
    word_id, mask, head_id, rel_id = [], [], [], []
    for item in items:
        word_id.append(torch.tensor(item.word_ids, dtype=torch.long))
        mask.append(torch.tensor(item.mask, dtype=torch.bool))
        head_id.append(torch.tensor(item.head_ids, dtype=torch.long))
        rel_id.append(torch.tensor(item.rel_ids, dtype=torch.long))

    word_id = pad_sequence(word_id, fill=words_vocab.pad_id)
    word_mask = pad_sequence(mask, fill=False)
    pad_mask = word_id == words_vocab.pad_id
    input = SyntaxInput(word_id, word_mask, pad_mask)

    head_id = pad_sequence(head_id)
    rel_id = pad_sequence(rel_id, fill=rels_vocab.pad_id)
    mask = rel_id == rels_vocab.pad_id
    target = SyntaxTarget(head_id, rel_id, mask)

    return Batch(input, target)


class BERTSyntaxEncoder:
    def __init__(self, words_vocab, rels_vocab,
                 seq_len=512, batch_size=8,
                 shuffle_size=1, size_size=1):
        self.words_vocab = words_vocab
        self.rels_vocab = rels_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)
        self.size = SizeBuffer(size_size)

    def __call__(self, markups):
        items = syntax_items(markups, self.words_vocab, self.rels_vocab)
        items = self.shuffle(items)
        chunks = self.size(items)

        max_items = self.seq_len * self.batch_size
        for chunk in chunks:
            for chunk in syntax_chop(chunk, self.seq_len, max_items):
                yield syntax_batch(chunk, self.words_vocab, self.rels_vocab)
