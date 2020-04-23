
import torch

from slovnet.record import Record
from slovnet.pad import pad_sequence
from slovnet.chop import chop, chop_drop
from slovnet.batch import Batch
from slovnet.mask import Masked, pad_masked
from slovnet.bert import bert_subs

from .buffer import ShuffleBuffer, SortBuffer


##########
#
#   MLM
#
########


class BERTMLMTrainEncoder:
    def __init__(self, vocab,
                 seq_len=512, batch_size=8, shuffle_size=1,
                 mask_prob=0.15):
        self.vocab = vocab
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

        self.mask_prob = mask_prob

    def items(self, texts):
        for text in texts:
            subs = bert_subs(text, self.vocab)
            for sub in subs:
                yield self.vocab.encode(sub)

    def seqs(self, items):
        for chunk in chop_drop(items, self.seq_len - 2):
            yield [self.vocab.cls_id] + chunk + [self.vocab.sep_id]

    def mask(self, input):
        prob = torch.full(input.shape, self.mask_prob)

        spec = (input == self.vocab.cls_id) | (input == self.vocab.sep_id)
        prob.masked_fill_(spec, 0)  # do not mask cls, sep

        return torch.bernoulli(prob).bool()

    def batch(self, chunk):
        input = torch.tensor(chunk).long()
        target = input.clone()

        mask = self.mask(input)
        input[mask] = self.vocab.mask_id

        return Batch(input, Masked(target, mask))

    def __call__(self, texts):
        items = self.items(texts)
        seqs = self.seqs(items)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


#########
#
#   NER
#
######


class BERTNERTrainEncoder:
    def __init__(self, words_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def items(self, markups):
        for markup in markups:
            for token in markup.tokens:
                subs = bert_subs(token.text, self.words_vocab)
                tag_id = self.tags_vocab.encode(token.tag)
                for index, sub in enumerate(subs):
                    yield (
                        self.words_vocab.encode(sub),
                        tag_id,
                        index == 0  # use first subtoken
                    )

    def seqs(self, items):
        cls = (self.words_vocab.cls_id, self.tags_vocab.pad_id, False)
        sep = (self.words_vocab.sep_id, self.tags_vocab.pad_id, False)
        for chunk in chop_drop(items, self.seq_len - 2):
            yield [cls] + chunk + [sep]

    def batch(self, chunk):
        chunk = torch.tensor(chunk)  # batch x seq x (word, mask, tag)
        word_id, tag_id, mask = chunk.unbind(-1)
        word_id, tag_id, mask = word_id.long(), tag_id.long(), mask.bool()

        input = Masked(word_id, mask)
        target = Masked(
            pad_masked(tag_id, input.mask),
            pad_masked(input.mask, input.mask)
        )

        return Batch(input, target)

    def __call__(self, markups):
        items = self.items(markups)
        seqs = self.seqs(items)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


###########
#
#   MORPH
#
#######


class BERTMorphTrainEncoder:
    def __init__(self, words_vocab, tags_vocab,
                 seq_len=512, batch_size=8, shuffle_size=1):
        self.words_vocab = words_vocab
        self.tags_vocab = tags_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.shuffle = ShuffleBuffer(shuffle_size)

    def items(self, markups):
        for markup in markups:
            for token in markup.tokens:
                subs = bert_subs(token.text, self.words_vocab)
                tag_id = self.tags_vocab.encode(token.tag)
                for index, sub in enumerate(subs):
                    yield (
                        self.words_vocab.encode(sub),
                        tag_id,
                        index == 0
                    )

    def seqs(self, items):
        cls = (self.words_vocab.cls_id, self.tags_vocab.pad_id, False)
        sep = (self.words_vocab.sep_id, self.tags_vocab.pad_id, False)
        for chunk in chop_drop(items, self.seq_len - 2):
            yield [cls] + chunk + [sep]

    def batch(self, chunk):
        chunk = torch.tensor(chunk)  # batch x seq x (word, mask, tag)
        word_id, tag_id, mask = chunk.unbind(-1)
        word_id, tag_id, mask = word_id.long(), tag_id.long(), mask.bool()

        input = Masked(word_id, mask)
        target = Masked(
            pad_masked(tag_id, input.mask),
            pad_masked(input.mask, input.mask)
        )

        return Batch(input, target)

    def __call__(self, markups):
        items = self.items(markups)
        seqs = self.seqs(items)
        seqs = self.shuffle(seqs)
        chunks = chop(seqs, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


########
#
#   SYNTAX
#
####


ROOT_ID = '0'


class BERTSyntaxTrainItem(Record):
    __attributes__ = ['word_ids', 'mask', 'head_ids', 'rel_ids']

    def __len__(self):
        return len(self.rel_ids)


class BERTSyntaxInput(Record):
    __attributes__ = ['word_id', 'word_mask', 'pad_mask']


class BERTSyntaxTarget(Record):
    __attributes__ = ['head_id', 'rel_id', 'mask']


class BERTSyntaxTrainEncoder:
    def __init__(self, words_vocab, rels_vocab,
                 seq_len=512, batch_size=8,
                 sort_size=1):
        self.words_vocab = words_vocab
        self.rels_vocab = rels_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

        self.sort = SortBuffer(sort_size, key=lambda _: len(_.tokens))

    def item(self, markup):
        word_ids, mask, head_ids, rel_ids = [], [], [], []
        ids = {ROOT_ID: 0}

        for index, token in enumerate(markup.tokens, 1):
            ids[token.id] = index
            head_ids.append(token.head_id)

            rel_id = self.rels_vocab.encode(token.rel)
            rel_ids.append(rel_id)

            subs = bert_subs(token.text, self.words_vocab)
            for index, sub in enumerate(subs):
                word_id = self.words_vocab.encode(sub)
                word_ids.append(word_id)
                mask.append(index == 0)

        word_ids = [self.words_vocab.cls_id] + word_ids + [self.words_vocab.sep_id]
        mask = [False] + mask + [False]
        head_ids = [ids[_] for _ in head_ids]
        return BERTSyntaxTrainItem(word_ids, mask, head_ids, rel_ids)

    def batch(self, items):
        word_id, mask, head_id, rel_id = [], [], [], []
        for item in items:
            word_id.append(torch.tensor(item.word_ids, dtype=torch.long))
            mask.append(torch.tensor(item.mask, dtype=torch.bool))
            head_id.append(torch.tensor(item.head_ids, dtype=torch.long))
            rel_id.append(torch.tensor(item.rel_ids, dtype=torch.long))

        word_id = pad_sequence(word_id, fill=self.words_vocab.pad_id)
        word_mask = pad_sequence(mask, fill=False)
        pad_mask = word_id == self.words_vocab.pad_id
        input = BERTSyntaxInput(word_id, word_mask, pad_mask)

        head_id = pad_sequence(head_id)
        rel_id = pad_sequence(rel_id, fill=self.rels_vocab.pad_id)
        mask = rel_id != self.rels_vocab.pad_id
        target = BERTSyntaxTarget(head_id, rel_id, mask)

        return Batch(input, target)

    def __call__(self, markups):
        markups = self.sort(markups)
        items = (self.item(_) for _ in markups)
        # 0.02% sents longer then 128, just drop them
        items = (_ for _ in items if len(_.word_ids) <= self.seq_len)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.batch(chunk)


#########
#
#   INFER
#
########


class BERTInferInput(Record):
    __attributes__ = ['word_id', 'word_mask', 'pad_mask']


class BERTInferEncoder:
    def __init__(self, words_vocab,
                 seq_len=128, batch_size=8):
        self.words_vocab = words_vocab

        self.seq_len = seq_len
        self.batch_size = batch_size

    def item(self, item):
        word_ids, mask = [], []
        for token in item.tokens:
            for index, sub in enumerate(token.subs):
                word_id = self.words_vocab.encode(sub)
                word_ids.append(word_id)
                mask.append(index == 0)

        size = self.seq_len - 2
        return (
            [self.words_vocab.cls_id] + word_ids[:size] + [self.words_vocab.sep_id],
            [False] + mask[:size] + [False]
        )

    def input(self, items):
        word_id, word_mask = [], []
        for word_ids, mask in items:
            word_id.append(torch.tensor(word_ids, dtype=torch.long))
            word_mask.append(torch.tensor(mask, dtype=torch.bool))
        word_id = pad_sequence(word_id, self.words_vocab.pad_id)
        word_mask = pad_sequence(word_mask, False)
        pad_mask = word_id == self.words_vocab.pad_id
        return BERTInferInput(word_id, word_mask, pad_mask)

    def __call__(self, items):
        items = (self.item(_) for _ in items)
        chunks = chop(items, self.batch_size)
        for chunk in chunks:
            yield self.input(chunk)
