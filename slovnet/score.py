
from .record import Record
from .mask import mask_like
from .bio import (
    I,
    bio_io,
    parse_bio,
    select_type_tags
)


class Acc(Record):
    __attributes__ = ['correct', 'total']

    def __init__(self, correct=0, total=0):
        """
        Initialize the object.

        Args:
            self: (todo): write your description
            correct: (todo): write your description
            total: (int): write your description
        """
        self.correct = correct
        self.total = total

    def add(self, other):
        """
        Add another : class : class : class to this : class :.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        self.correct += other.correct
        self.total += other.total

    @property
    def value(self):
        """
        Return the total number of values

        Args:
            self: (todo): write your description
        """
        if not self.total:
            return 0
        return self.correct / self.total

    def reset(self):
        """
        Reset the progress bar.

        Args:
            self: (todo): write your description
        """
        self.correct = 0
        self.total = 0


class Mean(Record):
    __attributes__ = ['accum', 'count']

    def __init__(self, accum=0, count=0):
        """
        Initialize the next instance.

        Args:
            self: (todo): write your description
            accum: (todo): write your description
            count: (int): write your description
        """
        self.accum = accum
        self.count = count

    def add(self, value):
        """
        Add * value to the list.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self.accum += value
        self.count += 1

    @property
    def value(self):
        """
        Return the current value of the result.

        Args:
            self: (todo): write your description
        """
        if not self.count:
            return 0
        return self.accum / self.count

    def reset(self):
        """
        Reset the state.

        Args:
            self: (todo): write your description
        """
        self.accum = 0
        self.count = 0


class F1(Record):
    __attributes__ = ['prec', 'recall']

    def __init__(self, prec=None, recall=None):
        """
        Initialize the record.

        Args:
            self: (todo): write your description
            prec: (float): write your description
            recall: (str): write your description
        """
        if not prec:
            prec = Acc()
        self.prec = prec
        if not recall:
            recall = Acc()
        self.recall = recall

    def add(self, other):
        """
        Add the contents of the other.

        Args:
            self: (todo): write your description
            other: (todo): write your description
        """
        self.prec.add(other.prec)
        self.recall.add(other.recall)

    @property
    def value(self):
        """
        Return the record value.

        Args:
            self: (todo): write your description
        """
        prec = self.prec.value
        recall = self.recall.value
        if not prec + recall:
            return 0
        return 2 * prec * recall / (prec + recall)

    def reset(self):
        """
        Reset the record.

        Args:
            self: (todo): write your description
        """
        self.prec.reset()
        self.recall.reset()


def topk_acc(pred, target, ks=(1, 2, 4, 8), mask=None):
    """
    Computes the accuracy.

    Args:
        pred: (todo): write your description
        target: (todo): write your description
        ks: (todo): write your description
        mask: (array): write your description
    """
    k = max(ks)
    pred = pred.topk(
        k,
        dim=-1,
        largest=True,
        sorted=True
    ).indices

    if mask is None:
        mask = mask_like(target)

    target = target[mask]
    pred = pred[mask].view(-1, k)  # restore shape

    pred = pred.t()  # k x tests
    target = target.expand_as(pred)  # k x tests

    correct = (pred == target)
    total = mask.sum().item()
    for k in ks:
        count = correct[:k].sum().item()
        yield Acc(count, total)


def acc(a, b, mask=None):
    """
    Accumulative accuracy.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        mask: (array): write your description
    """
    if mask is None:
        mask = mask_like(a)

    a = a[mask]
    b = b[mask]
    correct = (a == b).sum().item()
    total = len(a)
    return Acc(correct, total)


class BatchScore(Record):
    __attributes__ = ['loss']


class ScoreMeter(Record):
    __attributes__ = ['loss']

    def extend(self, scores):
        """
        Extend score to the list of scores.

        Args:
            self: (todo): write your description
            scores: (todo): write your description
        """
        for score in scores:
            self.add(score)


###########
#
#   MLM
#
#######


class MLMBatchScore(BatchScore):
    __attributes__ = ['loss', 'ks']

    def __init__(self, loss, ks):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            ks: (int): write your description
        """
        self.loss = loss
        self.ks = ks


class MLMScoreMeter(ScoreMeter):
    __attributes__ = ['loss', 'ks']

    def __init__(self, loss=None, ks=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            ks: (int): write your description
        """
        if not loss:
            loss = Mean()
        if not ks:
            ks = {}
        self.loss = loss
        self.ks = ks

    def add(self, score):
        """
        Add a loss.

        Args:
            self: (todo): write your description
            score: (int): write your description
        """
        self.loss.add(score.loss)
        for k, score in score.ks.items():
            if k not in self.ks:
                self.ks[k] = score
            else:
                self.ks[k].add(score)

    def reset(self):
        """
        Reset all losses.

        Args:
            self: (todo): write your description
        """
        self.loss.reset()
        for k in self.ks:
            self.ks[k].reset()

    def write(self, board):
        """
        Write out loss.

        Args:
            self: (todo): write your description
            board: (todo): write your description
        """
        board.add_scalar('01_loss', self.loss.value)
        for index, k in enumerate(self.ks, 2):
            key = '%02d_top%d' % (index, k)
            score = self.ks.get(k)
            if score:
                board.add_scalar(key, score.value)


def score_mlm_batch(batch, ks=(1, 2, 4, 8)):
    """
    Compute the hmm score.

    Args:
        batch: (todo): write your description
        ks: (todo): write your description
    """
    scores = ()
    if ks:
        scores = topk_acc(batch.pred, batch.target, ks)
    return MLMBatchScore(
        batch.loss.item(),
        ks=dict(zip(ks, scores))
    )


def score_mlm_batches(batches):
    """
    Iterate over - seqm batches return a batch.

    Args:
        batches: (todo): write your description
    """
    for batch in batches:
        yield score_mlm_batch(batch)


#######
#
#   NER
#
######


class NERBatchScore(BatchScore):
    __attributes__ = ['loss', 'types']

    def __init__(self, loss, types=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            types: (todo): write your description
        """
        self.loss = loss
        if not types:
            types = {}
        self.types = types


class NERScoreMeter(ScoreMeter):
    __attributes__ = ['loss', 'types']

    def __init__(self, loss=None, types=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            types: (todo): write your description
        """
        if not loss:
            loss = Mean()
        if not types:
            types = {}
        self.loss = loss
        self.types = types

    def add(self, score):
        """
        Add the given score.

        Args:
            self: (todo): write your description
            score: (int): write your description
        """
        self.loss.add(score.loss)
        for type, score in score.types.items():
            if type not in self.types:
                self.types[type] = score
            else:
                self.types[type].add(score)

    def reset(self):
        """
        Reset all losses.

        Args:
            self: (todo): write your description
        """
        self.loss.reset()
        for type in self.types:
            self.types[type].reset()

    def write(self, board):
        """
        Write out the loss.

        Args:
            self: (todo): write your description
            board: (todo): write your description
        """
        board.add_scalar('01_loss', self.loss.value)
        for index, type in enumerate(self.types, 2):
            key = '%02d_%s' % (index, type)
            score = self.types.get(type)
            if score:
                board.add_scalar(key, score.value)


def tag_f1(preds, targets, type):
    """
    Compute the f1 score.

    Args:
        preds: (array): write your description
        targets: (list): write your description
        type: (todo): write your description
    """
    score = F1()
    preds = list(bio_io(select_type_tags(preds, type)))
    targets = list(bio_io(select_type_tags(targets, type)))
    for pred, target in zip(preds, targets):
        pred, _ = parse_bio(pred)
        target, _ = parse_bio(target)
        if pred == I:
            score.prec.total += 1
            if target == I:
                score.prec.correct += 1
        if target == I:
            score.recall.total += 1
            if pred == I:
                score.recall.correct += 1
    return score


def decode_tags(seqs, tags_vocab):
    """
    Decode a sequence of strings.

    Args:
        seqs: (todo): write your description
        tags_vocab: (todo): write your description
    """
    return [
        tags_vocab.decode(id)
        for seq in seqs
        for id in seq
    ]


def score_ner_batch(batch, tags_vocab):
    """
    Compute the batch score for a batch.

    Args:
        batch: (todo): write your description
        tags_vocab: (todo): write your description
    """
    score = NERBatchScore(batch.loss)
    preds = decode_tags(batch.pred, tags_vocab)
    targets = decode_tags(batch.target, tags_vocab)
    for type in tags_vocab.types:
        score.types[type] = tag_f1(preds, targets, type)
    return score


###########
#
#   MORPH
#
######


class MorphBatchScore(BatchScore):
    __attributes__ = ['loss', 'acc']

    def __init__(self, loss, acc):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            acc: (todo): write your description
        """
        self.loss = loss
        self.acc = acc


class MorphScoreMeter(ScoreMeter):
    __attributes__ = ['loss', 'acc']

    def __init__(self, loss=None, acc=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            acc: (todo): write your description
        """
        if not loss:
            loss = Mean()
        if not acc:
            acc = Acc()
        self.loss = loss
        self.acc = acc

    def add(self, score):
        """
        Add the loss.

        Args:
            self: (todo): write your description
            score: (int): write your description
        """
        self.loss.add(score.loss)
        self.acc.add(score.acc)

    def reset(self):
        """
        Reset the loss.

        Args:
            self: (todo): write your description
        """
        self.loss.reset()
        self.acc.reset()

    def write(self, board):
        """
        Write loss to tensorboard.

        Args:
            self: (todo): write your description
            board: (todo): write your description
        """
        board.add_scalar('01_loss', self.loss.value)
        board.add_scalar('02_acc', self.acc.value)


def score_morph_batch(batch):
    """
    Score a batch score of a batch.

    Args:
        batch: (todo): write your description
    """
    return MorphBatchScore(
        batch.loss.item(),
        acc(batch.pred, batch.target)
    )


########
#
#   SYNTAX
#
#######


class SyntaxBatchScore(Record):
    __attributes__ = ['loss', 'uas', 'las']

    def __init__(self, loss, uas=None, las=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            uas: (todo): write your description
            las: (todo): write your description
        """
        self.loss = loss
        self.uas = uas
        self.las = las


class SyntaxScoreMeter(Record):
    __attributes__ = ['loss', 'uas', 'las']

    def __init__(self, loss=None, uas=None, las=None):
        """
        Initialize loss.

        Args:
            self: (todo): write your description
            loss: (todo): write your description
            uas: (todo): write your description
            las: (todo): write your description
        """
        if not loss:
            loss = Mean()
        if not uas:
            uas = Acc()
        if not las:
            las = Acc()
        self.loss = loss
        self.uas = uas
        self.las = las

    def add(self, score):
        """
        Add a score : pyas.

        Args:
            self: (todo): write your description
            score: (int): write your description
        """
        self.loss.add(score.loss)
        self.uas.add(score.uas)
        self.las.add(score.las)

    def reset(self):
        """
        Reset the loss.

        Args:
            self: (todo): write your description
        """
        self.loss.reset()
        self.uas.reset()
        self.las.reset()

    def write(self, board):
        """
        Writes : board to loss.

        Args:
            self: (todo): write your description
            board: (todo): write your description
        """
        board.add_scalar('01_loss', self.loss.value)
        board.add_scalar('02_uas', self.uas.value)
        board.add_scalar('03_las', self.las.value)


def uas(pred, target, mask=None):
    """
    Uassembles of the target.

    Args:
        pred: (array): write your description
        target: (str): write your description
        mask: (array): write your description
    """
    if mask is None:
        mask = mask_like(target)

    pred = pred[mask]
    target = target[mask]

    total = len(pred)
    correct = (pred == target).sum().item()
    return Acc(correct, total)


def las(head_pred, head_target, rel_pred, rel_target, mask=None):
    """
    Compute the head of head.

    Args:
        head_pred: (bool): write your description
        head_target: (list): write your description
        rel_pred: (str): write your description
        rel_target: (str): write your description
        mask: (array): write your description
    """
    if mask is None:
        mask = mask_like(head_target)

    head_pred = head_pred[mask]
    head_target = head_target[mask]
    rel_pred = rel_pred[mask]
    rel_target = rel_target[mask]

    total = len(head_pred)
    match = (head_pred == head_target) & (rel_pred == rel_target)
    correct = match.sum().item()
    return Acc(correct, total)


def score_syntax_batch(batch):
    """
    Return the score of a batch.

    Args:
        batch: (todo): write your description
    """
    input, target, loss, pred = batch
    return SyntaxBatchScore(
        loss.item(),
        uas(pred.head_id, target.head_id, target.mask),
        las(
            pred.head_id, target.head_id,
            pred.rel_id, target.rel_id,
            target.mask
        )
    )
