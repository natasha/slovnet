
from .record import Record
from .bio import (
    I,
    bio_io, parse_bio,
    select_type_tags
)


class Share(Record):
    __attributes__ = ['correct', 'total']

    def __init__(self, correct=0, total=0):
        self.correct = correct
        self.total = total

    def update(self, other):
        self.correct += other.correct
        self.total += other.total

    @property
    def value(self):
        if not self.total:
            return 0
        return self.correct / self.total


class Mean(Record):
    __attributes__ = ['accum', 'count']

    def __init__(self, accum=0, count=0):
        self.accum = accum
        self.count = count

    def add(self, value):
        self.accum += value
        self.count += 1

    @property
    def value(self):
        if not self.count:
            return 0
        return self.accum / self.count


class TagScore(Record):
    __attributes__ = ['prec', 'recall']

    def __init__(self, prec=None, recall=None):
        if not prec:
            prec = Share()
        self.prec = prec
        if not recall:
            recall = Share()
        self.recall = recall

    def update(self, other):
        self.prec.update(other.prec)
        self.recall.update(other.recall)

    @property
    def f1(self):
        prec = self.prec.value
        recall = self.recall.value
        if not prec + recall:
            return 0
        return 2 * prec * recall / (prec + recall)


class MarkupScore(Record):
    __attributes__ = ['types']

    def __init__(self, types=None):
        if not types:
            types = {}
        self.types = types

    def add(self, type, score):
        self.types[type] = score

    def get(self, type):
        return self.types.get(type)

    def update(self, other):
        for type, score in other.types.items():
            if type not in self.types:
                self.types[type] = score
            else:
                self.types[type].update(score)


class BatchScore(MarkupScore):
    __attributes__ = ['loss', 'types']

    def __init__(self, loss, types=None):
        self.loss = loss
        super(BatchScore, self).__init__(types)


def eval_tags_type(preds, targets, type):
    score = TagScore()
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


def eval_markup(pred, target, types):
    score = MarkupScore()
    for type in types:
        score.add(
            type,
            eval_tags_type(pred.tags, target.tags, type)
        )
    return score


def eval_markups(preds, targets, types):
    for pred, target in zip(preds, targets):
        yield eval_markup(pred, target, types)


def avg_markup_scores(scores):
    accum = MarkupScore()
    for score in scores:
        accum.update(score)
    return accum


def decode_tags(tags_vocab, ids):
    ids = ids.flatten().tolist()
    for id in ids:
        yield tags_vocab.decode(id)


def eval_batch(tags_vocab, batch):
    score = BatchScore(batch.loss)
    if batch.pred is None:
        # for performance just loss may be computed in a loop
        return score

    preds = list(decode_tags(tags_vocab, batch.pred))
    targets = list(decode_tags(tags_vocab, batch.target))
    for type in tags_vocab.types:
        score.add(
            type,
            eval_tags_type(preds, targets, type)
        )
    return score


def eval_batches(tags_vocab, batches):
    for batch in batches:
        yield eval_batch(tags_vocab, batch)


def avg_batch_scores(scores):
    accum = BatchScore(loss=Mean())
    for score in scores:
        accum.loss.add(score.loss)
        accum.update(score)
    return BatchScore(
        accum.loss.value,
        accum.types
    )
