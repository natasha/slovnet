
from .record import Record


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


def topk_acc(pred, target, ks=(1, 2, 4, 8), ignore_id=-100):
    k = max(ks)
    pred = pred.topk(
        k,
        dim=-1,
        largest=True,
        sorted=True
    ).indices

    pred = pred.view(-1, k)
    target = target.flatten()

    mask = (target != ignore_id)
    target = target[mask]
    pred = pred[mask].view(-1, k)  # restore shape

    pred = pred.t()  # k x tests
    target = target.expand_as(pred)  # k x tests

    correct = (pred == target)
    total = mask.sum().item()
    for k in ks:
        count = correct[:k].sum().item()
        yield Share(count, total)


class BatchScore(Record):
    __attributes__ = ['loss']


class ScoreMeter(Record):
    __attributes__ = ['loss']

    def extend(self, scores):
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
        self.loss = loss
        self.ks = ks


class MLMScoreMeter(ScoreMeter):
    __attributes__ = ['loss', 'ks']

    def __init__(self, loss=None, ks=None):
        if not loss:
            loss = Mean()
        if not ks:
            ks = {}
        self.loss = loss
        self.ks = ks

    def add(self, score):
        self.loss.add(score.loss)
        for k, score in score.ks.items():
            if k not in self.ks:
                self.ks[k] = score
            else:
                self.ks[k].update(score)

    def reset(self):
        self.loss.reset()
        for k in self.ks:
            self.ks[k].reset()

    def write(self, board):
        board.add_scalar('01_loss', self.loss.value)
        for index, k in enumerate(self.ks, 2):
            key = '%02d_top%d' % (index, k)
            score = self.ks.get(k)
            if score:
                board.add_scalar(key, score.value)
