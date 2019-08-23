
from torch.optim import lr_scheduler


class FinderLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, steps, last_epoch=-1):
        self.max_lr = max_lr
        self.steps = steps
        super(FinderLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / (self.steps - 1)
        return [
            _ * (self.max_lr / _) ** progress
            for _ in self.base_lrs
        ]

    @property
    def lr(self):
        lrs = self.get_lr()
        assert len(lrs) == 1
        return lrs[0]


def find_lr(model, optimizer, scheduler, batches):
    model.train()
    for step, batch in enumerate(batches):
        if step >= scheduler.steps:
            return

        optimizer.zero_grad()

        loss = model.loss(batch.input, batch.target)
        loss.backward()

        yield scheduler.lr, loss.item()

        optimizer.step()
        scheduler.step()


def exp_smoothing(values, a=0.1):
    previous = None
    for value in values:
        if previous is None:
            previous = value
        else:
            previous = value * a + previous * (1 - a)
        yield previous


def plot_find_lr(data, a=0.1, ax=None):
    from matplotlib import pyplot as plt

    if not ax:
        _, ax = plt.subplots()

    xs, ys = zip(*data)
    smooth_ys = list(exp_smoothing(ys, a))
    ax.plot(xs, ys)
    ax.plot(xs, smooth_ys)
    ax.set_xscale('log')
    ax.set_ylabel('loss')
