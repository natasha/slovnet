
import torch
from torch import nn

from slovnet.mask import mask_like

from .base import Module


class CRF(Module):
    # https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py

    def __init__(self, tags_num):
        super(CRF, self).__init__()
        self.tags_num = tags_num
        self.transitions = nn.Parameter(torch.empty(tags_num, tags_num))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def extra_repr(self):
        return 'tags_num=%d' % self.tags_num

    def forward(self, emissions, tags, mask=None):
        if mask is None:
            mask = mask_like(tags)

        emissions = emissions.transpose(1, 0)  # seq x batch x tags
        tags = tags.transpose(1, 0)  # seq x batch
        mask = mask.transpose(1, 0)

        log_likelihood = (
            self.score(emissions, tags, mask)
            - self.normalization(emissions, mask)
        )  # batch
        return -torch.mean(log_likelihood)  # 1

    def score(self, emissions, tags, mask):
        seq_len, batch_size = tags.shape
        batch_range = torch.arange(batch_size)
        score = emissions[0, batch_range, tags[0]]  # batch
        for index in range(1, seq_len):
            score += (
                self.transitions[tags[index - 1], tags[index]]
                + emissions[index, batch_range, tags[index]]
            ) * mask[index]
        return score

    def normalization(self, emissions, mask):
        seq_len, batch_size, tags_num = emissions.shape
        score = emissions[0]
        for index in range(1, seq_len):
            score_ = score.view(batch_size, tags_num, 1)
            emissions_ = emissions[index].view(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            score_ = torch.logsumexp(score_, dim=-2)  # batch x tags
            mask_ = mask[index].view(batch_size, 1)
            score = torch.where(mask_, score_, score)
        return torch.logsumexp(score, dim=-1)  # batch

    def decode(self, emissions, mask=None):
        batch_size, seq_len, tags_num = emissions.shape
        if mask is None:
            mask = torch.ones(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=emissions.device
            )

        emissions = emissions.transpose(1, 0)
        mask = mask.transpose(1, 0)

        history = []
        score = emissions[0]
        for index in range(1, seq_len):
            score_ = score.view(batch_size, tags_num, 1)
            emissions_ = emissions[index].view(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            score_, indexes = torch.max(score_, dim=-2)  # batch x tags
            mask_ = mask[index].view(batch_size, 1)
            score = torch.where(mask_, score_, score)
            history.append(indexes)

        sizes = mask.sum(0) - 1
        batch = []
        for index in range(batch_size):
            best = score[index].argmax()
            tags = [best]
            size = sizes[index]
            for indexes in reversed(history[:size]):
                best = indexes[index][best]
                tags.append(best)
            tags.reverse()
            batch.append(torch.tensor(tags))
        return batch
