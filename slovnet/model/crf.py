
import torch
from torch import nn


class CRF(nn.Module):
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

    def forward(self, emissions):
        emissions = emissions.transpose(1, 0)
        seq_len, batch_size, tags_num = emissions.shape

        history = []
        score = emissions[0]
        for index in range(1, seq_len):
            score_ = score.view(batch_size, tags_num, 1)
            emissions_ = emissions[index].view(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            score, indexes = torch.max(score_, dim=-2)  # batch x tags, tags
            history.append(indexes)

        batch = []
        for index in range(batch_size):
            best = score[index].argmax()
            tags = [best]
            for indexes in reversed(history):
                best = indexes[index][best]
                tags.append(best)
            tags.reverse()
            batch.append(tags)
        return torch.tensor(batch).to(emissions.device)

    def loss(self, emissions, tags):
        emissions = emissions.transpose(1, 0)  # seq x batch x tags
        tags = tags.transpose(1, 0)  # seq x batch

        log_likelihood = (
            self.score(emissions, tags)
            - self.normalization(emissions)
        )  # batch
        return -torch.mean(log_likelihood)  # 1

    def score(self, emissions, tags):
        seq_len, batch_size = tags.shape
        batch_range = torch.arange(batch_size)
        score = emissions[0, batch_range, tags[0]]  # batch
        for index in range(1, seq_len):
            score += self.transitions[tags[index - 1], tags[index]]
            score += emissions[index, batch_range, tags[index]]
        return score

    def normalization(self, emissions):
        seq_len, batch_size, tags_num = emissions.shape
        score = emissions[0]
        for index in range(1, seq_len):
            score_ = score.view(batch_size, tags_num, 1)
            emissions_ = emissions[index].view(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            score = torch.logsumexp(score_, dim=-2)  # batch x tags
        return torch.logsumexp(score, dim=-1)  # batch
