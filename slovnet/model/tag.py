
from torch import nn
from torch import functional as F

from .crf import CRF


class TagModel(nn.Module):
    pass


class LinearTagModel(TagModel):
    def __init__(self, input_dim, tags_num):
        super(LinearTagModel, self).__init__()
        self.proj = nn.Linear(
            input_dim,
            tags_num
        )

    def forward(self, input):
        proj = self.proj(input)
        return F.log_softmax(proj, dim=-1)

    def loss(self, input, target):
        pred = self.proj(input)  # batch x seq x tags
        pred = pred.view(-1, self.tags_num)  # ... x tags
        target = target.flatten()  # # batch x seq -> ...
        return F.nll_loss(pred, target)


class CRFTagModel(TagModel):
    def __init__(self, input_dim, tags_num):
        super(CRFTagModel, self).__init__()
        self.input_dim = input_dim
        self.tags_num = tags_num

        self.proj = nn.Linear(input_dim, tags_num)
        self.crf = CRF(tags_num)

    def forward(self, input):
        proj = self.proj(input)
        return self.crf(proj)

    def loss(self, input, target):
        proj = self.proj(input)
        return self.crf.loss(proj, target)
