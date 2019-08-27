
from torch import nn

from .infer import InferMixin


class NERModel(nn.Module, InferMixin):
    def __init__(self,
                 word_model,
                 context_model,
                 tag_model):
        super(NERModel, self).__init__()
        self.word_model = word_model
        self.context_model = context_model
        self.tag_model = tag_model

    def forward(self, input):
        emb = self.word_model(input)  # batch x seq x emb_dim
        context = self.context_model(emb)  # batch x seq x hidden_dim
        return self.tag_model(context)  # batch x seq

    def loss(self, input, target):
        emb = self.word_model(input)
        context = self.context_model(emb)
        return self.tag_model.loss(context, target)
