
from torch import nn

from .infer import InferMixin


class ShapeEmbedding(nn.Embedding):
    def __init__(self, vocab_size, dim, pad_id):
        super(ShapeEmbedding, self).__init__(vocab_size, dim, pad_id)
        self.dim = dim


class WordModel(nn.Module):
    def __init__(self, word_emb, shape_emb):
        super(WordModel, self).__init__()
        self.word_emb = word_emb
        self.shape_emb = shape_emb
        self.dim = word_emb.dim + shape_emb.dim

    def forward(self, input):
        word_id, shape_id = input
        word_emb = self.word_emb(word_id)
        shape_emb = self.shape_emb(shape_id)
        return torch.cat([word_emb, shape_emb], dim=-1)



class ContextModel(nn.Module):
    pass


def CNNLayer(in_dim, out_dim, kernel_size):
    padding = (kernel_size - 1) // 2
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv1d(
            in_dim, out_dim, kernel_size,
            padding=padding
        )),
        ('relu', nn.ReLU()),
        ('norm', nn.BatchNorm1d(out_dim))
    ]))


def CNNLayers(input_dim, layer_dims, kernel_size):
    dims = [input_dim] + layer_dims
    for index in range(1, len(dims)):
        in_dim = dims[index - 1]
        out_dim = dims[index]
        yield CNNLayer(in_dim, out_dim, kernel_size)


class CNNContextModel(ContextModel):
    def __init__(self, input_dim, layer_dims, kernel_size):
        super(CNNContextModel, self).__init__()
        self.layers = nn.Sequential(*CNNLayers(
            input_dim, layer_dims, kernel_size
        ))
        self.dim = layer_dims[-1]

    def forward(self, input):  # batch x seq x emb
        input = input.transpose(2, 1)  # batch x emb x seq
        context = self.layers(input)  # batch x dim x seq
        context = context.transpose(2, 1)  # batch x seq x dim
        return context


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
