
from collections import OrderedDict

import torch
from torch import nn


class ShapeEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, pad_id):
        super(ShapeEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, dim, pad_id)
        self.dim = dim

    def forward(self, shape_id):
        return self.emb(shape_id)


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


class CNNContextModel(nn.Module):
    def __init__(self, input_dim, layer_dims, kernel_size):
        super(CNNContextModel, self).__init__()
        self.layers = nn.Sequential(*CNNLayers(
            input_dim, layer_dims, kernel_size
        ))
        self.dim = layer_dims[-1]

    def forward(self, input):  # batch x seq x emb_dim
        input = input.permute(0, 2, 1)  # batch x dim x seq
        context = self.layers(input)  # batch x dim x seq
        context = context.permute(0, 2, 1)  # batch x seq x dim
        return context


class RNNContextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers_count):
        super(RNNContextModel, self).__init__()
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers_count,
            bidirectional=True,
            batch_first=True
        )
        self.dim = hidden_dim * 2  # forward + backward

    def forward(self, input):
        context, _ = self.rnn(input)
        return context


class CRFTagModel(nn.Module):
    # https://github.com/kmkurn/pytorch-crf/blob/master/torchcrf/__init__.py

    def __init__(self, input_dim, tags_num):
        super(CRFTagModel, self).__init__()
        self.input_dim = input_dim
        self.tags_num = tags_num

        self.fc = nn.Linear(input_dim, tags_num)
        self.transitions = nn.Parameter(torch.empty(tags_num, tags_num))
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self):
        return '{name}(input_dim={input_dim}, tags_num={tags_num})'.format(
            name=self.__class__.__name__,
            input_dim=self.input_dim,
            tags_num=self.tags_num
        )

    def forward(self, input):
        emissions = self.fc(input)  # batch x seq x tags

        emissions = emissions.transpose(1, 0)  # seq x ...
        seq_len, batch_size, tags_num = emissions.shape

        history = []
        score = emissions[0]
        for index in range(1, seq_len):
            score_ = score.view(batch_size, tags_num, 1)
            emissions_ = emissions[index].view(batch_size, 1, tags_num)
            score_ = score_ + self.transitions + emissions_  # batch x tags x tags
            score, indices = torch.max(score_, dim=-2)  # batch x tags, tags
            history.append(indices)

        batch = []
        for index in range(batch_size):
            best = score[index].argmax()
            tags = [best]
            for indices in reversed(history):
                best = indices[index][best]
                tags.append(best)
            tags.reverse()
            batch.append(tags)
        return torch.tensor(batch).to(input.device)

    def loss(self, input, tags):
        emissions = self.fc(input)  # batch x seq x tags

        emissions = emissions.transpose(1, 0)  # seq x ...
        tags = tags.transpose(1, 0)  # seq x ...

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


class NERModel(nn.Module):
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
