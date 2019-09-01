
import torch

from .record import Record
from .bio import io_spans
from .markup import SpanMarkup
from .batch import Batch


class NERTagger(Record):
    __attributes__ = ['tokenizer', 'token_encoder', 'tags_vocab', 'model', 'device']

    def __init__(self, tokenizer, token_encoder, tags_vocab, model, device):
        self.tokenizer = tokenizer
        self.token_encoder = token_encoder
        self.tags_vocab = tags_vocab
        self.model = model
        self.device = device

    def __call__(self, text):
        tokens = list(self.tokenizer(text))

        ids = self.token_encoder.map(tokens)
        batch = Batch.from_token_encoder(ids)  # (1 x seq, 1 x seq, ...)
        batch = batch.to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(batch.input)  # 1 x seq
            pred = pred.squeeze()  # seq
            pred = pred.tolist()

        tags = [self.tags_vocab.decode(_) for _ in pred]
        spans = list(io_spans(tokens, tags))  # in case of broken bio
        return SpanMarkup(text, spans)
