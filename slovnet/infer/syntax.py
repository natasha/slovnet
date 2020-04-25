
from slovnet.markup import SyntaxMarkup
from slovnet.mask import split_masked

from .base import Infer


class SyntaxDecoder:
    def __init__(self, rels_vocab):
        self.rels_vocab = rels_vocab

    def __call__(self, preds):
        for pred in preds:
            head_ids, rel_ids = pred
            ids = [str(_ + 1) for _ in range(len(head_ids))]
            head_ids = [str(_) for _ in head_ids.tolist()]
            rels = [self.rels_vocab.decode(_) for _ in rel_ids]
            yield ids, head_ids, rels


class SyntaxInfer(Infer):
    def process(self, inputs):
        for input in inputs:
            input = input.to(self.model.device)

            pred = self.model(input.word_id, input.shape_id, input.pad_mask)
            mask = ~input.pad_mask

            head_id = self.model.head.decode(pred.head_id, mask)
            head_id = split_masked(head_id, mask)

            rel_id = self.model.rel.decode(pred.rel_id, mask)
            rel_id = split_masked(rel_id, mask)

            yield from zip(head_id, rel_id)

    def __call__(self, items):
        inputs = self.encoder(items)
        preds = self.process(inputs)
        preds = self.decoder(preds)

        for item, pred in zip(items, preds):
            ids, head_ids, rels = pred
            tuples = zip(ids, item, head_ids, rels)
            yield SyntaxMarkup.from_tuples(tuples)
