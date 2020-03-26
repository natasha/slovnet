
import torch

from .mask import (
    Masked,
    pad_masked
)


def every(step, period):
    return step > 0 and step % period == 0


########
#
#   BERT MLM
#
#########


def process_bert_mlm_batch(model, criterion, batch):
    pred = model(batch.input)
    loss = criterion(pred, batch.target)
    return batch.processed(loss, pred)


def infer_bert_mlm_batches(model, criterion, batches):
    training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            yield process_bert_mlm_batch(model, criterion, batch)
    model.train(training)


########
#
#   BERT NER
#
########


def process_bert_ner_batch(model, criterion, batch):
    input, target = batch

    pred = model(input.value)
    pred = pad_masked(pred, input.mask)
    mask = pad_masked(input.mask, input.mask)

    loss = criterion(pred, target.value, target.mask)

    pred = Masked(pred, mask)
    return batch.processed(loss, pred)


########
#
#   BERT MORPH
#
########


def process_bert_morph_batch(model, criterion, batch):
    input, target = batch

    pred = model(input.value)
    pred = pad_masked(pred, input.mask)
    mask = pad_masked(input.mask, input.mask)

    loss = criterion(pred, target.value, target.mask)

    pred = pred.argmax(-1)
    pred = Masked(pred, mask)
    return batch.processed(loss, pred)
