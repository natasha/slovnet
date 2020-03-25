

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
