
import torch


def train_model(model, optimizer, batches):
    model.train()
    for batch in batches:
        optimizer.zero_grad()

        loss = model.loss(batch.input, batch.target)
        loss.backward()
        optimizer.step()

        yield batch.processed(loss, pred=None)


def infer_model(model, batches):
    model.eval()
    with torch.no_grad():
        for batch in batches:
            loss = model.loss(batch.input, batch.target)
            pred = model(batch.input)
            yield batch.processed(loss, pred)
