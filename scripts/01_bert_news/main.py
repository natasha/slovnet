
import re
from os import listdir, makedirs
from os.path import exists, dirname, join
from random import seed, shuffle

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from apex import amp

from corus import (
    load_buriy_news,
    load_taiga_fontanka,
    load_ods_gazeta,
    load_ods_interfax,
    load_lenta
)

from slovnet.io import (
    load_lines,
    dump_lines
)
from slovnet.env import Env
from slovnet.s3 import s3_client
from slovnet.device import get_device
from slovnet.board import Board
from slovnet.loop import every

from slovnet.model.state import (
    load_model as load_submodel,
    dump_model as dump_submodel
)
from slovnet.model.bert import (
    BERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTMLMHead,
    BERTMLM
)
from slovnet.vocab import BERTVocab
from slovnet.encoders.bert import BERTMLMEncoder
from slovnet.score import (
    MLMScoreMeter,
    score_mlm_batch as score_batch
)
from slovnet.loss import flatten_cross_entropy


env = Env.from_file()


########
#
#   LOOP
#
######


def score_batches(batches):
    for batch in batches:
        yield score_batch(batch)


def process_batch(model, criterion, batch):
    pred = model(batch.input)
    loss = criterion(pred, batch.target)
    return batch.processed(loss, pred)


def infer_batches(model, criterion, batches):
    training = model.training
    model.eval()
    with torch.no_grad():
        for batch in batches:
            yield process_batch(model, criterion, batch)
    model.train(training)


########
#
#   S3
#
######


s3 = s3_client(env.s3_key_id, env.s3_key)


def upload(path):
    s3.upload_file(path, env.s3_bucket, join('01_bert_news', path))


def download(path):
    makedirs(dirname(path), exist_ok=True)
    s3.download_file(env.s3_bucket, join('01_bert_news', path), path)


########
#
#  STATE
#
#######


def load_model(model, dir='model'):
    load_submodel(model.emb, join(dir, 'emb.pt'))
    load_submodel(model.encoder, join(dir, 'encoder.pt'))
    load_submodel(model.mlm, join(dir, 'mlm.pt'))


def dump_model(model, dir='model'):
    makedirs(dir, exist_ok=True)
    dump_submodel(model.emb, join(dir, 'emb.pt'))
    dump_submodel(model.encoder, join(dir, 'encoder.pt'))
    dump_submodel(model.mlm, join(dir, 'mlm.pt'))
