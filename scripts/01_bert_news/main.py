
import re
from os import listdir, makedirs
from os.path import exists, join, expanduser
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
from slovnet.s3 import S3
from slovnet.board import Board
from slovnet.loop import every
from slovnet.const import CUDA0

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
from slovnet.loss import flatten_cross_entropy as criterion


#######
#
#  PARTS
#
#####


PARTS = ['emb', 'encoder', 'mlm']


def load_model(model, parts=PARTS):
    for part in parts:
        load_submodel(
            getattr(model, part),
            'model/%s.pt' % part
        )


def dump_model(model, parts=PARTS):
    for part in parts:
        dump_submodel(
            getattr(model, part),
            'model/%s.pt' % part
        )


def upload_model(s3, parts=PARTS):
    for part in parts:
        s3.upload(
            'model/%s.pt' % part,
            '01_bert_news/model/%s.pt' % part
        )


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
