
from os import getenv, environ
from os.path import exists, join, expanduser
from random import seed, sample, randint, uniform
from subprocess import run

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from naeval.ner.datasets import (
    load_factru,
    load_ne5,
)

from slovnet.s3 import S3
from slovnet.io import (
    format_jl,
    parse_jl,

    load_gz_lines,
    dump_gz_lines
)
from slovnet.board import (
    TensorBoard,
    LogBoard,
    MultiBoard
)
from slovnet.const import (
    TRAIN, TEST,
    PER, LOC, ORG,
    CUDA0,
)
from slovnet.token import tokenize

from slovnet.model.bert import (
    RuBERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTNERHead,
    BERTNER
)
from slovnet.markup import (
    SpanMarkup,
    show_span_markup
)
from slovnet.vocab import BERTVocab, BIOTagsVocab
from slovnet.encoders.bert import BERTNERTrainEncoder
from slovnet.score import (
    NERBatchScore,
    NERScoreMeter,
    score_ner_batch
)
from slovnet.mask import (
    Masked,
    split_masked,
    pad_masked
)


DATA_DIR = 'data'
MODEL_DIR = 'model'
BERT_DIR = 'bert'
RAW_DIR = join(DATA_DIR, 'raw')

CORUS_NE5 = join(RAW_DIR, 'Collection5')
CORUS_FACTRU = join(RAW_DIR, 'factRuEval-2016-master')

NE5 = join(DATA_DIR, 'ne5.jl.gz')
FACTRU = join(DATA_DIR, 'factru.jl.gz')

S3_DIR = '02_bert_ner'
S3_NE5 = join(S3_DIR, NE5)
S3_FACTRU = join(S3_DIR, FACTRU)

VOCAB = 'vocab.txt'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
NER = 'ner.pt'

BERT_VOCAB = join(BERT_DIR, VOCAB)
BERT_EMB = join(BERT_DIR, EMB)
BERT_ENCODER = join(BERT_DIR, ENCODER)

S3_RUBERT_DIR = '01_bert_news/rubert'
S3_MLM_DIR = '01_bert_news/model'
S3_BERT_VOCAB = join(S3_RUBERT_DIR, VOCAB)
S3_BERT_EMB = join(S3_MLM_DIR, EMB)
S3_BERT_ENCODER = join(S3_MLM_DIR, ENCODER)

MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_NER = join(MODEL_DIR, NER)

S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_NER = join(S3_DIR, MODEL_NER)

BOARD_NAME = getenv('board_name', '02_bert_ner')
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

SEED = int(getenv('seed', 72))
DEVICE = getenv('device', CUDA0)
BERT_LR = float(getenv('bert_lr', 0.000045))
LR = float(getenv('lr', 0.0075))
LR_GAMMA = float(getenv('lr_gamma', 0.45))
EPOCHS = int(getenv('epochs', 5))


def process_batch(model, criterion, batch):
    input, target = batch

    pred = model(input.value)
    pred = pad_masked(pred, input.mask)
    mask = pad_masked(input.mask, input.mask)

    loss = criterion(pred, target.value, target.mask)

    pred = Masked(pred, mask)
    return batch.processed(loss, pred)
