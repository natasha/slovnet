
from os import getenv, environ
from os.path import exists, join
from random import seed, sample, randint, uniform
from itertools import islice as head
from subprocess import run

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from naeval.morph.datasets import load_dataset

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
    PAD, CUDA0,
)

from slovnet.model.bert import (
    RuBERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTMorphHead,
    BERTMorph
)
from slovnet.markup import MorphMarkup
from slovnet.vocab import BERTVocab, Vocab
from slovnet.encoders.bert import BERTMorphTrainEncoder
from slovnet.loss import masked_flatten_cross_entropy
from slovnet.batch import ProcessedBatch
from slovnet.score import (
    MorphScoreMeter,
    score_morph_batch
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

NEWS = join(DATA_DIR, 'news.jl.gz')
FICTION = join(DATA_DIR, 'fiction.jl.gz')
GRAMRU_DIR = join(RAW_DIR, 'GramEval2020-master')
GRAMRU_FILES = {
    NEWS: [
        'dataOpenTest/GramEval2020-RuEval2017-Lenta-news-dev.conllu',
        'dataTrain/MorphoRuEval2017-Lenta-train.conllu',
    ],
    FICTION: [
        'dataOpenTest/GramEval2020-SynTagRus-dev.conllu',
        'dataTrain/GramEval2020-SynTagRus-train-v2.conllu',
        'dataTrain/MorphoRuEval2017-JZ-gold.conllu'
    ],
}

S3_DIR = '03_bert_morph'
S3_NEWS = join(S3_DIR, NEWS)
S3_FICTION = join(S3_DIR, FICTION)

VOCAB = 'vocab.txt'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
MORPH = 'morph.pt'

BERT_VOCAB = join(BERT_DIR, VOCAB)
BERT_EMB = join(BERT_DIR, EMB)
BERT_ENCODER = join(BERT_DIR, ENCODER)

S3_RUBERT_DIR = '01_bert_news/rubert'
S3_MLM_DIR = '01_bert_news/model'
S3_BERT_VOCAB = join(S3_RUBERT_DIR, VOCAB)
S3_BERT_EMB = join(S3_MLM_DIR, EMB)
S3_BERT_ENCODER = join(S3_MLM_DIR, ENCODER)

TAGS_VOCAB = join(MODEL_DIR, 'tags_vocab.txt')
MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_MORPH = join(MODEL_DIR, MORPH)

S3_TAGS_VOCAB = join(S3_DIR, TAGS_VOCAB)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_MORPH = join(S3_DIR, MODEL_MORPH)

BOARD_NAME = getenv('board_name', '03_bert_morph_02')
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

SEED = int(getenv('seed', 1))
DEVICE = getenv('device', CUDA0)
BERT_LR = float(getenv('bert_lr', 0.0002))
LR = float(getenv('lr', 0.001))
LR_GAMMA = float(getenv('lr_gamma', 0.8))
EPOCHS = int(getenv('epochs', 5))


def process_batch(model, criterion, batch):
    input, target = batch

    pred = model(input.value)
    pred = pad_masked(pred, input.mask)
    mask = pad_masked(input.mask, input.mask)

    loss = criterion(pred, target.value, target.mask)

    pred = model.morph.decode(pred)
    pred = pred[mask]

    # unmask pred, target to fit in score batch
    target = target.value[target.mask]

    return ProcessedBatch(input, target, loss, pred)

