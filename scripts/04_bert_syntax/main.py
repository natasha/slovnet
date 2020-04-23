
from os import getenv, environ
from os.path import exists, join
from itertools import chain, islice as head
from random import seed, sample, randint, uniform
from subprocess import run

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from naeval.syntax.datasets import load_dataset

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
    BERTSyntaxHead,
    BERTSyntaxRel,
    BERTSyntax
)
from slovnet.markup import SyntaxMarkup
from slovnet.vocab import BERTVocab, Vocab
from slovnet.encoders.bert import BERTSyntaxTrainEncoder
from slovnet.loss import masked_flatten_cross_entropy
from slovnet.score import (
    SyntaxScoreMeter,
    score_syntax_batch
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

S3_DIR = '04_bert_syntax'
S3_NEWS = join(S3_DIR, NEWS)
S3_FICTION = join(S3_DIR, FICTION)

VOCAB = 'vocab.txt'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
HEAD = 'head.pt'
REL = 'rel.pt'

BERT_VOCAB = join(BERT_DIR, VOCAB)
BERT_EMB = join(BERT_DIR, EMB)
BERT_ENCODER = join(BERT_DIR, ENCODER)

S3_RUBERT_DIR = '01_bert_news/rubert'
S3_MLM_DIR = '01_bert_news/model'
S3_BERT_VOCAB = join(S3_RUBERT_DIR, VOCAB)
S3_BERT_EMB = join(S3_MLM_DIR, EMB)
S3_BERT_ENCODER = join(S3_MLM_DIR, ENCODER)

RELS_VOCAB = join(MODEL_DIR, 'rels_vocab.txt')
MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_HEAD = join(MODEL_DIR, HEAD)
MODEL_REL = join(MODEL_DIR, REL)

S3_RELS_VOCAB = join(S3_DIR, RELS_VOCAB)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_HEAD = join(S3_DIR, MODEL_HEAD)
S3_MODEL_REL = join(S3_DIR, MODEL_REL)

BOARD_NAME = getenv('board_name', '04_bert_syntax_01')
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

SEED = int(getenv('seed', 50))
DEVICE = getenv('device', CUDA0)
BERT_LR = float(getenv('bert_lr', 0.000058))
LR = float(getenv('lr', 0.00012))
LR_GAMMA = float(getenv('lr_gamma', 0.29))
EPOCHS = int(getenv('epochs', 2))


def process_batch(model, criterion, batch):
    input, target = batch

    pred = model(
        input.word_id, input.word_mask, input.pad_mask,
        target.mask, target.head_id
    )

    loss = (
        criterion(pred.head_id, target.head_id, target.mask)
        + criterion(pred.rel_id, target.rel_id, target.mask)
    )

    pred.head_id = model.head.decode(pred.head_id, target.mask)
    pred.rel_id = model.rel.decode(pred.rel_id, target.mask)

    return batch.processed(loss, pred)
