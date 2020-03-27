
import re
from os import listdir, makedirs
from os.path import exists, join, expanduser
from random import seed, shuffle

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from apex import amp
O2 = 'O2'

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
from slovnet.const import CUDA0

from slovnet.model.state import (
    load_model,
    dump_model
)
from slovnet.model.bert import (
    RuBERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTMLMHead,
    BERTMLM
)
from slovnet.vocab import BERTVocab
from slovnet.encoders.bert import BERTMLMEncoder
from slovnet.score import (
    MLMScoreMeter,
    score_mlm_batches as score_batches
)
from slovnet.loss import flatten_cross_entropy as criterion
from slovnet.loop import (
    every
    process_bert_mlm_batch as process_batch,
    infer_bert_mlm_batches as infer_batches
)


DATA_DIR = 'data'
MODEL_DIR = 'model'
RUBERT_DIR = 'rubert'
RAW_DIR = join(DATA_DIR, 'raw')

TRAIN = join(DATA_DIR, 'train.txt')
TEST = join(DATA_DIR, 'test.txt')

S3_DIR = '01_bert_news'
S3_TRAIN = join(S3_DIR, TRAIN)
S3_TEST = join(S3_DIR, TEST)

VOCAB = 'vocab.txt'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
MLM = 'mlm.pt'

RUBERT_VOCAB = join(RUBERT_DIR, VOCAB)
RUBERT_EMB = join(RUBERT_DIR, EMB)
RUBERT_ENCODER = join(RUBERT_DIR, ENCODER)
RUBERT_MLM = join(RUBERT_DIR, MLM)

S3_RUBERT_VOCAB = join(S3_DIR, RUBERT_VOCAB)
S3_RUBERT_EMB = join(S3_DIR, RUBERT_EMB)
S3_RUBERT_ENCODER = join(S3_DIR, RUBERT_ENCODER)
S3_RUBERT_MLM = join(S3_DIR, RUBERT_MLM)

MODEL_EMB = join(MODEL_DIR, EMB)
MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_MLM = join(MODEL_DIR, MLM)

S3_MODEL_EMB = join(S3_DIR, MODEL_EMB)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_MLM = join(S3_DIR, MODEL_MLM)

BOARD_NAME = '01_bert_news'
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'
