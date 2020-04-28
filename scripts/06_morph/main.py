
from os import getenv, environ
from os.path import exists, join, expanduser
from random import seed, shuffle, sample, randint, uniform
from itertools import islice as head
from subprocess import run

from tqdm.notebook import tqdm as log_progress

import torch
from torch import optim

from nerus import load_nerus

from navec import Navec

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
    CUDA0, PAD,
    WORD, SHAPE, TAG
)
from slovnet.token import tokenize
from slovnet.shape import SHAPES

from slovnet.markup import (
    MorphToken,
    MorphMarkup,
    show_morph_markup
)
from slovnet.model.emb import (
    Embedding,
    NavecEmbedding
)
from slovnet.model.tag import (
    TagEmbedding,
    TagEncoder,
    MorphHead,
    Morph
)
from slovnet.loss import flatten_cross_entropy
from slovnet.vocab import Vocab
from slovnet.encoders.tag import TagTrainEncoder
from slovnet.score import (
    MorphBatchScore,
    MorphScoreMeter,
    score_morph_batch
)

from slovnet.exec.pack import (
    Meta,
    DumpPack
)
from slovnet import api


DATA_DIR = 'data'
MODEL_DIR = 'model'
NAVEC_DIR = 'navec'
RAW_DIR = join(DATA_DIR, 'raw')
S3_DIR = '06_morph'

RAW_NERUS = join(RAW_DIR, 'nerus_lenta.conllu.gz')
NERUS_TOTAL = 739346

NERUS = join(DATA_DIR, 'nerus.jl.gz')
S3_NERUS = join(S3_DIR, NERUS)

NAVEC_URL = 'https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar'
NAVEC = join(NAVEC_DIR, 'navec_news_v1_1B_250K_300d_100q.tar')

TAGS_VOCAB = join(MODEL_DIR, 'tags_vocab.txt')
MODEL_SHAPE = join(MODEL_DIR, 'shape.pt')
MODEL_ENCODER = join(MODEL_DIR, 'encoder.pt')
MODEL_MORPH = join(MODEL_DIR, 'morph.pt')

S3_TAGS_VOCAB = join(S3_DIR, TAGS_VOCAB)
S3_MODEL_SHAPE = join(S3_DIR, MODEL_SHAPE)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_MORPH = join(S3_DIR, MODEL_MORPH)

ID = 'slovnet_morph_news_v1'
PACK = ID + '.tar'
S3_PACK = join('packs', PACK)

BOARD_NAME = getenv('board_name', '06_morph')
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

SEED = int(getenv('seed', 65))
DEVICE = getenv('device', CUDA0)

SHAPE_DIM = int(getenv('shape_dim', 30))
LAYERS_NUM = int(getenv('layers_num', 3))
LAYER_DIM = int(getenv('layer_dim', 64))
KERNEL_SIZE = int(getenv('kernel_size', 3))

LR = float(getenv('lr', 0.0033))
LR_GAMMA = float(getenv('lr_gamma', 0.9))
EPOCHS = int(getenv('epochs', 5))

LAYER_DIMS = [
    LAYER_DIM * 2**_
    for _ in reversed(range(LAYERS_NUM))
]


def adapt_markup(record):
    return MorphMarkup([
        MorphToken(_.text, _.pos, _.feats)
        for _ in record.tokens
    ])


def process_batch(model, criterion, batch):
    input, target = batch

    pred = model(input.word_id, input.shape_id)
    loss = criterion(pred, target)

    pred = model.morph.decode(pred)

    return batch.processed(loss, pred)
