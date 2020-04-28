
from os import getenv, environ
from os.path import exists, join, expanduser
from random import seed, shuffle, sample, randint, uniform
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
    WORD, SHAPE, REL
)
from slovnet.token import tokenize
from slovnet.shape import SHAPES

from slovnet.markup import (
    SyntaxToken,
    SyntaxMarkup,
    show_syntax_markup
)
from slovnet.model.emb import (
    Embedding,
    NavecEmbedding
)
from slovnet.model.syntax import (
    SyntaxEmbedding,
    SyntaxEncoder,
    SyntaxHead,
    SyntaxRel,
    Syntax
)
from slovnet.loss import masked_flatten_cross_entropy
from slovnet.vocab import Vocab
from slovnet.encoders.syntax import SyntaxTrainEncoder
from slovnet.score import (
    SyntaxBatchScore,
    SyntaxScoreMeter,
    score_syntax_batch
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
S3_DIR = '07_syntax'

RAW_NERUS = join(RAW_DIR, 'nerus_lenta.conllu.gz')
NERUS_TOTAL = 739346

NERUS = join(DATA_DIR, 'nerus.jl.gz')
S3_NERUS = join(S3_DIR, NERUS)

NAVEC_URL = 'https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar'
NAVEC = join(NAVEC_DIR, 'navec_news_v1_1B_250K_300d_100q.tar')

RELS_VOCAB = join(MODEL_DIR, 'rels_vocab.txt')
MODEL_SHAPE = join(MODEL_DIR, 'shape.pt')
MODEL_ENCODER = join(MODEL_DIR, 'encoder.pt')
MODEL_HEAD = join(MODEL_DIR, 'head.pt')
MODEL_REL = join(MODEL_DIR, 'rel.pt')

S3_RELS_VOCAB = join(S3_DIR, RELS_VOCAB)
S3_MODEL_SHAPE = join(S3_DIR, MODEL_SHAPE)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_HEAD = join(S3_DIR, MODEL_HEAD)
S3_MODEL_REL = join(S3_DIR, MODEL_REL)

ID = 'slovnet_syntax_news_v1'
PACK = ID + '.tar'
S3_PACK = join('packs', PACK)

BOARD_NAME = getenv('board_name', '07_syntax')
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'

SEED = int(getenv('seed', 17))
DEVICE = getenv('device', CUDA0)

SHAPE_DIM = int(getenv('shape_dim', 30))
LAYERS_NUM = int(getenv('layers_num', 3))
LAYER_DIM = int(getenv('layer_dim', 64))
KERNEL_SIZE = int(getenv('kernel_size', 3))

LR = float(getenv('lr', 0.0051))
LR_GAMMA = float(getenv('lr_gamma', 0.74))
EPOCHS = int(getenv('epochs', 3))

LAYER_DIMS = [
    LAYER_DIM * 2**_
    for _ in reversed(range(LAYERS_NUM))
]


def adapt_markup(record):
    return SyntaxMarkup([
        SyntaxToken(_.id, _.text, _.head_id, _.rel)
        for _ in record.tokens
    ])


def process_batch(model, criterion, batch):
    input, target = batch

    pred = model(
        input.word_id, input.shape_id, input.pad_mask,
        target.mask, target.head_id
    )
    loss = (
        criterion(pred.head_id, target.head_id, target.mask)
        + criterion(pred.rel_id, target.rel_id, target.mask)
    )

    pred.head_id = model.head.decode(pred.head_id, target.mask)
    pred.rel_id = model.rel.decode(pred.rel_id, target.mask)

    return batch.processed(loss, pred)
