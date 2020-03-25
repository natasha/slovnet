
from os.path import exists, join, expanduser

from tqdm.notebook import tqdm as log_progress

from naeval.ner.datasets import (
    load_bsnlp,
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
from slovnet.board import Board
from slovnet.const import CUDA0

from slovnet.model.state import (
    load_model,
    dump_model
)
from slovnet.model.bert import (
    BERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTNERHead,
    BERTNER
)
from slovnet.vocab import BERTVocab, BIOTagsVocab
from slovnet.encoders.bert import BERTNEREncoder
from slovnet.score import (
    NERScoreMeter,
    score_ner_batches as score_batches
)
from slovnet.loop import (
    every,
    process_bert_ner_batch as process_batch
)


DATA_DIR = 'data'
MODEL_DIR = 'model'
RUBERT_DIR = 'rubert'

CORUS_DIR = expanduser('~/proj/corus-data/')
CORUS_NE5 = join(CORUS_DIR, 'Collection5')
CORUS_BSNLP = join(CORUS_DIR, 'bsnlp')
CORUS_FACTRU = join(CORUS_DIR, 'factRuEval-2016-master')

NE5 = join(DATA_DIR, 'ne5.jl.gz')
BSNLP = join(DATA_DIR, 'bsnlp.jl.gz')
FACTRU = join(DATA_DIR, 'factru.jl.gz')

S3_DIR = '02_bert_ner'
S3_NE5 = join(S3_DIR, 'ne5.jl.gz')
S3_BSNLP = join(S3_DIR, 'bsnlp.jl.gz')
S3_FACTRU = join(S3_DIR, 'factru.jl.gz')

VOCAB = 'vocab.txt'
EMB = 'emb.pt'
ENCODER = 'encoder.pt'
NER = 'ner.pt'

RUBERT_VOCAB = join(RUBERT_DIR, VOCAB)
RUBERT_EMB = join(RUBERT_DIR, EMB)
RUBERT_ENCODER = join(RUBERT_DIR, ENCODER)
RUBERT_NER = join(RUBERT_DIR, NER)

S3_RUBERT_VOCAB = join(S3_DIR, RUBERT_VOCAB)
S3_RUBERT_EMB = join(S3_DIR, RUBERT_EMB)
S3_RUBERT_ENCODER = join(S3_DIR, RUBERT_ENCODER)
S3_RUBERT_NER = join(S3_DIR, RUBERT_NER)

MODEL_EMB = join(MODEL_DIR, EMB)
MODEL_ENCODER = join(MODEL_DIR, ENCODER)
MODEL_NER = join(MODEL_DIR, NER)

S3_MODEL_EMB = join(S3_DIR, MODEL_EMB)
S3_MODEL_ENCODER = join(S3_DIR, MODEL_ENCODER)
S3_MODEL_NER = join(S3_DIR, MODEL_NER)

BOARD_NAME = '02_bert_ner'
RUNS_DIR = 'runs'

TRAIN_BOARD = '01_train'
TEST_BOARD = '02_test'
