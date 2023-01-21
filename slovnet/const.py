

from os.path import exists, abspath

from .io import load_json

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

#########
#  DEVICE
#########

CUDA0 = 'cuda:0'
CUDA1 = 'cuda:1'
CUDA2 = 'cuda:2'
CUDA3 = 'cuda:3'
CPU = 'cpu'

#########
#   VOCAB
#########

UNK = '<unk>'
PAD = '<pad>'
CLS = '<cls>'
SEP = '<sep>'
MASK = '<mask>'

WORD = 'word'
SHAPE = 'shape'
TAG = 'tag'
REL = 'rel'

#########
#  BIO
#########

B = 'B'
I = 'I'  # noqa E741
O = 'O'  # noqa

PER = 'PER'
LOC = 'LOC'
ORG = 'ORG'

#########
#  CONFIG
#########

config = {}
path = abspath('../slovnet.json')

if exists(path):
    config = load_json(path)

#########
#  S3
#########

S3_KEY_ID = config.get('s3_key_id')
S3_KEY = config.get('s3_key')
S3_BUCKET = 'natasha-slovnet'
S3_REGION = 'us-east-1'
S3_ENDPOINT = 'https://storage.yandexcloud.net'
