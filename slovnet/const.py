
from os.path import exists, expanduser

from .io import load_json


TRAIN = 'train'
TEST = 'test'

TXT = '.txt'
PT = '.pt'

CUDA0 = 'cuda:0'
CUDA1 = 'cuda:1'
CUDA2 = 'cuda:2'
CUDA3 = 'cuda:3'
CPU = 'cpu'


######
#  CONFIG
#####


config = {}
path = expanduser('~/.slovnet.json')
if exists(path):
    config = load_json(path)


########
#   S3
######


S3_KEY_ID = config.get('s3_key_id')
S3_KEY = config.get('s3_key')
S3_BUCKET = 'natasha-slovnet'
S3_REGION = 'us-east-1'
S3_ENDPOINT = 'https://storage.yandexcloud.net'
