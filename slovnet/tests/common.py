
from os.path import (
    join as join_path,
    dirname
)


def relative_path(*parts):
    return join_path(dirname(__file__), *parts)


NAVEC = relative_path('data', 'navec.tar')
NERUS = relative_path('data', 'nerus.jsonl.gz')
SLOVNET = relative_path('data', 'slovnet.tar')
