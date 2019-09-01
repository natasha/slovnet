
from navec import Navec
from slovnet import NERTagger

from .common import (
    NAVEC,
    SLOVNET
)


TEXT = '''
Россия планирует создать рейтинг среднего профобразования, в его разработке примет участие Китай, заявила заместитель председателя правительства РФ Татьяна Голикова.
'''
ETALON = ['Россия', 'Китай', 'РФ', 'Татьяна Голикова']


def test_ner_tagger():
    navec = Navec.load(NAVEC)
    tagger = NERTagger.load(SLOVNET, navec)
    markup = tagger(TEXT)
    guess = [TEXT[_.start:_.stop] for _ in markup.spans]
    assert guess == ETALON
