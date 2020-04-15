
import pytest

from os.path import join, dirname, exists
from os import makedirs
from urllib.request import urlopen
from shutil import copyfileobj

from slovnet import NER, Morph


DATA_DIR = join(dirname(__file__), '../../data/models')


def download(url, path):
    dir = dirname(path)
    if not exists(dir):
        makedirs(dir)

    with urlopen(url) as source:
        with open(path, 'wb') as target:
            copyfileobj(source, target)


@pytest.fixture(scope='module')
def ner():
    url = 'https://storage.yandexcloud.net/natasha-slovnet/05_ner/pack/slovnet_ner_news_v1.tar'
    path = join(DATA_DIR, 'slovnet_ner_news_v1.tar')

    if not exists(path):
        download(url, path)

    return NER(path)


@pytest.fixture(scope='module')
def morph():
    url = 'https://storage.yandexcloud.net/natasha-slovnet/06_morph/pack/slovnet_morph_news_v1.tar'
    path = join(DATA_DIR, 'slovnet_morph_news_v1.tar')

    if not exists(path):
        download(url, path)

    return Morph(path)


def test_ner(ner):
    text = 'На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.'

    markup = next(ner([text]))

    pred = [
        markup.text[_.start:_.stop]
        for _ in markup.spans
    ]
    assert pred == ['Денис Пушилин', 'Донецкая республика', 'ДНР', 'Леонид Пасечник', 'Мир Луганщине', 'ЛНР']


def test_morph(morph):
    words = ['Об', 'этом', 'говорится', 'в', 'документе', ',', 'опубликованном', 'в', 'официальном', 'журнале', 'Евросоюза', '.']

    markup = next(morph([words]))

    pred = [_.pos for _ in markup.tokens]
    assert pred == ['ADP', 'PRON', 'VERB', 'ADP', 'NOUN', 'PUNCT', 'VERB', 'ADP', 'ADJ', 'NOUN', 'PROPN', 'PUNCT']
