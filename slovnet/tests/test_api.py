
import pytest

from os.path import join, dirname, basename, exists
from os import makedirs
from urllib.request import urlopen
from shutil import copyfileobj

from navec import Navec
from slovnet import NER, Morph, Syntax


DATA_DIR = join(dirname(__file__), '../../data/test')


def download(url, dir=DATA_DIR):
    path = join(dir, basename(url))
    if exists(path):
        return path

    if not exists(dir):
        makedirs(dir)

    with urlopen(url) as source:
        with open(path, 'wb') as target:
            copyfileobj(source, target)

    return path


@pytest.fixture(scope='module')
def navec():
    path = download('https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar')
    return Navec.load(path)


@pytest.fixture(scope='module')
def ner(navec):
    path = download('https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_ner_news_v1.tar')
    return NER.load(path).navec(navec)


@pytest.fixture(scope='module')
def morph(navec):
    path = download('https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_morph_news_v1.tar')
    return Morph.load(path).navec(navec)


@pytest.fixture(scope='module')
def syntax(navec):
    path = download('https://storage.yandexcloud.net/natasha-slovnet/packs/slovnet_syntax_news_v1.tar')
    return Syntax.load(path).navec(navec)


def test_ner(ner):
    text = 'На них удержали лидерство действующие руководители и партии — Денис Пушилин и «Донецкая республика» в ДНР и Леонид Пасечник с движением «Мир Луганщине» в ЛНР.'

    markup = ner(text)

    pred = []
    for span in markup.spans:
        chunk = markup.text[span.start:span.stop]
        pred.append([span.type, chunk])

    assert pred == [
        ['PER', 'Денис Пушилин'],
        ['ORG', 'Донецкая республика'],
        ['LOC', 'ДНР'],
        ['PER', 'Леонид Пасечник'],
        ['ORG', 'Мир Луганщине'],
        ['LOC', 'ЛНР']
    ]


def test_morph(morph):
    words = ['Об', 'этом', 'говорится', 'в', 'документе', ',', 'опубликованном', 'в', 'официальном', 'журнале', 'Евросоюза', '.']

    markup = morph(words)

    pred = [
        [_.text, _.tag]
        for _ in markup.tokens
    ]
    assert pred == [
        ['Об', 'ADP'],
        ['этом', 'PRON|Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing'],
        ['говорится', 'VERB|Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Pass'],
        ['в', 'ADP'],
        ['документе', 'NOUN|Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing'],
        [',', 'PUNCT'],
        ['опубликованном', 'VERB|Aspect=Perf|Case=Loc|Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part|Voice=Pass'],
        ['в', 'ADP'],
        ['официальном', 'ADJ|Case=Loc|Degree=Pos|Gender=Masc|Number=Sing'],
        ['журнале', 'NOUN|Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing'],
        ['Евросоюза', 'PROPN|Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing'],
        ['.', 'PUNCT']
    ]


def test_syntax(syntax):
    words = ['Опубликованы', 'новые', 'данные', 'по', 'заражению', 'коронавирусом', 'в', 'Москве']

    markup = syntax(words)

    ids = {_.id: _ for _ in markup.tokens}
    pred = []
    for token in markup.tokens:
        head = ids.get(token.head_id)
        if head:
            pred.append([token.text, head.rel, head.text])
        else:
            pred.append(token.text)

    assert pred == [
        'Опубликованы',
        ['новые', 'nsubj:pass', 'данные'],
        ['данные', 'root', 'Опубликованы'],
        ['по', 'nmod', 'заражению'],
        ['заражению', 'nsubj:pass', 'данные'],
        ['коронавирусом', 'nmod', 'заражению'],
        ['в', 'obl', 'Москве'],
        ['Москве', 'nmod', 'коронавирусом']
    ]
