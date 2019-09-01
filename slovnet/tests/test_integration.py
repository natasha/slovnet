
import torch
from torch import optim

from navec import Navec

from slovnet.device import get_device
from slovnet.tokenizer import Tokenizer
from slovnet.dataset import NerusDataset
from slovnet.bio import PER, LOC, ORG
from slovnet.shape import SHAPES
from slovnet.vocab import (
    WordsVocab,
    ShapesVocab,
    TagsVocab
)
from slovnet.encoder import (
    WordEncoder,
    ShapeEncoder,
    StackEncoder,
    MarkupEncoder,
    TagEncoder,
    BatchEncoder
)
from slovnet.model.word import (
    ShapeEmbedding,
    WordModel
)
from slovnet.model.navec import NavecEmbedding
from slovnet.model.context import CNNContextModel
from slovnet.model.tag import CRFTagModel
from slovnet.model.ner import NERModel
from slovnet.board import Board
from slovnet.loop import (
    train_model,
    infer_model
)
from slovnet.eval import (
    eval_batches,
    avg_batch_scores
)
from slovnet.tagger import NERTagger

from slovnet.infer.impl import NavecEmbedding as InferNavecEmbedding
from slovnet.infer.pack import Pack
from slovnet.infer.tagger import NERTagger as InferNERTagger

from .common import (
    NERUS,
    NAVEC
)


TEXT = '''
Россия планирует создать рейтинг среднего профобразования, в его разработке примет участие Китай, заявила заместитель председателя правительства РФ Татьяна Голикова.
'''


def test_integration(tmpdir):
    torch.manual_seed(1)
    device = get_device()

    navec = Navec.load(NAVEC)

    words_vocab = WordsVocab(navec.vocab.words)
    shapes_vocab = ShapesVocab(SHAPES)
    tags_vocab = TagsVocab([PER, LOC, ORG])

    word_emb = NavecEmbedding.from_navec(navec)
    shape_emb = ShapeEmbedding(
        vocab_size=len(shapes_vocab),
        dim=10,
        pad_id=shapes_vocab.pad_id
    )
    word_model = WordModel(
        word_emb,
        shape_emb
    )
    context_model = CNNContextModel(
        input_dim=word_model.dim,
        layer_dims=[64, 32],
        kernel_size=3,
    )
    tag_model = CRFTagModel(
        input_dim=context_model.dim,
        tags_num=len(tags_vocab)
    )
    ner_model = NERModel(
        word_model,
        context_model,
        tag_model
    ).to(device)

    dataset = NerusDataset(NERUS)
    test_dataset = dataset.slice(0, 10)
    train_dataset = dataset.slice(10, 30)

    token_encoder = StackEncoder([
        WordEncoder(words_vocab),
        ShapeEncoder(shapes_vocab)
    ])
    markup_encoder = MarkupEncoder(
        token_encoder,
        TagEncoder(tags_vocab)
    )

    tokenizer = Tokenizer()
    batch_encoder = BatchEncoder(
        tokenizer,
        markup_encoder,
        seq_len=100,
        batch_size=32,
        shuffle_buffer_size=256,
    )

    test_batches = [_.to(device) for _ in batch_encoder.map(test_dataset)]
    train_batches = [_.to(device) for _ in batch_encoder.map(train_dataset)]

    path = str(tmpdir.mkdir('root'))
    board = Board('01', path)
    train_board = board.prefixed('01_train')
    test_board = board.prefixed('02_test')

    optimizer = optim.Adam(
        ner_model.parameters(),
        lr=0.001
    )
    proced = train_model(
        ner_model, optimizer,
        train_batches
    )
    scores = eval_batches(tags_vocab, proced)
    score = avg_batch_scores(scores)
    train_board.add_batch_score(score)

    proced = infer_model(
        ner_model,
        test_batches
    )
    scores = eval_batches(tags_vocab, proced)
    score = avg_batch_scores(scores)
    test_board.add_batch_score(score)

    ner_model.eval()
    tagger = NERTagger(
        tokenizer,
        token_encoder,
        tags_vocab,
        ner_model,
        device
    )
    markup1 = tagger(TEXT)

    pack = ner_model.as_infer.pack('slovnet_ner_v1')
    path = str(tmpdir.join('slovnet_ner_v1.tar'))
    pack.dump(path)

    pack = Pack.load(path)
    pack.context.navec = InferNavecEmbedding.from_navec(navec)
    ner_model = pack.scheme.to_impl(pack.context)

    tagger = InferNERTagger(
        tokenizer,
        token_encoder,
        tags_vocab,
        ner_model
    )
    markup2 = tagger(TEXT)

    assert markup1 == markup2
