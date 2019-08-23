
import torch
from torch import optim

from navec import Navec

from .record import Record
from .log import (
    log,
    log_progress,
    temp_log_progress
)
from .bio import PER, LOC, ORG
from .shape import SHAPES
from .vocab import (
    WordsVocab,
    ShapesVocab,
    TagsVocab
)
from .encoders import (
    WordEncoder,
    ShapeEncoder,
    StackEncoder,
    MarkupEncoder,
    TagEncoder,
    BatchEncoder
)
from .tokenizer import Tokenizer
from .dataset import NerusDataset
from .device import get_device
from .model import (
    ShapeEmbedding,
    WordModel,
    CNNContextModel,
    CRFTagModel,
    NERModel
)
from .board import Board
from .loop import (
    train_model,
    infer_model
)
from .eval import (
    eval_batches,
    avg_batch_scores
)


class Params(Record):
    __attributes__ = [
        'navec_path',
        'nerus_path',
        'test_size',
        'train_size',
        'seq_len',
        'batch_size',
        'shuffle_buffer_size',
        'seed',
        'shape_dim',
        'layer_dims',
        'kernel_size',
        'board_dir',
        'board_root',
        'lr',
        'epochs',
    ]

    def __init__(self,
                 navec_path,
                 nerus_path,
                 test_size=1000,
                 train_size=10000,
                 seq_len=128,
                 batch_size=64,
                 shuffle_buffer_size=5120,
                 seed=1,
                 shape_dim=30,
                 layer_dims=[256, 128, 64],
                 kernel_size=3,
                 board_dir='exp',
                 board_root='root',
                 lr=0.001,
                 epochs=5):
        self.navec_path = navec_path
        self.nerus_path = nerus_path
        self.test_size = test_size
        self.train_size = train_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.shape_dim = shape_dim
        self.layer_dims = layer_dims
        self.kernel_size = kernel_size
        self.board_dir = board_dir
        self.board_root = board_root
        self.lr = lr
        self.epochs = epochs


def train(params):
    _ = params

    ######
    #  VOCABS
    ########

    log('Load navec: %s', _.navec_path)
    navec = Navec.load(_.navec_path)
    words_vocab = WordsVocab(navec.vocab.words)
    shapes_vocab = ShapesVocab(SHAPES)
    tags_vocab = TagsVocab([PER, LOC, ORG])

    ##########
    #   ENCODERS
    #######

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
        seq_len=_.seq_len,
        batch_size=_.batch_size,
        shuffle_buffer_size=_.shuffle_buffer_size,
    )

    #######
    #   DATA
    #######

    dataset = NerusDataset(_.nerus_path)
    test_dataset = dataset.slice(0, _.test_size)
    train_dataset = dataset.slice(
        _.test_size,
        _.test_size + _.train_size
    )

    device = get_device()

    def load(dataset, size, prefix):
        dataset = temp_log_progress(
            dataset,
            prefix=prefix,
            total=size
        )
        for batch in batch_encoder.map(dataset):
            yield batch.to(device)

    test_batches = list(load(test_dataset, _.test_size, 'Load test'))
    train_batches = list(load(train_dataset, _.train_size, 'Load train'))

    ######
    #  MODEL
    #########

    torch.manual_seed(_.seed)
    torch.backends.cudnn.deterministic = True

    word_emb = navec.as_torch
    shape_emb = ShapeEmbedding(
        vocab_size=len(shapes_vocab),
        dim=_.shape_dim,
        pad_id=shapes_vocab.pad_id
    )
    word_model = WordModel(
        word_emb,
        shape_emb
    )
    context_model = CNNContextModel(
        input_dim=word_model.dim,
        layer_dims=_.layer_dims,
        kernel_size=_.kernel_size,
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

    ########
    #   BOARD
    ########

    board = Board(_.board_dir, _.board_root)
    train_board = board.prefixed('01_train')
    test_board = board.prefixed('02_test')

    ########
    #  LOOP
    #########

    log('Train')
    optimizer = optim.Adam(
        ner_model.parameters(),
        lr=_.lr
    )
    epochs = log_progress(range(_.epochs), 'Epoch')
    for epoch in epochs:
        proced = train_model(
            ner_model, optimizer,
            temp_log_progress(train_batches, 'Train')
        )
        scores = eval_batches(tags_vocab, proced)
        score = avg_batch_scores(scores)
        train_board.add_batch_score(score)

        proced = infer_model(
            ner_model,
            temp_log_progress(test_batches, 'Test')
        )
        scores = eval_batches(tags_vocab, proced)
        score = avg_batch_scores(scores)
        test_board.add_batch_score(score)
