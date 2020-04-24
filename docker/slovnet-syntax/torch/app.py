
from os import getenv

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(message)s'
)
log = logging.info

from aiohttp import web

import torch
torch.set_grad_enabled(False)

from navec import Navec

from slovnet.const import PAD, CUDA0
from slovnet.shape import SHAPES
from slovnet.vocab import Vocab
from slovnet.model.emb import (
    NavecEmbedding,
    Embedding
)
from slovnet.model.syntax import (
    SyntaxEmbedding,
    SyntaxEncoder,
    SyntaxHead,
    SyntaxRel,
    Syntax
)
from slovnet.encoders.syntax import SyntaxInferEncoder
from slovnet.infer.syntax import SyntaxInfer, SyntaxDecoder


NAVEC = getenv('NAVEC', 'navec.tar')
SHAPE = getenv('SHAPE', 'shape.pt')
ENCODER = getenv('ENCODER', 'encoder.pt')
HEAD = getenv('HEAD', 'head.pt')
REL = getenv('REL', 'rel.pt')
RELS_VOCAB = getenv('RELS_VOCAB', 'rels_vocab.txt')

SHAPE_DIM = 30
LAYER_DIMS = [256, 128, 64]
KERNEL_SIZE = 3

DEVICE = getenv('DEVICE', CUDA0)
BATCH_SIZE = int(getenv('BATCH_SIZE', 64))

HOST = getenv('HOST', '0.0.0.0')
PORT = int(getenv('PORT', 8080))
MB = 1024 * 1024
MAX_SIZE = int(getenv('MAX_SIZE', 100 * MB))


log('Load navec: %r' % NAVEC)
navec = Navec.load(NAVEC)

words_vocab = Vocab(navec.vocab.words)
shapes_vocab = Vocab([PAD] + SHAPES)
rels_vocab = Vocab.load(RELS_VOCAB)

word = NavecEmbedding(navec)
shape = Embedding(
    vocab_size=len(shapes_vocab),
    dim=SHAPE_DIM,
    pad_id=shapes_vocab.pad_id
)
emb = SyntaxEmbedding(word, shape)
encoder = SyntaxEncoder(
    input_dim=emb.dim,
    layer_dims=LAYER_DIMS,
    kernel_size=KERNEL_SIZE,
)
head = SyntaxHead(
    input_dim=encoder.dim,
    hidden_dim=encoder.dim // 2,
)
rel = SyntaxRel(
    input_dim=encoder.dim,
    hidden_dim=encoder.dim // 2,
    rel_dim=len(rels_vocab)
)
model = Syntax(emb, encoder, head, rel)
model.eval()

log('Load shape: %r' % SHAPE)
model.emb.shape.load(SHAPE)
log('Load encoder: %r' % ENCODER)
model.encoder.load(ENCODER)
log('Load head: %r' % HEAD)
model.head.load(HEAD)
log('Load rel: %r' % REL)
model.rel.load(REL)
log('Device: %r' % DEVICE)
model = model.to(DEVICE)

log('Batch size: %r' % BATCH_SIZE)
encoder = SyntaxInferEncoder(
    words_vocab, shapes_vocab,
    batch_size=BATCH_SIZE
)
decoder = SyntaxDecoder(rels_vocab)
infer = SyntaxInfer(model, encoder, decoder)


async def handle(request):
    chunk = await request.json()
    log('Post chunk size: %r' % len(chunk))
    markups = list(infer(chunk))

    tokens = sum(len(_.tokens) for _ in markups)
    log('Infer tokens: %r', tokens)

    data = [_.as_json for _ in markups]
    return web.json_response(data)


log('Max size: %r' % (MAX_SIZE // MB))
app = web.Application(client_max_size=MAX_SIZE)
app.add_routes([web.post('/', handle)])

web.run_app(app, host=HOST, port=PORT)
