
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

from slovnet.const import (
    PER, LOC, ORG,
    PAD,
    CUDA0
)
from slovnet.shape import SHAPES
from slovnet.vocab import (
    Vocab,
    BIOTagsVocab
)
from slovnet.model.emb import (
    NavecEmbedding,
    Embedding
)
from slovnet.model.tag import (
    TagEmbedding,
    TagEncoder,
    NERHead,
    NER as NERModel
)
from slovnet.encoders.tag import TagInferEncoder
from slovnet.infer.tag import NERInfer, TagDecoder


NAVEC = getenv('NAVEC', 'navec.tar')
SHAPE = getenv('SHAPE', 'shape.pt')
ENCODER = getenv('ENCODER', 'encoder.pt')
NER = getenv('NER', 'ner.pt')

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
tags_vocab = BIOTagsVocab([PER, LOC, ORG])

word = NavecEmbedding(navec)
shape = Embedding(
    vocab_size=len(shapes_vocab),
    dim=SHAPE_DIM,
    pad_id=shapes_vocab.pad_id
)
emb = TagEmbedding(word, shape)
encoder = TagEncoder(
    input_dim=emb.dim,
    layer_dims=LAYER_DIMS,
    kernel_size=KERNEL_SIZE,
)
ner = NERHead(encoder.dim, len(tags_vocab))
model = NERModel(emb, encoder, ner)
model.eval()

log('Load shape: %r' % SHAPE)
model.emb.shape.load(SHAPE)
log('Load encoder: %r' % ENCODER)
model.encoder.load(ENCODER)
log('Load ner: %r' % NER)
model.head.load(NER)
log('Device: %r' % DEVICE)
model = model.to(DEVICE)

log('Batch size: %r' % BATCH_SIZE)
encoder = TagInferEncoder(
    words_vocab, shapes_vocab,
    batch_size=BATCH_SIZE
)
decoder = TagDecoder(tags_vocab)
infer = NERInfer(model, encoder, decoder)


async def handle(request):
      """
      Handles the request

      Args:
          request: (todo): write your description
      """
    chunk = await request.json()
    log('Post chunk size: %r' % len(chunk))
    markups = list(infer(chunk))

    spans = sum(len(_.spans) for _ in markups)
    log('Infer spans: %r', spans)

    data = [_.as_json for _ in markups]
    return web.json_response(data)


log('Max size: %r' % (MAX_SIZE // MB))
app = web.Application(client_max_size=MAX_SIZE)
app.add_routes([web.post('/', handle)])

web.run_app(app, host=HOST, port=PORT)
