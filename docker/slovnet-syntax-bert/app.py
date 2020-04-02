
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

from slovnet.const import CUDA0
from slovnet.vocab import (
    BERTVocab,
    Vocab
)
from slovnet.model.bert import (
    RuBERTConfig,
    BERTEmbedding,
    BERTEncoder,
    BERTSyntaxHead,
    BERTSyntaxRel,
    BERTSyntax
)
from slovnet.encoders.bert import BERTInferEncoder
from slovnet.infer.bert import BERTSyntaxInfer, BERTSyntaxDecoder


WORDS_VOCAB = getenv('WORDS_VOCAB', 'vocab.txt')
RELS_VOCAB = getenv('RELS_VOCAB', 'rels_vocab.txt')
EMB = getenv('EMB', 'emb.pt')
ENCODER = getenv('ENCODER', 'encoder.pt')
HEAD = getenv('HEAD', 'head.pt')
REL = getenv('REL', 'rel.pt')

DEVICE = getenv('DEVICE', CUDA0)
SEQ_LEN = int(getenv('SEQ_LEN', 512))
BATCH_SIZE = int(getenv('BATCH_SIZE', 32))

HOST = getenv('HOST', '0.0.0.0')
PORT = int(getenv('PORT', 8080))
MB = 1024 * 1024
MAX_SIZE = int(getenv('MAX_SIZE', 100 * MB))


log('Load vocabs: %r, %r' % (WORDS_VOCAB, RELS_VOCAB))
words_vocab = BERTVocab.load(WORDS_VOCAB)
rels_vocab = Vocab.load(RELS_VOCAB)

config = RuBERTConfig()
emb = BERTEmbedding.from_config(config)
encoder = BERTEncoder.from_config(config)
head = BERTSyntaxHead(
    input_dim=config.emb_dim,
    hidden_dim=config.emb_dim // 2,
)
rel = BERTSyntaxRel(
    input_dim=config.emb_dim,
    hidden_dim=config.emb_dim // 2,
    rel_dim=len(rels_vocab)
)
model = BERTSyntax(emb, encoder, head, rel)
model.eval()

log('Load emb: %r' % EMB)
model.emb.load(EMB)
log('Load encoder: %r' % ENCODER)
model.encoder.load(ENCODER)
log('Load head, rel: %r, %r' % (HEAD, REL))
model.head.load(HEAD)
model.rel.load(REL)
log('Device: %r' % DEVICE)
model = model.to(DEVICE)

log('Seq len: %r' % SEQ_LEN)
log('Batch size: %r' % BATCH_SIZE)
encoder = BERTInferEncoder(
    words_vocab,
    seq_len=SEQ_LEN, batch_size=BATCH_SIZE
)
decoder = BERTSyntaxDecoder(rels_vocab)
infer = BERTSyntaxInfer(model, encoder, decoder)


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
