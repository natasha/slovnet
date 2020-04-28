
from os import getenv

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)-15s %(message)s'
)
log = logging.info

from aiohttp import web

from navec import Navec
from slovnet import Syntax


NAVEC = getenv('NAVEC', 'navec.tar')
PACK = getenv('PACK', 'pack.tar')
BATCH_SIZE = int(getenv('BATCH_SIZE', 8))

HOST = getenv('HOST', '0.0.0.0')
PORT = int(getenv('PORT', 8080))
MB = 1024 * 1024
MAX_SIZE = int(getenv('MAX_SIZE', 100 * MB))

log('Load navec: %r' % NAVEC)
navec = Navec.load(NAVEC)

log('Load pack: %r' % PACK)
log('Batch size: %r' % BATCH_SIZE)
syntax = Syntax.load(PACK)
syntax.navec(navec)


async def handle(request):
    chunk = await request.json()
    log('Post chunk size: %r' % len(chunk))
    markups = list(syntax.map(chunk))

    tokens = sum(len(_.tokens) for _ in markups)
    log('Infer tokens: %r', tokens)

    data = [_.as_json for _ in markups]
    return web.json_response(data)


log('Max size: %r' % (MAX_SIZE // MB))
app = web.Application(client_max_size=MAX_SIZE)
app.add_routes([web.post('/', handle)])

web.run_app(app, host=HOST, port=PORT)
