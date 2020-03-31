

def chop(items, size):
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def chop_drop(items, size):
    chunks = chop(items, size)
    for chunk in chunks:
        if len(chunk) < size:
            continue
        yield chunk


def chop_weighted(items, size, weight):
    buffer = []
    accum = 0
    for item in items:
        value = weight(item)
        if accum + value > size:
            yield buffer
            buffer = []
            accum = 0
        buffer.append(item)
        accum += value
    if buffer:
        yield buffer
