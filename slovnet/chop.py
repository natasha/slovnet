

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


