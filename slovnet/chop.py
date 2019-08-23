

def chop(items, size):
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def chop_equal(items, size):
    previous = None
    chunks = chop(items, size)
    for chunk in chunks:
        if len(chunk) < size:
            if previous:  # last chunk
                chunk = (previous + chunk)[-size:]
            else:  # |items| < size
                repeat = size // len(chunk) + 1
                chunk = (chunk * repeat)[:size]
        yield chunk
        previous = chunk
