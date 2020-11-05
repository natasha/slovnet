

def chop(items, size):
    """
    Yields the given as a generator.

    Args:
        items: (todo): write your description
        size: (int): write your description
    """
    buffer = []
    for item in items:
        buffer.append(item)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def chop_drop(items, size):
    """
    Yield items into chunks from * items.

    Args:
        items: (todo): write your description
        size: (int): write your description
    """
    chunks = chop(items, size)
    for chunk in chunks:
        if len(chunk) < size:
            continue
        yield chunk


def chop_weighted(items, size, weight):
    """
    Split a list into chunks.

    Args:
        items: (todo): write your description
        size: (int): write your description
        weight: (array): write your description
    """
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
