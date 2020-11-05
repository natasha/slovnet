
from .span import Span
from .const import B, I, O


def parse_bio(tag):
    """
    Parse a bio tag.

    Args:
        tag: (str): write your description
    """
    if '-' in tag:
        part, type = tag.split('-', 1)
    else:
        part = tag
        type = None
    return part, type


def format_bio(part, type):
    """
    Format a bio. bio.

    Args:
        part: (int): write your description
        type: (str): write your description
    """
    if not type:
        return part
    return '%s-%s' % (part, type)


##########
#
#    IO
#
#########

# assert tokens and spans are sorted
# assert spans do not overlap
# assert span bounds align with token bounds


def append_ellipsis(items, ellipsis=None):
    """
    Yield all ellipsis.

    Args:
        items: (todo): write your description
        ellipsis: (str): write your description
    """
    for item in items:
        yield item
    yield ellipsis


def spans_io(tokens, spans):
    """
    Yields spans from a list of spans.

    Args:
        tokens: (str): write your description
        spans: (todo): write your description
    """
    spans = append_ellipsis(spans)
    span = next(spans)
    for token in tokens:
        part = O
        type = None
        if span:
            if token.start >= span.start:
                part = I
                type = span.type
            if token.stop >= span.stop:
                span = next(spans)
        yield format_bio(part, type)


def io_spans(tokens, tags):
    """
    Yields a sequence of ) tuples.

    Args:
        tokens: (str): write your description
        tags: (list): write your description
    """
    previous = None
    start = None
    stop = None
    for token, tag in zip(tokens, tags):
        part, type = parse_bio(tag)
        # wikiner splits on I-PER B-PER for example
        if previous != type or part == B:
            if not previous and type:
                # O I
                start = token.start
            elif previous and type:
                # I-A I-B
                yield Span(start, stop, previous)
                start = token.start
            elif previous and not type:
                # I O
                yield Span(start, stop, previous)
                previous = None
        previous = type
        stop = token.stop
    if previous:
        yield Span(start, stop, previous)


#######
#
#   BIO
#
#########


def spans_bio(tokens, spans):
    """
    Yields spans from a list of spans.

    Args:
        tokens: (str): write your description
        spans: (todo): write your description
    """
    spans = append_ellipsis(spans)
    span = next(spans)
    for token in tokens:
        part = O
        type = None
        if span:
            if token.start >= span.start:
                type = span.type
                if token.start == span.start:
                    part = B
                else:
                    part = I
            if token.stop >= span.stop:
                span = next(spans)
        yield format_bio(part, type)


def bio_spans(tokens, tags):
    """
    Yields spans.

    Args:
        tokens: (str): write your description
        tags: (array): write your description
    """
    previous = None
    start = None
    stop = None
    for token, tag in zip(tokens, tags):
        part, type = parse_bio(tag)
        if part == O:
            if previous:
                yield Span(start, stop, previous)
                previous = None
        elif part == B:
            if previous:
                yield Span(start, stop, previous)
            previous = type
            start = token.start
            stop = token.stop
        elif part == I:
            stop = token.stop
    if previous:
        yield Span(start, stop, previous)


#########
#
#   CONVERT
#
#########


def bio_io(tags):
    """
    Iterate over bio. io tags ).

    Args:
        tags: (list): write your description
    """
    for tag in tags:
        part, type = parse_bio(tag)
        if part == B:
            part = I
        yield format_bio(part, type)


########
#
#   SELECT
#
######


def select_type_tags(tags, selected):
    """
    Yield tags of tags.

    Args:
        tags: (str): write your description
        selected: (bool): write your description
    """
    for tag in tags:
        part, type = parse_bio(tag)
        if type != selected:
            part = O
            type = None
        yield format_bio(part, type)
