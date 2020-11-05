
from .record import Record


class Span(Record):
    __attributes__ = ['start', 'stop', 'type']

    def __init__(self, start, stop, type=None):
        """
        Initialize a new type

        Args:
            self: (todo): write your description
            start: (int): write your description
            stop: (int): write your description
            type: (str): write your description
        """
        self.start = start
        self.stop = stop
        self.type = type

    def offset(self, delta):
        """
        Return the offset.

        Args:
            self: (todo): write your description
            delta: (todo): write your description
        """
        return Span(
            self.start + delta,
            self.stop + delta,
            self.type
        )


def offset_spans(spans, delta):
    """
    Yields spans that match the given span.

    Args:
        spans: (str): write your description
        delta: (float): write your description
    """
    for span in spans:
        yield span.offset(delta)


def envelop_span(envelope, span):
    """
    Envelop the span of a span.

    Args:
        envelope: (dict): write your description
        span: (todo): write your description
    """
    return envelope.start <= span.start and span.stop <= envelope.stop


def envelop_spans(envelope, spans):
    """
    Envelop all spans. spans.

    Args:
        envelope: (dict): write your description
        spans: (todo): write your description
    """
    for span in spans:
        if envelop_span(envelope, span):
            yield span


def select_type_spans(spans, types):
    """
    Selects all spans from the given spans.

    Args:
        spans: (todo): write your description
        types: (str): write your description
    """
    for span in spans:
        if span.type in types:
            yield span
