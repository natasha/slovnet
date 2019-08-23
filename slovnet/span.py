
from .record import Record


class Span(Record):
    __attributes__ = ['start', 'stop', 'type']

    def __init__(self, start, stop, type=None):
        self.start = start
        self.stop = stop
        self.type = type

    def offset(self, delta):
        return Span(
            self.start + delta,
            self.stop + delta,
            self.type
        )


def offset_spans(spans, delta):
    for span in spans:
        yield span.offset(delta)


def envelop_span(envelope, span):
    return envelope.start <= span.start and span.stop <= envelope.stop


def envelop_spans(envelope, spans):
    for span in spans:
        if envelop_span(envelope, span):
            yield span


def select_type_spans(spans, types):
    for span in spans:
        if span.type in types:
            yield span
