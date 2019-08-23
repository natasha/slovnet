
import re

from .record import Record
from .span import (
    envelop_spans,
    offset_spans
)


class Sent(Record):
    __attributes__ = ['start', 'stop', 'text']

    def __init__(self, start, stop, text):
        self.start = start
        self.stop = stop
        self.text = text


def split_lines(text):
    for match in re.finditer(r'([^\r\n]+)', text):
        start = match.start()
        stop = match.end()
        line = match.group(1)
        yield Sent(start, stop, line)


def sentenize(text):
    from razdel import sentenize as sentenize_

    for line in split_lines(text):
        for sent in sentenize_(line.text):
            if not sent.text:  # '\n\t\n' for example
                continue
            yield Sent(
                sent.start + line.start,
                sent.stop + line.start,
                sent.text
            )


def sent_spans(sent, spans):
    spans = envelop_spans(sent, spans)
    return offset_spans(spans, -sent.start)
