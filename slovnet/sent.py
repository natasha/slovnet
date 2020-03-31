
import re

from razdel import sentenize as sentenize_

from .record import Record


class Sent(Record):
    __attributes__ = ['start', 'stop', 'text']


def split_lines(text):
    for match in re.finditer(r'([^\r\n]+)', text):
        start = match.start()
        stop = match.end()
        line = match.group(1)
        yield Sent(start, stop, line)


def sentenize(text):
    for line in split_lines(text):
        for sent in sentenize_(line.text):
            if not sent.text:  # '\n\t\n' for example
                continue
            yield Sent(
                sent.start + line.start,
                sent.stop + line.start,
                sent.text
            )
