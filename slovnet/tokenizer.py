# coding: utf8

import re

from .record import Record


class TokenRule(Record):
    __attributes__ = ['type', 'pattern']

    def __init__(self, type, pattern):
        self.type = type
        self.pattern = pattern


RU = 'RU'
LATIN = 'LATIN'
INT = 'INT'
PUNCT = 'PUNCT'
OTHER = 'OTHER'
PUNCTS = (
    '!#$%&()[]\\/*+,.:;<=>?@^_{|}~'  # string.punctuation
    '-‐−‒⁃–—―'  # https://habr.com/ru/post/20588/
    '`"\'«»„“ʼʻ”'
    '№…'
)


RULES = [
    TokenRule(RU, r'[а-яё]+[а-яё‐-]+[а-яё]+|[а-яё]+'),  # сине-бело-голубых
    TokenRule(LATIN, r'[a-z]+[a-z‐-][a-z]+|[a-z]+'),
    TokenRule(INT, r'\d+'),
    TokenRule(
        PUNCT,
        r'[' + re.escape(PUNCTS) + ']'
    ),
    TokenRule(OTHER, r'\S'),
]


class Token(Record):
    __attributes__ = ['text', 'start', 'stop', 'type']

    def __init__(self, text, start, stop, type):
        self.text = text
        self.start = start
        self.stop = stop
        self.type = type


class Tokenizer(Record):
    __attributes__ = ['rules']

    def __init__(self, rules=RULES):
        self.rules = rules
        self.regexp, self.mapping, self.types = self.compile(self.rules)

    def compile(self, rules):
        types = set()
        mapping = {}
        patterns = []
        for rule in rules:
            type, pattern = rule
            name = 'rule_{id}'.format(id=id(rule))
            pattern = r'(?P<{name}>{pattern})'.format(
                name=name,
                pattern=pattern
            )
            mapping[name] = type
            types.add(type)
            patterns.append(pattern)
        pattern = '|'.join(patterns)
        regexp = re.compile(pattern, re.UNICODE | re.IGNORECASE)
        return regexp, mapping, types

    def __call__(self, text):
        for match in re.finditer(self.regexp, text):
            name = match.lastgroup
            text = match.group(0)
            start, stop = match.span()
            type = self.mapping[name]
            token = Token(text, start, stop, type)
            yield token
