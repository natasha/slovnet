
from os import getenv
from os.path import expanduser

from .record import Record
from .io import load_text, parse_json


PATH = expanduser('~/.slovnet.json')


def try_eval(value):
    try:
        return eval(value)
    except:
        return


class Env(Record):
    __attributes__ = ['vars']

    def __init__(self, vars):
        self.vars = vars

    def __getattr__(self, key):
        return self.vars[key]

    @classmethod
    def from_file(cls, path=PATH):
        vars = parse_json(load_text(path))
        return cls(vars)

    @classmethod
    def from_env(cls, names=()):
        vars = {
            _: try_eval(getenv(_))
            for _ in names
        }
        return cls(vars)

    def merge(self, other):
        self.vars.update(other.vars)
