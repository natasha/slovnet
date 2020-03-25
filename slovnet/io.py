
import json
import gzip


def load_lines(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        for line in file:
            yield line.rstrip('\n')


def dump_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as file:
        for line in lines:
            file.write(line + '\n')


def load_gz_lines(path, encoding='utf8'):
    with gzip.open(path) as file:
        for line in file:
            yield line.decode(encoding).rstrip()


def dump_gz_lines(lines, path):
    with gzip.open(path, 'wt') as file:
        for line in lines:
            file.write(line + '\n')


def load_json(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return json.load(file)


def format_jl(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False)


def parse_jl(lines):
    for line in lines:
        yield json.loads(line)
