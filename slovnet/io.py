
import json
import gzip


def load_lines(path, encoding='utf8'):
    """
    Load lines from file.

    Args:
        path: (str): write your description
        encoding: (str): write your description
    """
    with open(path, encoding=encoding) as file:
        for line in file:
            yield line.rstrip('\n')


def dump_lines(lines, path, encoding='utf8'):
    """
    Write lines to file.

    Args:
        lines: (list): write your description
        path: (str): write your description
        encoding: (str): write your description
    """
    with open(path, 'w', encoding=encoding) as file:
        for line in lines:
            file.write(line + '\n')


def load_gz_lines(path, encoding='utf8'):
    """
    Load lines from a file.

    Args:
        path: (str): write your description
        encoding: (str): write your description
    """
    with gzip.open(path) as file:
        for line in file:
            yield line.decode(encoding).rstrip()


def dump_gz_lines(lines, path):
    """
    Write a gzipped to a gz file.

    Args:
        lines: (array): write your description
        path: (str): write your description
    """
    with gzip.open(path, 'wt') as file:
        for line in lines:
            file.write(line + '\n')


def load_json(path, encoding='utf8'):
    """
    Load json file ascii file.

    Args:
        path: (str): write your description
        encoding: (str): write your description
    """
    with open(path, encoding=encoding) as file:
        return json.load(file)


def format_jl(items):
    """
    Format an iterable items as json.

    Args:
        items: (todo): write your description
    """
    for item in items:
        yield json.dumps(item, ensure_ascii=False)


def parse_jl(lines):
    """
    Parse a generator. json.

    Args:
        lines: (str): write your description
    """
    for line in lines:
        yield json.loads(line)
