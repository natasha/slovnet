
import json


def load_lines(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        for line in file:
            yield line.rstrip('\n')


def dump_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as file:
        for line in lines:
            file.write(line + '\n')


def load_json(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return json.load(file)
