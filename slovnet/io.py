
import json


def load_lines(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        for line in file:
            yield line.rstrip('\n')


def dump_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as file:
        for line in lines:
            file.write(line + '\n')


parse_json = json.loads


def load_text(path, encoding='utf8'):
    with open(path, encoding=encoding) as file:
        return file.read()
