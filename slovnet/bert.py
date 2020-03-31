
import re


def bert_chunks(text):
    # diff with bert tokenizer 28 / 10000 ~0.3%
    # школа №3 -> школа, №3
    # @diet_prada -> @, diet, _, prada
    return re.findall(r'\w+|[^\w\s]', text)


def wordpiece(text, vocab, prefix='##'):
    start = 0
    stop = size = len(text)
    subs = []
    while start < size:
        sub = text[start:stop]
        if start > 0:
            sub = prefix + sub
        if sub in vocab.item_ids:
            subs.append(sub)
            start = stop
            stop = size
        else:
            stop -= 1
            if stop < start:
                return
    return subs


def safe_wordpiece(text, vocab):
    subs = wordpiece(text, vocab)
    if not subs:
        return [text]
    return subs


def bert_subs(text, vocab):
    return [
        sub
        for chunk in bert_chunks(text)
        for sub in safe_wordpiece(chunk, vocab)
    ]
