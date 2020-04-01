

def format_conll_tag(pos, feats):
    if not feats:
        return pos

    feats = '|'.join(
        '%s=%s' % (_, feats[_])
        for _ in sorted(feats)
    )
    return '%s|%s' % (pos, feats)


def parse_conll_tag(tag):
    if '|' not in tag:
        return tag, {}

    pos, feats = tag.split('|', 1)
    feats = dict(
        _.split('=', 1)
        for _ in feats.split('|')
    )
    return pos, feats
