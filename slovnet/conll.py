

def format_conll_tag(pos, feats):
    """
    Format pos tag.

    Args:
        pos: (int): write your description
        feats: (todo): write your description
    """
    if not feats:
        return pos

    feats = '|'.join(
        '%s=%s' % (_, feats[_])
        for _ in sorted(feats)
    )
    return '%s|%s' % (pos, feats)


def parse_conll_tag(tag):
    """
    Parse a single tag from the given tag.

    Args:
        tag: (str): write your description
    """
    if '|' not in tag:
        return tag, {}

    pos, feats = tag.split('|', 1)
    feats = dict(
        _.split('=', 1)
        for _ in feats.split('|')
    )
    return pos, feats
