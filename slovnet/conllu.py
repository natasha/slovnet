

def conllu_tag(feats):
    return '|'.join(
        '%s=%s' % (_, feats[_])
        for _ in sorted(feats)
    )


def parse_conllu_tag(tag):
    return dict(
        _.split('=', 1)
        for _ in tag.split('|')
    )
