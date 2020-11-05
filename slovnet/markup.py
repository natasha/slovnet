
try:
    from ipymarkup import (
        show_span_box_markup,
        show_dep_markup
    )
except ImportError:
    pass

from .record import Record
from .bio import (
    spans_bio,
    bio_spans
)
from .token import find_tokens
from .sent import sentenize
from .span import (
    Span,
    envelop_spans,
    offset_spans
)
from .conll import (
    format_conll_tag,
    parse_conll_tag
)


########
#
#   SPAN
#
#######


class SpanMarkup(Record):
    __attributes__ = ['text', 'spans']
    __annotations__ = {
        'spans': [Span]
    }

    @property
    def sents(self):
        """
        Return a list of text.

        Args:
            self: (todo): write your description
        """
        for sent in sentenize(self.text):
            spans = envelop_spans(sent, self.spans)
            spans = offset_spans(spans, -sent.start)
            yield SpanMarkup(sent.text, list(spans))

    def to_bio(self, tokens):
        """
        Convert a list of tokens into a list of tokens.

        Args:
            self: (todo): write your description
            tokens: (str): write your description
        """
        tags = spans_bio(tokens, self.spans)
        words = [_.text for _ in tokens]
        return BIOMarkup.from_tuples(zip(words, tags))


def show_span_markup(markup):
    """
    Show the marker marker marker.

    Args:
        markup: (todo): write your description
    """
    show_span_box_markup(markup.text, markup.spans)


########
#
#   TAG
#
######


class TagToken(Record):
    __attributes__ = ['text', 'tag']


class TagMarkup(Record):
    __attributes__ = ['tokens']
    __annotations__ = {
        'tokens': [TagToken]
    }

    @property
    def words(self):
        """
        Returns a list of words.

        Args:
            self: (todo): write your description
        """
        return [_.text for _ in self.tokens]

    @property
    def tags(self):
        """
        The list of tags.

        Args:
            self: (todo): write your description
        """
        return [_.tag for _ in self.tokens]

    @classmethod
    def from_tuples(cls, tuples):
        """
        Constructs a list of tuples from a list of tuples.

        Args:
            cls: (todo): write your description
            tuples: (str): write your description
        """
        return cls([
            TagToken(word, tag)
            for word, tag in tuples
        ])


class BIOMarkup(TagMarkup):
    def to_span(self, text):
        """
        Converts a list of a list to a list of words.

        Args:
            self: (todo): write your description
            text: (str): write your description
        """
        tokens = find_tokens(text, self.words)
        spans = list(bio_spans(tokens, self.tags))
        return SpanMarkup(text, spans)


########
#
#   MORPH
#
########


class MorphToken(TagToken):
    __attributes__ = ['text', 'pos', 'feats']

    @property
    def tag(self):
        """
        The list of this word.

        Args:
            self: (todo): write your description
        """
        return format_conll_tag(self.pos, self.feats)


class MorphMarkup(TagMarkup):
    __attributes__ = ['tokens']
    __annotations__ = {
        'tokens': [MorphToken]
    }

    @classmethod
    def from_tuples(cls, tuples):
        """
        Creates a list of word objects.

        Args:
            cls: (todo): write your description
            tuples: (str): write your description
        """
        tokens = []
        for word, tag in tuples:
            pos, feats = parse_conll_tag(tag)
            tokens.append(MorphToken(word, pos, feats))
        return cls(tokens)


def format_morph_markup(markup, size=20):
    """
    Generate markup tags.

    Args:
        markup: (todo): write your description
        size: (int): write your description
    """
    for word, tag in zip(markup.words, markup.tags):
        word = word.rjust(size)
        yield '%s %s' % (word, tag)


def show_morph_markup(markup):
    """
    Displays the markers in the given markup.

    Args:
        markup: (todo): write your description
    """
    for line in format_morph_markup(markup):
        print(line)


def format_morph_markup_diff(a, b, size=20):
    """
    Yields - in a string.

    Args:
        a: (int): write your description
        b: (int): write your description
        size: (int): write your description
    """
    for word, a_token, b_token in zip(a.words, a.tokens, b.tokens):
        word = word.rjust(size)
        a_tag = format_conll_tag(a_token.pos, a_token.feats)
        yield '%s   %s' % (word, a_tag)
        if a_token != b_token:
            word = ' ' * size
            b_tag = format_conll_tag(b_token.pos, b_token.feats)
            yield '%s ! %s' % (word, b_tag)


def show_morph_markup_diff(a, b):
    """
    Show the difference between two diffs.

    Args:
        a: (int): write your description
        b: (int): write your description
    """
    for line in format_morph_markup_diff(a, b):
        print(line)


#######
#
#   SYNTAX
#
#######


class SyntaxToken(TagToken):
    __attributes__ = ['id', 'text', 'head_id', 'rel']


class SyntaxMarkup(TagMarkup):
    __attributes__ = ['tokens']
    __annotations__ = {
        'tokens': [SyntaxToken]
    }

    @classmethod
    def from_tuples(cls, tuples):
        """
        Create a list of tuples of tuples of - tuples.

        Args:
            cls: (todo): write your description
            tuples: (str): write your description
        """
        return cls([
            SyntaxToken(id, text, head_id, rel)
            for id, text, head_id, rel in tuples
        ])


def syntax_markup_deps(tokens):
    """
    Yields all markdown.

    Args:
        tokens: (str): write your description
    """
    for token in tokens:
        id = int(token.id)
        head_id = int(token.head_id)
        # skip root=0, skip loop
        # ipymarkup crashes
        if head_id == 0 or head_id == id:
            continue

        rel = token.rel
        id = id - 1
        head_id = head_id - 1
        yield head_id, id, rel


def show_syntax_markup(markup):
    """
    Show the contents of the given markdown file.

    Args:
        markup: (todo): write your description
    """
    deps = syntax_markup_deps(markup.tokens)
    show_dep_markup(markup.words, deps)
