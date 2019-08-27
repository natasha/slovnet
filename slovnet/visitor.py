

class Visitor(object):
    def resolve_method(self, item):
        for cls in item.__class__.__mro__:
            name = 'visit_' + cls.__name__
            method = getattr(self, name, None)
            if method:
                return method
        raise ValueError('no method for {type!r}'.format(
            type=type(item)
        ))

    def visit(self, item):
        return self.resolve_method(item)(item)

    def __call__(self, item):
        return self.visit(item)
