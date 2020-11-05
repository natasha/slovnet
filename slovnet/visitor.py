

class Visitor(object):
    def resolve_method(self, item):
        """
        Resolve a method.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        for cls in item.__class__.__mro__:
            name = 'visit_' + cls.__name__
            method = getattr(self, name, None)
            if method:
                return method
        raise ValueError('no method for {type!r}'.format(
            type=type(item)
        ))

    def visit(self, item):
        """
        Called when an item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return self.resolve_method(item)(item)

    def __call__(self, item):
        """
        Call the callable call.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return self.visit(item)
