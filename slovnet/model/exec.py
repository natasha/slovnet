
import torch

from slovnet.visitor import Visitor
from slovnet.exec import model as exec


class ExecVisitor(Visitor):
    def visit_Parameter(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return self.visit(item.data)

    def visit_Tensor(self, item):
        """
        Create a tensor to a tensor.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        array = item.detach().numpy()
        return exec.Weight(
            array.shape,
            array.dtype.name,
            array
        )

    def visit_Linear(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        # in torch linear is xA^T + b
        weight = item.weight.transpose(1, 0)
        return exec.Linear(
            self.visit(weight),
            self.visit(item.bias)
        )

    def visit_Conv1d(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        padding, = item.padding  # tuple -> int
        return exec.Conv1d(
            self.visit(item.weight),
            self.visit(item.bias),
            padding
        )

    def visit_ReLU(self, item):
        """
        Return an item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.ReLU()

    def visit_BatchNorm1d(self, item):
        """
        Evaluate batch.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        running_std = torch.sqrt(item.running_var + item.eps)
        return exec.BatchNorm1d(
            self.visit(item.weight),
            self.visit(item.bias),
            self.visit(item.running_mean),
            self.visit(running_std),
        )

    def visit_Embedding(self, item):
        """
        Return an astroid. decledding node.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.Embedding(
            self.visit(item.weight)
        )

    def visit_NavecEmbedding(self, item):
        """
        Create a navecEmbing for the given item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        # recover initial qdim x centroids x chunk
        codes = item.codes.transpose(1, 0)
        return exec.NavecEmbedding(
            item.id,
            self.visit(item.indexes),
            self.visit(codes)
        )

    def visit_WordShapeEmbedding(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.WordShapeEmbedding(
            self.visit(item.word),
            self.visit(item.shape)
        )

    def visit_CNNEncoderLayer(self, item):
        """
        Return an astroid. encoder.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.CNNEncoderLayer(
            self.visit(item.conv),
            self.visit(item.relu),
            self.visit(item.norm)
        )

    def visit_CNNEncoder(self, item):
        """
        Return an encoder for the given item.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.CNNEncoder([
            self.visit(_)
            for _ in item.layers
        ])

    def visit_NERHead(self, item):
        """
        Convert the given item into an ast.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.NERHead(
            self.visit(item.proj),
            self.visit(item.crf)
        )

    def visit_MorphHead(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.MorphHead(
            self.visit(item.proj)
        )

    def visit_Tag(self, item):
        """
        Create an html tag.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        from slovnet.model.tag import NERHead, MorphHead

        cls = type(item.head)
        if cls is NERHead:
            Tag = exec.NER
        elif cls is MorphHead:
            Tag = exec.Morph

        return Tag(
            self.visit(item.emb),
            self.visit(item.encoder),
            self.visit(item.head)
        )

    def visit_FF(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.FF(
            self.visit(item.proj),
            self.visit(item.relu)
        )

    def visit_SyntaxHead(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.SyntaxHead(
            self.visit(item.head),
            self.visit(item.tail),
            self.visit(item.root),
            self.visit(item.kernel)
        )

    def visit_SyntaxRel(self, item):
        """
        Return an ast.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.SyntaxRel(
            self.visit(item.head),
            self.visit(item.tail),
            self.visit(item.root),
            self.visit(item.kernel)
        )

    def visit_Syntax(self, item):
        """
        Visit an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.Syntax(
            self.visit(item.emb),
            self.visit(item.encoder),
            self.visit(item.head),
            self.visit(item.rel)
        )

    def visit_CRF(self, item):
        """
        Return an astroid.

        Args:
            self: (todo): write your description
            item: (todo): write your description
        """
        return exec.CRF(
            self.visit(item.transitions)
        )


class ExecMixin:
    # super stange error if as_exec property
    # torch Module does some magic
    def to_exec(self):
        """
        Convert the program as an execitor.

        Args:
            self: (todo): write your description
        """
        visitor = ExecVisitor()
        return visitor(self)
