
import torch

from slovnet.visitor import Visitor
from slovnet.infer import impl as infer


class InferVisitor(Visitor):
    def visit_Parameter(self, item):
        return self.visit_Tensor(item.data)

    def visit_Tensor(self, item):
        array = item.detach().numpy()
        return array.view(infer.Weight)

    def visit_Sequential(self, item):
        return infer.Sequential([
            self.visit(_)
            for _ in item
        ])

    def visit_Linear(self, item):
        # in torch linear is xA^T + b
        weight = item.weight.transpose(1, 0)
        return infer.Linear(
            self.visit(weight),
            self.visit(item.bias)
        )

    def visit_Conv1d(self, item):
        padding, = item.padding  # tuple -> int
        return infer.Conv1d(
            self.visit(item.weight),
            self.visit(item.bias),
            padding
        )

    def visit_ReLU(self, item):
        return infer.ReLU()

    def visit_BatchNorm1d(self, item):
        running_std = torch.sqrt(item.running_var + item.eps)
        return infer.BatchNorm1d(
            self.visit(item.weight),
            self.visit(item.bias),
            self.visit(item.running_mean),
            self.visit(running_std),
        )

    def visit_ShapeEmbedding(self, item):
        return infer.Embedding(
            self.visit(item.weight)
        )

    def visit_NavecEmbedding(self, item):
        # recover initial qdim x centroids x chunk
        codes = item.codes.transpose(1, 0)
        return infer.NavecEmbedding(
            item.id,
            self.visit(item.indexes),
            self.visit(codes)
        )

    def visit_WordModel(self, item):
        return infer.StackEmbedding([
            self.visit(item.word_emb),
            self.visit(item.shape_emb)
        ])

    def visit_CNNContextModel(self, item):
        return infer.Sequential([
            infer.Transpose(2, 1),
            infer.Sequential([
                self.visit(_)
                for _ in item.layers
            ]),
            infer.Transpose(2, 1)
        ])

    def visit_CRF(self, item):
        return infer.CRF(
            self.visit(item.transitions)
        )

    def visit_CRFTagModel(self, item):
        return infer.Sequential([
            self.visit(item.proj),
            self.visit(item.crf)
        ])

    def visit_NERModel(self, item):
        return infer.Sequential([
            self.visit(item.word_model),
            self.visit(item.context_model),
            self.visit(item.tag_model)
        ])


class InferMixin:
    @property
    def as_infer(self):
        visitor = InferVisitor()
        return visitor(self)
