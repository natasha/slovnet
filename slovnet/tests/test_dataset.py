
from .common import NERUS

from slovnet.dataset import NerusDataset


def test_dataset():
    dataset = NerusDataset(NERUS).slice(0, 5)
    for record in dataset:
        for span in record.spans:
            record.text[span.start:span.stop]
