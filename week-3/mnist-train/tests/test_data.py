import pytest
from pathlib import Path
from mnist_train.utils import get_config
from mnist_train.data import load_data
import pandas as pd
import torch

root = Path(__file__).parent.parent
path_config = root / 'conf/config.yaml'


@pytest.fixture(scope='session')
def load_labels():
    config = get_config(path_config)
    dataset_train, _ = load_data(root / Path(config['dir_data']))

    shapes, labels = [], []
    for i in range(len(dataset_train)):
        x, y = dataset_train[i]
        labels.append(y)
        shapes.append(x.shape)
    return dataset_train, labels, shapes


def test_dist_y(load_labels):
    _, labels, _ = load_labels
    rng = (0.08, 0.12)
    dist = pd.Series(labels).value_counts(normalize=True).to_dict()
    for lbl, dst in dist.items():
        assert rng[0] <= dst <= rng[1], f"Label {lbl} out of range {dst}"


def test_shapes(load_labels):
    _, _, shapes = load_labels
    assert all(sh == torch.Size([1, 28, 28]) for sh in shapes)
