import torch
import pytest


@pytest.fixture
def test_eval_code():
    labels = torch.arange(0, 10).float()
    preds = torch.cat([torch.arange(0, 5), torch.ones(5)])
    return labels, preds


def test_metrics(test_eval_code):
    preds, labels = test_eval_code
    accuracy = preds.eq(labels.view_as(preds)).sum().item() / preds.size(0)
    assert accuracy == 0.5
