from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from dpdl.trainer import ClassificationAdapter, LanguageModelAdapter


def _classification_adapter() -> ClassificationAdapter:
    return ClassificationAdapter(torch.device('cpu'))


def _language_model_adapter() -> LanguageModelAdapter:
    return LanguageModelAdapter(torch.device('cpu'))


def test_classification_denominator_is_batch_size() -> None:
    adapter = _classification_adapter()
    y = torch.tensor([0, 1, 2, 1])

    # No ignored targets in classification, so the mean divides by the batch size
    assert adapter.num_loss_items(None, (None, y), None) == 4


def test_classification_denominator_ignores_feature_shape() -> None:
    adapter = _classification_adapter()
    # A larger, differently shaped X must not change the count
    X = torch.zeros(3, 8, 8)
    y = torch.tensor([2, 0, 1])

    assert adapter.num_loss_items(None, (X, y), None) == 3

def test_language_model_denominator_drops_first_column() -> None:
    adapter = _language_model_adapter()
    model = SimpleNamespace(criterion=SimpleNamespace(ignore_index=-100))

    # Every token is valid, so the count is only based on the shift:
    # The first column is dropped, leaving 2 of the 3 columns per row.
    y = torch.tensor([[5, 6, 7]])

    assert adapter.num_loss_items(model, (None, y), None) == 2

def test_language_model_denominator_counts_shifted_non_ignored_tokens() -> None:
    adapter = _language_model_adapter()
    model = SimpleNamespace(criterion=SimpleNamespace(ignore_index=-100))

    # criterion averages over shifted targets.
    # After dropping column 0, we only have two non-ignored tokens.
    y = torch.tensor(
        [
            [1, 2, 3, -100],
            [5, -100, -100, -100],
        ]
    )

    denominator = adapter.num_loss_items(model, (None, y), None)
    assert denominator == 2
    assert isinstance(denominator, int)
