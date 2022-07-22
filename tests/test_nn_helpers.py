from selfattention_audio import nn_helpers as nhelp
import torch
import pytest


def test_compute_attention_weights() -> None:
    x = torch.rand((1, 15, 30))
    x = x / x.sum(dim=1, keepdim=True)

    learned_weights = x[:, 10, :].squeeze()
    logits = (x * learned_weights).sum(dim=-1)
    weights = nhelp.compute_attention_weights(logits)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert torch.argmax(weights) == 10
    assert weights.min() >= 0 and weights.max() <= 1
