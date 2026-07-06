"""Unit tests for the single-layer linear bigram model."""

import math

import pytest
import torch

from bznames.linear_model import (
    compute_linear_nn_nll_for_name,
    compute_linear_nn_nll_for_tokens,
)
from bznames.tokenizer import CharacterEncoder


def test_compute_linear_nn_nll_for_tokens_zero_weights() -> None:
    """Zero weights give a uniform softmax, so the NLL is the weighted log(vocab_size)."""
    vocab_size = 3
    W = torch.zeros((vocab_size, vocab_size))

    # Targets and per-example weights are arbitrary here since every row of a
    # zero-W softmax is uniform (prob 1/vocab_size for any target).
    input_tokens = torch.tensor([0, 1], dtype=torch.long)
    output_tokens = torch.tensor([1, 2], dtype=torch.long)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float32)

    nll = compute_linear_nn_nll_for_tokens(W, input_tokens, output_tokens, weights)

    # -sum(w * log(1/vocab_size)) = log(vocab_size) * sum(w)
    expected = math.log(vocab_size) * weights.sum().item()
    assert pytest.approx(nll.item(), abs=1e-6) == expected


def test_compute_linear_nn_nll_for_tokens_squeezes_2d_input() -> None:
    """Input tokens of shape (N, 1) are accepted and give the same result as (N,)."""
    vocab_size = 3
    W = torch.zeros((vocab_size, vocab_size))
    output_tokens = torch.tensor([1, 2], dtype=torch.long)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

    nll_1d = compute_linear_nn_nll_for_tokens(
        W, torch.tensor([0, 1], dtype=torch.long), output_tokens, weights
    )
    nll_2d = compute_linear_nn_nll_for_tokens(
        W, torch.tensor([[0], [1]], dtype=torch.long), output_tokens, weights
    )
    assert pytest.approx(nll_2d.item(), abs=1e-6) == nll_1d.item()

    # A 2D input whose second dim is not 1 is an error
    bad_input = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="input_tokens must be 1D or have shape"):
        compute_linear_nn_nll_for_tokens(W, bad_input, output_tokens, weights)


def test_compute_linear_nn_nll_for_tokens_is_differentiable() -> None:
    """The returned scalar tensor must support backprop through W."""
    vocab_size = 3
    W = torch.zeros((vocab_size, vocab_size), requires_grad=True)
    input_tokens = torch.tensor([0, 1], dtype=torch.long)
    output_tokens = torch.tensor([1, 2], dtype=torch.long)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

    nll = compute_linear_nn_nll_for_tokens(W, input_tokens, output_tokens, weights)
    assert isinstance(nll, torch.Tensor)

    nll.backward()
    assert W.grad is not None
    assert W.grad.shape == W.shape


def test_compute_linear_nn_nll_for_name_zero_weights() -> None:
    """With zero weights every prediction is uniform, so each name's NLL is log(vocab_size)."""
    encoder = CharacterEncoder(["a", "b"], special_token=".")
    W = torch.zeros((encoder.vocab_size, encoder.vocab_size))

    nll = compute_linear_nn_nll_for_name("ab", W, encoder)

    assert pytest.approx(nll, abs=1e-6) == math.log(encoder.vocab_size)


def test_compute_linear_nn_nll_for_name_recovers_confident_transitions() -> None:
    """Large logits toward the true next character drive the name NLL toward zero."""
    encoder = CharacterEncoder(["a", "b"], special_token=".")
    # Index map: 0 = '.', 1 = 'a', 2 = 'b'. Name "ab" pads to ".ab." with transitions
    # . -> a, a -> b, b -> . ; make each of those logits dominate its row.
    W = torch.full((encoder.vocab_size, encoder.vocab_size), -50.0)
    W[0, 1] = 50.0  # . -> a
    W[1, 2] = 50.0  # a -> b
    W[2, 0] = 50.0  # b -> .

    nll = compute_linear_nn_nll_for_name("ab", W, encoder)

    assert nll == pytest.approx(0.0, abs=1e-6)
