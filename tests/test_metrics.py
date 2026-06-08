"""Unit tests for metrics and loss calculations."""

import pytest
import torch

from bznames.metrics import compute_bigram_nll_for_name, compute_bigram_nll_for_tokens
from bznames.tokenizer import CharacterEncoder


def test_compute_bigram_nll_for_tokens() -> None:
    """Test NLL calculation for bigram dataset."""
    # Mock bigram probs of shape (3, 3)
    # 0 = '.', 1 = 'a', 2 = 'b'
    # log(0.5) = -0.693147, log(1.0) = 0.0
    probs = torch.tensor(
        [
            [0.5, 0.5, 0.0],  # From . -> . (0.5), . -> a (0.5)
            [0.0, 0.0, 1.0],  # From a -> b (1.0)
            [1.0, 0.0, 0.0],  # From b -> . (1.0)
        ],
        dtype=torch.float32,
    )

    # Let's mock a dataset:
    # 1. . -> a (prob 0.5, log prob -0.693147, count/weight 2.0)
    #    -> nll contribution = 2.0 * 0.693147 = 1.386294
    # 2. a -> b (prob 1.0, log prob 0.0, count/weight 3.0) -> nll contribution = 0.0
    # 3. b -> . (prob 1.0, log prob 0.0, count/weight 5.0) -> nll contribution = 0.0
    # Expected NLL = 1.386294
    input_tokens = torch.tensor([0, 1, 2], dtype=torch.long)
    output_tokens = torch.tensor([1, 2, 0], dtype=torch.long)
    weights = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float32)

    nll = compute_bigram_nll_for_tokens(probs, input_tokens, output_tokens, weights)
    assert pytest.approx(nll, abs=1e-5) == 1.386294

    # Test that a 2D input_tokens of shape (N, 1) is handled correctly too
    input_tokens_2d = torch.tensor([[0], [1], [2]], dtype=torch.long)
    nll_2d = compute_bigram_nll_for_tokens(probs, input_tokens_2d, output_tokens, weights)
    assert pytest.approx(nll_2d, abs=1e-5) == 1.386294

    # Test error when input_tokens is 2D but shape is not (N, 1)
    input_tokens_invalid = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="input_tokens must be 1D or have shape"):
        compute_bigram_nll_for_tokens(probs, input_tokens_invalid, output_tokens, weights)


def test_compute_bigram_nll_for_name() -> None:
    """Test NLL calculation for a single name."""
    encoder = CharacterEncoder(["a", "b"], special_token=".")

    # probs mapping
    # 0 = '.', 1 = 'a', 2 = 'b'
    # log(0.5) = -0.693147, log(1.0) = 0.0
    probs = torch.tensor(
        [
            [0.5, 0.5, 0.0],  # From . -> . (0.5), . -> a (0.5)
            [0.0, 0.0, 1.0],  # From a -> b (1.0)
            [1.0, 0.0, 0.0],  # From b -> . (1.0)
        ],
        dtype=torch.float32,
    )

    # Let's test name "ab":
    # Padded: ".ab." -> transitions:
    # 1. . -> a (prob 0.5, log prob -0.693147)
    # 2. a -> b (prob 1.0, log prob 0.0)
    # 3. b -> . (prob 1.0, log prob 0.0)
    # Mean NLL = - (-0.693147 + 0.0 + 0.0) / 3 = 0.231049
    nll = compute_bigram_nll_for_name("ab", probs, encoder)
    assert pytest.approx(nll, abs=1e-5) == 0.231049


def test_compute_bigram_model_nll_comparison() -> None:
    """Test compute_bigram_model_nll_comparison calculates dataset and name NLLs."""
    from bznames.metrics import compute_bigram_model_nll_comparison

    vocab = ["a", "b"]
    encoder = CharacterEncoder(vocab, special_token=".")

    models = {
        "Test Model": torch.ones((3, 3)) / 3.0,
    }

    input_tokens = torch.zeros((5, 1), dtype=torch.long)
    output_tokens = torch.ones(5, dtype=torch.long)
    weights = torch.ones(5, dtype=torch.float32) / 5.0

    results = compute_bigram_model_nll_comparison(
        models,
        input_tokens,
        output_tokens,
        weights,
        encoder,
        test_names=["ab", "bb"],
    )

    assert "Test Model" in results
    data = results["Test Model"]
    assert "dataset_nll" in data
    assert len(data["name_nlls"]) == 2
    assert data["name_nlls"][0]["name"] == "ab"
    assert "nll" in data["name_nlls"][0]
