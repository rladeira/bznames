"""Unit tests for sampling and name generation."""

import numpy as np

from bznames.sampling import sample_name_from_bigram
from bznames.tokenizer import CharacterEncoder


def test_sample_name_from_bigram() -> None:
    """Test sampling names using a mock bigram probability matrix."""
    encoder = CharacterEncoder(["a", "b"], special_token=".")

    # Mock probs where:
    # From '.' (0): always go to 'a' (1)
    # From 'a' (1): always go to 'b' (2)
    # From 'b' (2): always go to '.' (0)
    probs = np.array(
        [
            [0.0, 1.0, 0.0],  # From . -> always a
            [0.0, 0.0, 1.0],  # From a -> always b
            [1.0, 0.0, 0.0],  # From b -> always .
        ]
    )

    # With deterministic probabilities, the sampled name should always be 'ab'
    name = sample_name_from_bigram(probs, encoder)
    assert name == "ab"


def test_sample_name_reproducibility() -> None:
    """Test that sampling is reproducible when numpy seed is set."""
    encoder = CharacterEncoder(["a", "b", "c"], special_token=".")

    # Equal probabilities
    probs = np.ones((4, 4)) / 4.0

    # Under seed 42, check that we get a consistent name
    np.random.seed(42)
    name1 = sample_name_from_bigram(probs, encoder)

    np.random.seed(42)
    name2 = sample_name_from_bigram(probs, encoder)

    assert name1 == name2
