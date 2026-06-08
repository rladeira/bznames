"""Module for calculating metrics and losses like negative log-likelihood."""

import itertools
from typing import Any

import numpy as np
import torch

from bznames.tokenizer import CharacterEncoder


def compute_bigram_nll_for_tokens(
    probs: torch.Tensor,
    input_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Compute the negative log-likelihood for the tokens using bigram probabilities.

    Args:
        probs: A 2D probability tensor of shape (vocab_size, vocab_size).
        input_tokens: A 1D or 2D tensor of input token indices of shape (N,) or (N, 1).
        output_tokens: A 1D tensor of output/target token indices of shape (N,).
        weights: A 1D tensor of weights (frequencies or normalized probabilities) of shape (N,).

    Returns:
        The negative log-likelihood as a float.

    Raises:
        ValueError: If input_tokens is 2D but the second dimension is not 1.
    """
    # Auto-squeeze input_tokens if it has shape (N, 1)
    if input_tokens.ndim == 2:
        if input_tokens.shape[1] != 1:
            raise ValueError(
                f"input_tokens must be 1D or have shape (N, 1), got shape {input_tokens.shape}"
            )
        input_tokens = input_tokens.squeeze(1)

    assert input_tokens.shape == output_tokens.shape == weights.shape, (
        f"Shape mismatch: input_tokens {input_tokens.shape}, "
        f"output_tokens {output_tokens.shape}, weights {weights.shape}"
    )

    # Compute: -sum(weights * log(P(output_token | input_token)))
    nll = -1 * (weights * torch.log(probs[input_tokens, output_tokens])).sum()

    return nll.item()


def compute_bigram_nll_for_name(
    name: str,
    probs: torch.Tensor,
    encoder: CharacterEncoder,
) -> float:
    """Compute the negative log-likelihood for a single name using bigram probabilities.

    Args:
        name: The name string.
        probs: A 2D probability tensor of shape (vocab_size, vocab_size).
        encoder: The CharacterEncoder to map characters to indices.

    Returns:
        The mean negative log-likelihood for the transitions in the name.
    """
    # Pad name with the special token (e.g. ".") at both ends
    special_token = encoder.special_token
    padded_name = special_token + name + special_token
    likelihoods = []

    for c, c_next in itertools.pairwise(padded_name):
        i = encoder.encode_char(c)
        i_next = encoder.encode_char(c_next)
        likelihoods.append(probs[i, i_next].item())

    return float(-np.log(likelihoods).mean())


def compute_bigram_model_nll_comparison(
    models: dict[str, Any],
    input_tokens: Any,
    output_tokens: Any,
    weights: Any,
    encoder: CharacterEncoder,
    test_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute dataset NLL and individual name NLLs for multiple bigram models.

    Args:
        models: A dictionary mapping model names to their 2D conditional probability tensors/arrays.
        input_tokens: Tokenized input dataset.
        output_tokens: Tokenized output dataset.
        weights: Sample weights.
        encoder: The CharacterEncoder to encode/decode characters.
        test_names: List of name strings to compute NLLs for.

    Returns:
        A dictionary mapping model names to NLL results.
        Specifically:
        {
            model_name: {
                "dataset_nll": float,
                "name_nlls": list[dict[str, Any]] # Each item: {"name": str, "nll": float | None}
            }
        }
    """
    results = {}

    for model_name, model_probs in models.items():
        if model_probs is None:
            continue

        # Convert to torch tensor
        if isinstance(model_probs, torch.Tensor):
            probs_tensor = model_probs
        else:
            probs_tensor = torch.from_numpy(np.asarray(model_probs))

        # Compute dataset NLL
        dataset_nll = compute_bigram_nll_for_tokens(
            probs_tensor,
            input_tokens,
            output_tokens,
            weights,
        )

        # Compute NLL for each test name
        name_nlls = []
        for name in test_names:
            try:
                nll = compute_bigram_nll_for_name(name, probs_tensor, encoder)
            except Exception:
                nll = None

            name_nlls.append({
                "name": name,
                "nll": nll,
            })

        results[model_name] = {
            "dataset_nll": dataset_nll,
            "name_nlls": name_nlls,
        }

    return results
