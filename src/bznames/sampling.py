"""Module for sampling and generating names from trained models."""

from typing import Any

import numpy as np
import torch

from bznames.metrics import compute_bigram_nll_for_name
from bznames.tokenizer import CharacterEncoder


def sample_name_from_bigram_model(probs: np.ndarray, encoder: CharacterEncoder) -> str:
    """Sample a name from a bigram probability matrix.

    Args:
        probs: A 2D numpy array representing the conditional bigram probabilities.
        encoder: The CharacterEncoder to encode/decode characters.

    Returns:
        The generated name.
    """
    chars = []
    vocab_size = probs.shape[1]
    i = np.random.choice(vocab_size, p=probs[0])

    while True:
        c = encoder.decode_index(i)

        if c == encoder.special_token:
            break
        else:
            i = np.random.choice(vocab_size, p=probs[i])

        chars.append(c)

    return "".join(chars)


def compute_bigram_model_samples(
    models: dict[str, Any],
    encoder: CharacterEncoder,
    num_samples: int = 10,
) -> dict[str, dict[str, Any]]:
    """Generate sampled names and calculate statistics for multiple bigram models.

    Args:
        models: A dictionary mapping model names to their 2D conditional probability tensors/arrays.
        encoder: The CharacterEncoder to encode/decode characters.
        num_samples: Number of samples to generate per model.

    Returns:
        A dictionary mapping model name to computed sample data and statistics.
        Specifically:
        {
            model_name: {
                "samples": [{"name": str, "length": int, "nll": float | None}],
                "avg_len": float,
                "max_len": int,
                "unique_chars": int,
                "avg_nll": float | None
            }
        }
    """
    results = {}

    for model_name, model_probs in models.items():
        if model_probs is None:
            continue

        # Convert model_probs to numpy for sampling, and keep torch tensor for NLL
        if isinstance(model_probs, torch.Tensor):
            probs_tensor = model_probs
            probs_numpy = model_probs.detach().cpu().numpy()
        else:
            probs_numpy = np.asarray(model_probs)
            probs_tensor = torch.from_numpy(probs_numpy)

        samples = []
        nlls = []
        for _ in range(num_samples):
            name = sample_name_from_bigram_model(probs_numpy, encoder)
            samples.append(name)
            try:
                nll = compute_bigram_nll_for_name(name, probs_tensor, encoder)
                nlls.append(nll)
            except Exception:
                nlls.append(None)

        if not samples:
            continue

        avg_len = sum(len(s) for s in samples) / len(samples)
        unique_chars = len(set("".join(samples)))
        max_len = max(len(s) for s in samples)

        valid_nlls = [n for n in nlls if n is not None]
        avg_nll = sum(valid_nlls) / len(valid_nlls) if valid_nlls else None

        samples_list = []
        for s, nll in zip(samples, nlls, strict=True):
            samples_list.append({
                "name": s,
                "length": len(s),
                "nll": nll,
            })

        results[model_name] = {
            "samples": samples_list,
            "avg_len": avg_len,
            "max_len": max_len,
            "unique_chars": unique_chars,
            "avg_nll": avg_nll,
        }

    return results
