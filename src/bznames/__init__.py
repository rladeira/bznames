"""Brazilian name generator package using language models."""

from bznames.metrics import compute_bigram_nll_for_name, compute_bigram_nll_for_tokens

__all__ = ["compute_bigram_nll_for_name", "compute_bigram_nll_for_tokens"]
