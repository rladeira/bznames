"""Module for sampling and learning N-gram models for name generation."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .base import NameSampler


class NgramSampler(NameSampler):
    """Sampler based on character-level N-gram count distributions."""

    def __init__(self, n: int) -> None:
        """Initialize the N-gram sampler with context size n.

        Args:
            n: The order of the N-gram (context length + 1).
        """
        self.n = n
        self.counts: dict[tuple, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_count = 0

    def fit(self, data: list[dict[str, Any]]) -> None:
        """Fit the N-gram frequency distribution on name datasets.

        Args:
            data: A list of dicts with keys "name" and "freq".
        """
        for x in data:
            name = x["name"]
            freq = x["freq"]

            for chars in get_ngrams(name, self.n):
                prev = chars[:-1]
                next_char = chars[-1]

                self.counts[prev][next_char] += freq
                self.total_count += freq

    def compute_nll(self, name: str) -> float:
        """Compute the Negative Log-Likelihood of a given name.

        Args:
            name: The name to compute NLL for.

        Returns:
            The calculated negative log-likelihood.
        """
        raise NotImplementedError

    def sample(self) -> str:
        """Sample a new name from the learned N-gram model.

        Returns:
            The generated name string.
        """
        raise NotImplementedError


def get_ngrams(name: str, n: int) -> Iterable[tuple]:
    """Extract character N-grams of order n from a name.

    Args:
        name: The input name string.
        n: The size of N-grams to extract.

    Returns:
        An iterable of character tuples.
    """
    name = (n - 1) * "." + name + "."

    return zip(*[name[i:] for i in range(n)], strict=False)
