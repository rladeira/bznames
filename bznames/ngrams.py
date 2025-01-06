from collections import defaultdict
from typing import Any, Iterable

from .base import NameSampler


class NgramSampler(NameSampler):
    def __init__(self, n: int) -> None:
        self.n = n
        self.counts: dict[tuple, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_count = 0

    def fit(self, data: list[dict[str, Any]]):
        for x in data:
            name = x["name"]
            freq = x["freq"]

            for chars in get_ngrams(name, self.n):
                prev = chars[:-1]
                next = chars[-1]

                self.counts[prev][next] += freq
                self.total_count += freq

    def compute_nll(self, name: str) -> float:
        pass

    def sample(self) -> str:
        pass


def get_ngrams(name: str, n: int) -> Iterable[tuple]:
    name = (n - 1) * "." + name + "."

    return zip(*[name[i:] for i in range(n)])
