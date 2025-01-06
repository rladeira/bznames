from abc import ABC, abstractmethod
from typing import Any


class NameSampler(ABC):
    @abstractmethod
    def fit(self, data: list[dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def compute_nll(self, name: str) -> float:
        pass

    @abstractmethod
    def sample(self) -> str:
        pass
