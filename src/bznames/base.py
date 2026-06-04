"""Base abstractions and interfaces for name samplers."""

from abc import ABC, abstractmethod
from typing import Any


class NameSampler(ABC):
    """Abstract base class for all name generator models."""

    @abstractmethod
    def fit(self, data: list[dict[str, Any]]) -> None:
        """Fit the model to the provided name frequency dataset.

        Args:
            data: A list of dictionaries containing "name" and "freq" keys.
        """
        pass

    @abstractmethod
    def compute_nll(self, name: str) -> float:
        """Compute the negative log-likelihood of a given name under the model.

        Args:
            name: The name string.

        Returns:
            The calculated negative log-likelihood as a float.
        """
        pass

    @abstractmethod
    def sample(self) -> str:
        """Sample a new name from the model's learned distribution.

        Returns:
            The generated name string.
        """
        pass
