"""Module for tokenization and character encoding for name generation."""

import itertools
from collections.abc import Iterable
from typing import Any

from tqdm import tqdm

DEFAULT_SPECIAL_TOKEN = "."


# ======================================================================
# Character Tokenizer
# ======================================================================


class CharacterTokenizer:
    """Encodes characters to indices and decodes indices to characters."""

    def __init__(self, vocab: Iterable[str], special_token: str = DEFAULT_SPECIAL_TOKEN) -> None:
        """Initialize the tokenizer with a sequence of unique characters.

        Args:
            vocab: An iterable of unique characters (excluding the special token).
            special_token: A special padding/start/end token.
        """
        unique_chars = sorted(set(vocab))
        if special_token in unique_chars:
            unique_chars.remove(special_token)

        self.special_token = special_token
        self.vocab = [special_token, *unique_chars]
        self.char_to_index = {c: i for i, c in enumerate(self.vocab)}
        self.index_to_char = dict(enumerate(self.vocab))

    @classmethod
    def from_words(
        cls, words: Iterable[str], special_token: str = DEFAULT_SPECIAL_TOKEN
    ) -> "CharacterTokenizer":
        """Initialize the tokenizer from a corpus of words.

        Args:
            words: An iterable of words (strings) to extract the vocabulary from.
            special_token: A special padding/start/end token.

        Returns:
            An instance of CharacterTokenizer.
        """
        unique_chars = sorted(set("".join(words)))
        return cls(unique_chars, special_token=special_token)

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary (including the special token)."""
        return len(self.vocab)

    def encode_char(self, char: str) -> int:
        """Encode a single character to its index.

        Args:
            char: The character to encode. Must be in the vocabulary.

        Returns:
            The integer index.

        Raises:
            KeyError: If the character is not in the vocabulary.
        """
        if char not in self.char_to_index:
            raise KeyError(f"Character {char!r} not in vocabulary.")
        return self.char_to_index[char]

    def decode_index(self, index: int) -> str:
        """Decode a single index to its character.

        Args:
            index: The index to decode.

        Returns:
            The character.

        Raises:
            KeyError: If the index is not in the vocabulary.
        """
        if index not in self.index_to_char:
            raise KeyError(f"Index {index} not in vocabulary.")
        return self.index_to_char[index]

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of indices.

        Args:
            text: The string to encode.

        Returns:
            A list of integer indices.
        """
        return [self.encode_char(c) for c in text]

    def decode(self, indices: Iterable[int]) -> str:
        """Decode an iterable of indices to a string.

        Args:
            indices: The iterable of indices to decode.

        Returns:
            The decoded string.
        """
        return "".join(self.decode_index(idx) for idx in indices)


# ======================================================================
# N-Gram Extraction
# ======================================================================


def extract_ngrams[T](
    sequence: Iterable[T], ngram_size: int, special_token: T
) -> list[tuple[T, ...]]:
    """Extract N-grams of order ngram_size from a sequence with padding.

    Args:
        sequence: The input sequence (e.g., string or list of items).
        ngram_size: The size of N-grams to extract. Must be >= 2.
        special_token: The token used to pad the sequence.

    Returns:
        A list of tuples representing the N-grams.

    Raises:
        ValueError: If ngram_size is less than 2.
    """
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2.")

    padded_sequence = (ngram_size - 1) * [special_token]
    padded_sequence.extend(sequence)
    padded_sequence.append(special_token)

    iters = (itertools.islice(padded_sequence, i, None) for i in range(ngram_size))
    return list(zip(*iters, strict=False))


# ======================================================================
# Dataset Preparation
# ======================================================================


def tokenize_dataset(
    data: Iterable[dict[str, Any]],
    tokenizer: CharacterTokenizer,
    ngram_size: int,
    show_progress: bool = False,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Tokenize and extract N-gram training data from a name dataset.

    Args:
        data: An iterable of dictionaries, each containing "name" (str) and "freq" (int) keys.
        tokenizer: The CharacterTokenizer instance to use for encoding characters.
        ngram_size: The N-gram size (ngram_size >= 2).
        show_progress: If True, displays a tqdm progress bar during processing.

    Returns:
        A tuple of (input_tokens, output_tokens, freqs) where:
            - input_tokens: List of list of encoded indices for the context.
            - output_tokens: List of encoded indices for the target character.
            - freqs: List of frequencies corresponding to each N-gram.
    """
    input_tokens: list[list[int]] = []
    output_tokens: list[int] = []
    freqs: list[int] = []

    items = tqdm(data) if show_progress else data

    for x in items:
        name = x["name"]
        freq = x["freq"]

        ngrams = extract_ngrams(name, ngram_size=ngram_size, special_token=tokenizer.special_token)
        for ngram in ngrams:
            encoded_input = [tokenizer.encode_char(c) for c in ngram[:-1]]
            encoded_output = tokenizer.encode_char(ngram[-1])
            input_tokens.append(encoded_input)
            output_tokens.append(encoded_output)
            freqs.append(freq)

    return input_tokens, output_tokens, freqs
