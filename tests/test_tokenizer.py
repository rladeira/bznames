"""Unit tests for N-gram functions."""

import pytest

from bznames.tokenizer import CharacterEncoder, extract_ngrams


def test_extract_ngrams() -> None:
    """Test extracting character N-grams from a name."""
    ngrams = extract_ngrams("test", n=2, special_token=".")

    assert ngrams == [
        (".", "t"),
        ("t", "e"),
        ("e", "s"),
        ("s", "t"),
        ("t", "."),
    ]


def test_extract_ngrams_invalid_n() -> None:
    """Test that extract_ngrams raises ValueError when n < 2."""
    with pytest.raises(ValueError, match=r"n must be at least 2\."):
        extract_ngrams("test", n=1, special_token=".")


def test_character_encoder() -> None:
    """Test CharacterEncoder setup, encoding, and decoding."""
    vocab = ["a", "b", "e", "f", "g", "i", "l", "r"]
    encoder = CharacterEncoder(vocab, special_token=".")

    # Vocabulary should be: '.', 'a', 'b', 'e', 'f', 'g', 'i', 'l', 'r'
    assert encoder.vocab == [".", "a", "b", "e", "f", "g", "i", "l", "r"]
    assert encoder.vocab_size == 9

    # Test single char encoding
    assert encoder.encode_char(".") == 0
    assert encoder.encode_char("a") == 1
    assert encoder.encode_char("r") == 8

    # Test single index decoding
    assert encoder.decode_index(0) == "."
    assert encoder.decode_index(1) == "a"
    assert encoder.decode_index(8) == "r"

    # Test string encoding
    assert encoder.encode("rafael") == [8, 1, 4, 1, 3, 7]

    # Test list decoding
    assert encoder.decode([8, 1, 4, 1, 3, 7]) == "rafael"

    # Test errors
    with pytest.raises(KeyError):
        encoder.encode_char("z")

    with pytest.raises(KeyError):
        encoder.decode_index(9)


def test_character_encoder_from_words() -> None:
    """Test initializing CharacterEncoder from a corpus of words."""
    words = ["rafael", "gabriela"]
    encoder = CharacterEncoder.from_words(words, special_token=".")

    assert encoder.vocab == [".", "a", "b", "e", "f", "g", "i", "l", "r"]
    assert encoder.vocab_size == 9
