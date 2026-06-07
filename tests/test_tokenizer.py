"""Unit tests for N-gram functions and character tokenization."""

import pytest

from bznames.tokenizer import CharacterTokenizer, extract_ngrams, tokenize_dataset


def test_extract_ngrams() -> None:
    """Test extracting character N-grams from a name."""
    ngrams = extract_ngrams("test", ngram_size=2, special_token=".")

    assert ngrams == [
        (".", "t"),
        ("t", "e"),
        ("e", "s"),
        ("s", "t"),
        ("t", "."),
    ]


def test_extract_ngrams_invalid_n() -> None:
    """Test that extract_ngrams raises ValueError when ngram_size < 2."""
    with pytest.raises(ValueError, match=r"ngram_size must be at least 2\."):
        extract_ngrams("test", ngram_size=1, special_token=".")


def test_character_tokenizer() -> None:
    """Test CharacterTokenizer setup, encoding, and decoding."""
    vocab = ["a", "b", "e", "f", "g", "i", "l", "r"]
    tokenizer = CharacterTokenizer(vocab, special_token=".")

    # Vocabulary should be: '.', 'a', 'b', 'e', 'f', 'g', 'i', 'l', 'r'
    assert tokenizer.vocab == [".", "a", "b", "e", "f", "g", "i", "l", "r"]
    assert tokenizer.vocab_size == 9

    # Test single char encoding
    assert tokenizer.encode_char(".") == 0
    assert tokenizer.encode_char("a") == 1
    assert tokenizer.encode_char("r") == 8

    # Test single index decoding
    assert tokenizer.decode_index(0) == "."
    assert tokenizer.decode_index(1) == "a"
    assert tokenizer.decode_index(8) == "r"

    # Test string encoding
    assert tokenizer.encode("rafael") == [8, 1, 4, 1, 3, 7]

    # Test list decoding
    assert tokenizer.decode([8, 1, 4, 1, 3, 7]) == "rafael"

    # Test errors
    with pytest.raises(KeyError):
        tokenizer.encode_char("z")

    with pytest.raises(KeyError):
        tokenizer.decode_index(9)


def test_character_tokenizer_from_words() -> None:
    """Test initializing CharacterTokenizer from a corpus of words."""
    words = ["rafael", "gabriela"]
    tokenizer = CharacterTokenizer.from_words(words, special_token=".")

    assert tokenizer.vocab == [".", "a", "b", "e", "f", "g", "i", "l", "r"]
    assert tokenizer.vocab_size == 9


def test_tokenize_dataset() -> None:
    """Test tokenize_dataset extracts and encodes bigrams correctly."""
    data = [
        {"name": "ab", "freq": 10},
        {"name": "ba", "freq": 5},
    ]
    tokenizer = CharacterTokenizer(["a", "b"], special_token=".")

    input_tokens, output_tokens, freqs = tokenize_dataset(data, tokenizer, ngram_size=2)

    assert input_tokens == [[0], [1], [2], [0], [2], [1]]
    assert output_tokens == [1, 2, 0, 2, 1, 0]
    assert freqs == [10, 10, 10, 5, 5, 5]
