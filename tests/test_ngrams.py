"""Unit tests for N-gram functions."""

from bznames.ngrams import get_ngrams


def test_get_ngrams() -> None:
    """Test extracting character N-grams from a name."""
    ngrams = get_ngrams("test", n=2)

    assert list(ngrams) == [
        (".", "t"),
        ("t", "e"),
        ("e", "s"),
        ("s", "t"),
        ("t", "."),
    ]
