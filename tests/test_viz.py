"""Unit tests for visualization functions."""

from unittest.mock import patch

import pytest

from bznames.tokenizer import CharacterEncoder
from bznames.viz import display_tokenized_examples


def test_display_tokenized_examples_text(capsys) -> None:
    """Test text formatting output."""
    vocab = ["a", "b"]
    tokenizer = CharacterEncoder(vocab, special_token=".")

    input_tokens = [[0], [1], [2]]
    output_tokens = [1, 2, 0]
    freqs = [10, 20, 30]

    display_tokenized_examples(
        input_tokens,
        output_tokens,
        freqs,
        tokenizer,
        limit=2,
        format_type="text",
    )

    captured = capsys.readouterr()
    assert "Input (Context)" in captured.out
    assert "Output (Target)" in captured.out
    assert "[0] ('.')" in captured.out
    assert "1 ('a')" in captured.out


@patch("bznames.viz.display")
@patch("bznames.viz.Markdown")
def test_display_tokenized_examples_markdown(mock_markdown, mock_display) -> None:
    """Test markdown formatting output."""
    vocab = ["a", "b"]
    tokenizer = CharacterEncoder(vocab, special_token=".")

    input_tokens = [[0], [1]]
    output_tokens = [1, 0]
    freqs = [10, 20]

    display_tokenized_examples(
        input_tokens,
        output_tokens,
        freqs,
        tokenizer,
        limit=2,
        format_type="markdown",
    )

    mock_markdown.assert_called_once()
    mock_display.assert_called_once()


@patch("bznames.viz.display")
@patch("bznames.viz.HTML")
def test_display_tokenized_examples_html(mock_html, mock_display) -> None:
    """Test HTML formatting output."""
    vocab = ["a", "b"]
    tokenizer = CharacterEncoder(vocab, special_token=".")

    input_tokens = [[0], [1]]
    output_tokens = [1, 0]
    freqs = [10, 20]

    display_tokenized_examples(
        input_tokens,
        output_tokens,
        freqs,
        tokenizer,
        limit=2,
        format_type="html",
    )

    mock_html.assert_called_once()
    mock_display.assert_called_once()


def test_display_tokenized_examples_invalid_format() -> None:
    """Test that display_tokenized_examples raises ValueError for invalid format."""
    vocab = ["a", "b"]
    tokenizer = CharacterEncoder(vocab, special_token=".")

    with pytest.raises(ValueError, match="format_type must be one of"):
        display_tokenized_examples(
            [], [], [], tokenizer, format_type="invalid"
        )
