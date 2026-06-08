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


@patch("bznames.viz.display")
@patch("bznames.viz.HTML")
def test_display_bigram_model_samples(mock_html, mock_display) -> None:
    """Test display_bigram_model_samples generation and formatting output."""
    import torch

    vocab = ["a", "b", "c", "d", "e", "f", "x", "y", "z"]
    encoder = CharacterEncoder(vocab, special_token=".")

    models = {
        "Computed Model": torch.ones((10, 10)) / 10.0,
        "Uniform Model": torch.ones((10, 10)) / 10.0,
    }

    from bznames.viz import display_bigram_model_samples

    display_bigram_model_samples(models, encoder, num_samples=3)

    mock_html.assert_called_once()
    mock_display.assert_called_once()

    # Retrieve HTML passed to viz.HTML
    called_html = mock_html.call_args[0][0]
    assert "Avg NLL:" in called_html
    assert "nll" in called_html


@patch("bznames.viz.display")
@patch("bznames.viz.HTML")
def test_display_bigram_model_nll_comparison(mock_html, mock_display) -> None:
    """Test display_bigram_model_nll_comparison outputs formatted HTML."""
    import torch

    vocab = ["a", "b", "c", "d", "e", "f", "x", "y", "z"]
    encoder = CharacterEncoder(vocab, special_token=".")

    models = {
        "Computed Model": {
            "probs": torch.ones((10, 10)) / 10.0,
            "dataset_nll": 1.25,
        },
        "Uniform Model": {
            "probs": torch.ones((10, 10)) / 10.0,
            "dataset_nll": 1.75,
        },
    }

    from bznames.viz import display_bigram_model_nll_comparison

    display_bigram_model_nll_comparison(
        models,
        encoder,
        test_names=["abc", "def"],
    )

    mock_html.assert_called_once()
    mock_display.assert_called_once()

    called_html = mock_html.call_args[0][0]
    assert "Dataset NLL:" in called_html
    assert "Name-Level NLL" in called_html
    assert "nll" in called_html


def test_render_tokenized_examples_formats() -> None:
    """Test internal render functions for tokenized examples."""
    from bznames.viz import (
        _render_tokenized_examples_html,
        _render_tokenized_examples_markdown,
        _render_tokenized_examples_text,
    )

    examples = [
        {
            "input_tokens": [0],
            "output_token": 1,
            "context_str": ".",
            "target_str": "a",
            "bigram_str": ".a",
            "frequency": 10,
        }
    ]

    text_out = _render_tokenized_examples_text(examples)
    assert "Input (Context)" in text_out
    assert "[0] ('.')" in text_out
    assert "1 ('a')" in text_out

    md_out = _render_tokenized_examples_markdown(examples)
    assert "| Input (Context) |" in md_out
    assert "`[0]` ('.')" in md_out
    assert "`1` ('a')" in md_out

    html_out = _render_tokenized_examples_html(examples)
    assert '<table class="bznames-table">' in html_out
    assert "[0]" in html_out
    assert "1" in html_out


def test_render_bigram_model_samples_html() -> None:
    """Test HTML rendering of bigram model samples."""
    from bznames.viz import _render_bigram_model_samples_html

    samples_data = {
        "Test Model": {
            "samples": [
                {"name": "ab", "length": 2, "nll": 1.23},
                {"name": "bb", "length": 2, "nll": 2.34},
            ],
            "avg_len": 2.0,
            "max_len": 2,
            "unique_chars": 2,
            "avg_nll": 1.78,
        }
    }

    html_out = _render_bigram_model_samples_html(samples_data)
    assert "Test Model" in html_out
    assert "ab" in html_out
    assert "1.23 nll" in html_out
    assert "Avg NLL: 1.78" in html_out


def test_render_bigram_model_nll_comparison_html() -> None:
    """Test HTML rendering of bigram model NLL comparison."""
    from bznames.viz import _render_bigram_model_nll_comparison_html

    nll_data = {
        "Test Model": {
            "dataset_nll": 1.5,
            "name_nlls": [
                {"name": "ab", "nll": 1.25},
                {"name": "bb", "nll": 1.75},
            ],
        }
    }

    html_out = _render_bigram_model_nll_comparison_html(nll_data)
    assert "Test Model" in html_out
    assert "Dataset NLL: 1.500" in html_out
    assert "'ab'" in html_out
    assert "1.250 nll" in html_out
