"""Visualization tools for bznames models and tokenizers."""

from importlib.resources import files
from typing import Any

import numpy as np
import torch
from IPython.display import HTML, Markdown, display

from bznames.metrics import compute_bigram_nll_for_name
from bznames.sampling import compute_bigram_model_samples
from bznames.tokenizer import CharacterEncoder, compute_tokenized_examples

_STYLES_CSS = files("bznames").joinpath("styles.css").read_text(encoding="utf-8")
_STYLE_TAG = f"<style>\n{_STYLES_CSS}\n</style>"


def display_tokenized_examples(
    input_tokens: Any,
    output_tokens: Any,
    freqs: Any,
    encoder: CharacterEncoder,
    limit: int = 10,
    format_type: str = "text",
) -> None:
    """Display a preview of the tokenized dataset in text, markdown, or HTML format.

    Args:
        input_tokens: List or tensor of encoded inputs.
        output_tokens: List or tensor of encoded outputs.
        freqs: List or tensor of frequencies.
        encoder: The CharacterEncoder used to encode the dataset.
        limit: Number of examples to display.
        format_type: Display format. Can be "text", "markdown", or "html".

    Raises:
        ValueError: If format_type is not "text", "markdown", or "html".
    """
    valid_formats = {"text", "markdown", "html"}
    if format_type not in valid_formats:
        raise ValueError(f"format_type must be one of {valid_formats}, got {format_type!r}")

    examples = compute_tokenized_examples(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        freqs=freqs,
        encoder=encoder,
        limit=limit,
    )

    if format_type == "text":
        print(_render_tokenized_examples_text(examples))
    elif format_type == "markdown":
        display(Markdown(_render_tokenized_examples_markdown(examples)))
    elif format_type == "html":
        display(HTML(_render_tokenized_examples_html(examples)))


def display_bigram_model_samples(
    models: dict[str, Any],
    encoder: CharacterEncoder,
    num_samples: int = 10,
) -> None:
    """Generate and display sampled names comparison from bigram models side-by-side.

    Args:
        models: A dictionary mapping model names to their 2D conditional probability tensors/arrays.
        encoder: The CharacterEncoder to encode/decode characters.
        num_samples: Number of samples to generate per model.
    """
    samples_data = compute_bigram_model_samples(
        models=models,
        encoder=encoder,
        num_samples=num_samples,
    )
    html = _render_bigram_model_samples_html(samples_data)
    display(HTML(html))


def display_bigram_model_nll_comparison(
    models: dict[str, dict[str, Any]],
    encoder: CharacterEncoder,
    test_names: list[str],
) -> None:
    """Display side-by-side comparison of dataset NLL and individual name NLLs for bigram models.

    Args:
        models: A dictionary mapping model names to dictionaries containing:
            - "probs": The 2D conditional probability tensor/array.
            - "dataset_nll": The precomputed dataset negative log-likelihood (float).
        encoder: The CharacterEncoder to encode/decode characters.
        test_names: List of name strings to compute NLLs for.
    """
    nll_data = {}
    for model_name, model_info in models.items():
        probs = model_info["probs"]
        dataset_nll = model_info["dataset_nll"]

        # Convert to torch tensor
        if isinstance(probs, torch.Tensor):
            probs_tensor = probs
        else:
            probs_tensor = torch.from_numpy(np.asarray(probs))

        # Compute NLL for each test name
        name_nlls = []
        for name in test_names:
            try:
                nll = compute_bigram_nll_for_name(name, probs_tensor, encoder)
            except Exception:
                nll = None

            name_nlls.append({
                "name": name,
                "nll": nll,
            })

        nll_data[model_name] = {
            "dataset_nll": dataset_nll,
            "name_nlls": name_nlls,
        }

    html = _render_bigram_model_nll_comparison_html(nll_data)
    display(HTML(html))


# ======================================================================
# Internal HTML/Markdown/Text Renderers
# ======================================================================


def _render_tokenized_examples_text(examples: list[dict[str, Any]]) -> str:
    """Format tokenized examples as plain text.

    Args:
        examples: List of tokenized examples metadata.

    Returns:
        The formatted plain text string.
    """
    headers = (
        f"{'Input (Context)':<20} -> {'Output (Target)':<18} | "
        f"{'Bigram':<8} | {'Frequency':<10}"
    )
    lines = [headers, "-" * 65]
    for example in examples:
        inp = example["input_tokens"]
        out = example["output_token"]
        context_str = example["context_str"]
        target_str = example["target_str"]
        bigram_str = example["bigram_str"]
        freq = example["frequency"]

        context_repr = f"[{', '.join(map(str, inp))}] ({context_str!r})"
        target_repr = f"{out} ({target_str!r})"

        lines.append(f"{context_repr:<20} -> {target_repr:<18} | {bigram_str!r:<8} | {freq:,}")
    return "\n".join(lines)


def _render_tokenized_examples_markdown(examples: list[dict[str, Any]]) -> str:
    """Format tokenized examples as markdown.

    Args:
        examples: List of tokenized examples metadata.

    Returns:
        The formatted markdown string.
    """
    lines = [
        "| Input (Context) | Output (Target) | Bigram | Frequency |",
        "|:---|:---|:---:|---:|",
    ]
    for example in examples:
        inp = example["input_tokens"]
        out = example["output_token"]
        context_str = example["context_str"]
        target_str = example["target_str"]
        bigram_str = example["bigram_str"]
        freq = example["frequency"]

        context_repr = f"`[{', '.join(map(str, inp))}]` ({context_str!r})"
        target_repr = f"`{out}` ({target_str!r})"

        lines.append(f"| {context_repr} | {target_repr} | `{bigram_str!r}` | {freq:,} |")
    return "\n".join(lines)


def _render_tokenized_examples_html(examples: list[dict[str, Any]]) -> str:
    """Format tokenized examples as HTML.

    Args:
        examples: List of tokenized examples metadata.

    Returns:
        The formatted HTML table string.
    """
    rows = []
    for example in examples:
        inp = example["input_tokens"]
        out = example["output_token"]
        context_str = example["context_str"]
        target_str = example["target_str"]
        bigram_str = example["bigram_str"]
        freq = example["frequency"]

        inp_list_str = ", ".join(map(str, inp))

        row = (
            "<tr>\n"
            f'  <td><code class="bznames-code-token">[{inp_list_str}]</code>'
            f'<span class="bznames-char-lbl">{context_str!r}</span></td>\n'
            f'  <td><code class="bznames-code-token">{out}</code>'
            f'<span class="bznames-char-lbl">{target_str!r}</span></td>\n'
            f'  <td style="text-align: center;">'
            f'<span class="bznames-bigram-lbl">{bigram_str!r}</span></td>\n'
            f'  <td style="text-align: right;" class="bznames-freq-lbl">{freq:,}</td>\n'
            "</tr>"
        )
        rows.append(row)

    table_html = f"""{_STYLE_TAG}
    <table class="bznames-table">
      <thead>
        <tr>
          <th>Input (Context)</th>
          <th>Output (Target)</th>
          <th style="text-align: center;">Bigram</th>
          <th style="text-align: right;">Frequency</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """
    return table_html


def _render_bigram_model_samples_html(samples_data: dict[str, dict[str, Any]]) -> str:
    """Format bigram model samples and metrics as HTML cards.

    Args:
        samples_data: Dict of model samples and metrics.

    Returns:
        The formatted HTML cards string.
    """
    card_htmls = []

    for model_name, data in samples_data.items():
        # Determine header class
        header_class = (
            "computed"
            if "computed" in model_name.lower() or "trained" in model_name.lower()
            else "uniform"
        )

        items_html = []
        for sample in data["samples"]:
            s = sample["name"]
            length = sample["length"]
            nll = sample["nll"]
            nll_html = ""
            if nll is not None:
                nll_html = f'<span class="sample-item-nll">{nll:.2f} nll</span>'

            items_html.append(
                f'  <li class="sample-item">'
                f'    <span class="sample-item-text">{s}</span>'
                f'    <div class="sample-item-metrics">'
                f'      <span class="sample-item-length">{length} ch</span>'
                f"      {nll_html}"
                f"    </div>"
                f"  </li>"
            )

        nll_meta_html = ""
        avg_nll = data["avg_nll"]
        if avg_nll is not None:
            nll_meta_html = f'<span class="sample-meta-item">Avg NLL: {avg_nll:.2f}</span>'

        card = f"""
        <div class="sample-card">
          <div class="sample-card-header {header_class}">{model_name}</div>
          <div class="sample-meta">
            <span class="sample-meta-item">Avg: {data["avg_len"]:.1f} ch</span>
            <span class="sample-meta-item">Max: {data["max_len"]} ch</span>
            <span class="sample-meta-item">Unique: {data["unique_chars"]} ch</span>
            {nll_meta_html}
          </div>
          <ul class="sample-list">
            {"".join(items_html)}
          </ul>
        </div>
        """
        card_htmls.append(card)

    html = f"""{_STYLE_TAG}
    <div class="sample-comparison-container">
      {"".join(card_htmls)}
    </div>
    """
    return html


def _render_bigram_model_nll_comparison_html(
    nll_data: dict[str, dict[str, Any]]
) -> str:
    """Format NLL comparison metrics as HTML cards.

    Args:
        nll_data: Dict of NLL comparison metrics.

    Returns:
        The formatted HTML cards string.
    """
    card_htmls = []

    for model_name, data in nll_data.items():
        # Compute NLL for each test name
        items_html = []
        for item in data["name_nlls"]:
            name = item["name"]
            nll = item["nll"]
            nll_str = f"{nll:.3f}" if nll is not None else "N/A"

            items_html.append(
                f'  <li class="sample-item">'
                f'    <span class="sample-item-text">\'{name}\'</span>'
                f'    <span class="sample-item-nll">{nll_str} nll</span>'
                f"  </li>"
            )

        header_class = (
            "computed"
            if "computed" in model_name.lower() or "trained" in model_name.lower()
            else "uniform"
        )

        card = f"""
        <div class="sample-card">
          <div class="sample-card-header {header_class}">{model_name}</div>
          <div class="sample-meta">
            <span class="sample-meta-item">Dataset NLL: {data["dataset_nll"]:.3f}</span>
          </div>
          <div style="
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 12px 0 8px 0;
            opacity: 0.6;
            font-weight: 700;
          ">
            Name-Level NLL
          </div>
          <ul class="sample-list">
            {"".join(items_html)}
          </ul>
        </div>
        """
        card_htmls.append(card)

    html = f"""{_STYLE_TAG}
    <div class="sample-comparison-container">
      {"".join(card_htmls)}
    </div>
    """
    return html
