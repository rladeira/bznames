"""Visualization tools for bznames models and tokenizers."""

from importlib.resources import files
from typing import Any

from IPython.display import HTML, Markdown, display

from bznames.tokenizer import CharacterEncoder

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

    inputs = (
        input_tokens[:limit].tolist() if hasattr(input_tokens, "tolist") else input_tokens[:limit]
    )
    outputs = (
        output_tokens[:limit].tolist()
        if hasattr(output_tokens, "tolist")
        else output_tokens[:limit]
    )
    frequencies = freqs[:limit].tolist() if hasattr(freqs, "tolist") else freqs[:limit]

    if format_type == "text":
        headers = (
            f"{'Input (Context)':<20} -> {'Output (Target)':<18} | "
            f"{'Bigram':<8} | {'Frequency':<10}"
        )
        print(headers)
        print("-" * 65)
        for inp, out, freq in zip(inputs, outputs, frequencies, strict=True):
            context_str = encoder.decode(inp)
            target_str = encoder.decode_index(out)
            bigram_str = context_str + target_str

            context_repr = f"[{', '.join(map(str, inp))}] ({context_str!r})"
            target_repr = f"{out} ({target_str!r})"

            print(f"{context_repr:<20} -> {target_repr:<18} | {bigram_str!r:<8} | {freq:,}")

    elif format_type == "markdown":
        lines = [
            "| Input (Context) | Output (Target) | Bigram | Frequency |",
            "|:---|:---|:---:|---:|",
        ]
        for inp, out, freq in zip(inputs, outputs, frequencies, strict=True):
            context_str = encoder.decode(inp)
            target_str = encoder.decode_index(out)
            bigram_str = context_str + target_str

            context_repr = f"`[{', '.join(map(str, inp))}]` ({context_str!r})"
            target_repr = f"`{out}` ({target_str!r})"

            lines.append(f"| {context_repr} | {target_repr} | `{bigram_str!r}` | {freq:,} |")

        display(Markdown("\n".join(lines)))

    elif format_type == "html":
        rows = []
        for inp, out, freq in zip(inputs, outputs, frequencies, strict=True):
            context_str = encoder.decode(inp)
            target_str = encoder.decode_index(out)
            bigram_str = context_str + target_str

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
        display(HTML(table_html))


def display_sampled_names(
    model_samples: dict[str, list[str]],
) -> None:
    """Display sampled names comparison in a side-by-side card format.

    Args:
        model_samples: A dictionary mapping model names to lists of generated names.
    """
    card_htmls = []

    for model_name, samples in model_samples.items():
        if not samples:
            continue

        avg_len = sum(len(s) for s in samples) / len(samples)
        unique_chars = len(set("".join(samples)))
        max_len = max(len(s) for s in samples)

        # Determine header class
        header_class = (
            "computed"
            if "computed" in model_name.lower() or "trained" in model_name.lower()
            else "uniform"
        )

        items_html = []
        for s in samples:
            items_html.append(
                f'  <li class="sample-item">'
                f'    <span class="sample-item-text">{s}</span>'
                f'    <span class="sample-item-length">{len(s)} ch</span>'
                f"  </li>"
            )

        card = f"""
        <div class="sample-card">
          <div class="sample-card-header {header_class}">{model_name}</div>
          <div class="sample-meta">
            <span class="sample-meta-item">Avg: {avg_len:.1f} ch</span>
            <span class="sample-meta-item">Max: {max_len} ch</span>
            <span class="sample-meta-item">Unique: {unique_chars} ch</span>
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
    display(HTML(html))
