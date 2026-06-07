"""Visualization tools for bznames models and tokenizers."""

from typing import Any

from IPython.display import HTML, Markdown, display

from bznames.tokenizer import CharacterEncoder

_TABLE_CSS = """<style>
.bznames-table {
  border-collapse: collapse;
  width: 100%;
  max-width: 800px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  overflow: hidden;
  margin: 16px 0;
}
.bznames-table th {
  background-color: rgba(148, 163, 184, 0.08);
  border-bottom: 2px solid rgba(148, 163, 184, 0.2);
  text-align: left;
  padding: 12px 16px;
  font-size: 0.875rem;
  font-weight: 600;
  color: inherit;
}
.bznames-table td {
  padding: 12px 16px;
  font-size: 0.875rem;
  border-bottom: 1px solid rgba(148, 163, 184, 0.15);
  color: inherit;
}
.bznames-table tr:hover {
  background-color: rgba(148, 163, 184, 0.05);
}
.bznames-code-token {
  background-color: rgba(148, 163, 184, 0.15);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.8125rem;
  color: inherit;
}
.bznames-char-lbl {
  opacity: 0.7;
  font-size: 0.75rem;
  margin-left: 8px;
  font-family: monospace;
}
.bznames-bigram-lbl {
  background-color: rgba(59, 130, 246, 0.15);
  color: #3b82f6;
  padding: 4px 8px;
  border-radius: 6px;
  font-family: monospace;
  font-weight: 600;
  font-size: 0.875rem;
}
.bznames-freq-lbl {
  font-family: monospace;
  color: inherit;
  text-align: right;
}
</style>
"""


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

        table_html = f"""{_TABLE_CSS}
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
