"""Single-layer linear model that learns the bigram distribution via gradient descent.

The model is a single weight matrix ``W`` of shape ``(vocab_size, vocab_size)``:
each input character is one-hot encoded and projected through ``W`` to logits, and
a softmax turns those logits into next-character probabilities. Training it to
minimise the weighted negative log-likelihood recovers the frequentist bigram model.
"""

import torch
import torch.nn.functional as F

from bznames.tokenizer import CharacterEncoder


def _linear_forward(input_tokens: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """One-hot the input tokens and project them through ``W`` (a single linear layer).

    Args:
        input_tokens: 1D tensor of input token indices of shape (N,).
        W: Weight matrix of shape (vocab_size, vocab_size).

    Returns:
        Logits of shape (N, vocab_size).
    """
    vocab_size = W.shape[0]
    X = F.one_hot(input_tokens, num_classes=vocab_size).float()
    return X @ W


def _target_log_probs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-probabilities the model assigns to each target token.

    Args:
        logits: Logits of shape (N, vocab_size).
        targets: 1D tensor of target token indices of shape (N,).

    Returns:
        1D tensor of shape (N,) with the log-probability of each target.
    """
    log_probs = logits.log_softmax(dim=1)  # stable: subtracts row max internally
    rows = torch.arange(len(targets))
    return log_probs[rows, targets]


def compute_linear_nn_nll_for_tokens(
    W: torch.Tensor,
    input_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the weighted negative log-likelihood of the tokens under ``W``.

    Args:
        W: Weight matrix of shape (vocab_size, vocab_size).
        input_tokens: Input token indices of shape (N,) or (N, 1).
        output_tokens: Target token indices of shape (N,).
        weights: Per-example weights (frequencies or normalized probabilities) of shape (N,).

    Returns:
        The negative log-likelihood as a scalar tensor. A tensor (not a float) is
        returned so callers can backpropagate through it during training.

    Raises:
        ValueError: If input_tokens is 2D but the second dimension is not 1.
    """
    # Auto-squeeze input_tokens if it has shape (N, 1)
    if input_tokens.ndim == 2:
        if input_tokens.shape[1] != 1:
            raise ValueError(
                f"input_tokens must be 1D or have shape (N, 1), got shape {input_tokens.shape}"
            )
        input_tokens = input_tokens.squeeze(1)

    assert input_tokens.shape == output_tokens.shape == weights.shape, (
        f"Shape mismatch: input_tokens {input_tokens.shape}, "
        f"output_tokens {output_tokens.shape}, weights {weights.shape}"
    )

    logits = _linear_forward(input_tokens, W)
    return -torch.dot(weights, _target_log_probs(logits, output_tokens))


def compute_linear_nn_nll_for_name(name: str, W: torch.Tensor, encoder: CharacterEncoder) -> float:
    """Compute the mean negative log-likelihood of a single name under ``W``.

    Args:
        name: The name string.
        W: Weight matrix of shape (vocab_size, vocab_size).
        encoder: The CharacterEncoder used to map characters to indices.

    Returns:
        The mean negative log-likelihood over the transitions in the name.
    """
    # Pad the name with the special token (e.g. ".") at both ends, then split each
    # adjacent (input -> output) character transition into its own example.
    special_token = encoder.special_token
    encoded = encoder.encode(special_token + name + special_token)
    input_tokens = torch.tensor(encoded[:-1])
    output_tokens = torch.tensor(encoded[1:])

    logits = _linear_forward(input_tokens, W)
    nll = -1 * _target_log_probs(logits, output_tokens).mean()

    return nll.item()
