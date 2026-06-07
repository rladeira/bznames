"""Module for sampling and generating names from trained models."""

import numpy as np

from bznames.tokenizer import CharacterEncoder


def sample_name_from_bigram(
    probs: np.ndarray, encoder: CharacterEncoder
) -> str:
    """Sample a name from a bigram probability matrix.

    Args:
        probs: A 2D numpy array representing the conditional bigram probabilities.
        encoder: The CharacterEncoder to encode/decode characters.

    Returns:
        The generated name.
    """
    chars = []
    vocab_size = probs.shape[1]
    i = np.random.choice(vocab_size, p=probs[0])

    while True:
        c = encoder.decode_index(i)

        if c == encoder.special_token:
            break
        else:
            i = np.random.choice(vocab_size, p=probs[i])

        chars.append(c)

    return "".join(chars)
