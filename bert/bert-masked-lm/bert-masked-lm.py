import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
        # Copy input
    masked_ids = token_ids.copy()

    # Initialize labels with ignore index
    labels = np.full_like(token_ids, -100)

    # Store original tokens only at masked positions
    labels[mask_positions] = token_ids[mask_positions]

    # 80% -> replace with [MASK]
    mask_token_mask = mask_positions & (replace_probs < 0.8)
    masked_ids[mask_token_mask] = mask_token_id

    random_replace_mask = (
        mask_positions &
        (replace_probs >= 0.8) &
        (replace_probs < 0.9)
    )
    masked_ids[random_replace_mask] = random_tokens[random_replace_mask]
    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        return hidden_states@self.W+self.b
