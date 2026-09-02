from typing import Optional, Sequence
import numpy as np
import pandas as pd

DEFAULT_MIN_AGREEMENT = 2
DEFAULT_NEGATIVE_DOWNSAMPLE_RATIO = 0.30
DEFAULT_RANDOM_STATE = 42


class ConsensusTriangulator:
    def __init__(
        self,
        min_agreement: int = DEFAULT_MIN_AGREEMENT,
        negative_downsample_ratio: float = DEFAULT_NEGATIVE_DOWNSAMPLE_RATIO,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.min_agreement = min_agreement
        self.negative_downsample_ratio = negative_downsample_ratio
        self.random_state = random_state

    def generate_pseudo_labels(
        self,
        predictions: Sequence[np.ndarray | pd.Series],
        negative_downsample_ratio: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        downsample_ratio = negative_downsample_ratio if negative_downsample_ratio is not None else self.negative_downsample_ratio
        seed = random_state if random_state is not None else self.random_state

        pred_matrix = np.column_stack([np.asarray(p, dtype=int) for p in predictions])
        agreement_count = pred_matrix.sum(axis=1)

        positive_mask = agreement_count >= self.min_agreement
        pos_indices = np.where(positive_mask)[0]

        negative_mask = agreement_count == 0
        neg_indices = np.where(negative_mask)[0]

        rng = np.random.default_rng(seed)
        n_neg = int(len(neg_indices) * downsample_ratio)
        sampled_neg_indices = rng.choice(neg_indices, size=n_neg, replace=False)

        selected_indices = np.concatenate([pos_indices, sampled_neg_indices])
        targets = np.concatenate([np.ones(len(pos_indices), dtype=int), np.zeros(len(sampled_neg_indices), dtype=int)])

        shuffle_perm = rng.permutation(len(selected_indices))
        return selected_indices[shuffle_perm], targets[shuffle_perm]
