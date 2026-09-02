import numpy as np
import pandas as pd


def compute_anomaly_distribution(predictions: np.ndarray | pd.Series) -> dict[str, float]:
    arr = np.asarray(predictions, dtype=int)
    count = int(np.sum(arr == 1))
    rate = float(np.mean(arr == 1) * 100.0)
    return {
        "anomaly_count": count,
        "anomaly_rate_pct": round(rate, 2),
        "total_samples": len(arr),
    }


def compute_pairwise_jaccard_matrix(
    predictions_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    names = list(predictions_dict.keys())
    matrix = np.zeros((len(names), len(names)), dtype=float)

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i == j:
                matrix[i, j] = 1.0
            else:
                p1 = np.asarray(predictions_dict[n1], dtype=bool)
                p2 = np.asarray(predictions_dict[n2], dtype=bool)
                inter = np.sum(p1 & p2)
                union = np.sum(p1 | p2)
                matrix[i, j] = round(float(inter / union) if union > 0 else 0.0, 4)

    return pd.DataFrame(matrix, index=names, columns=names)


def compute_pairwise_overlap_count_matrix(
    predictions_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    names = list(predictions_dict.keys())
    matrix = np.zeros((len(names), len(names)), dtype=int)

    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            p1 = np.asarray(predictions_dict[n1], dtype=bool)
            p2 = np.asarray(predictions_dict[n2], dtype=bool)
            matrix[i, j] = int(np.sum(p1 & p2))

    return pd.DataFrame(matrix, index=names, columns=names)
