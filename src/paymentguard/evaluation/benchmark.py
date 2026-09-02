import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from paymentguard.evaluation.metrics import (
    compute_anomaly_distribution,
    compute_pairwise_jaccard_matrix,
    compute_pairwise_overlap_count_matrix,
)
from paymentguard.utils.logger import get_logger

logger = get_logger("Benchmark")


def compare_existing_runs(reports_dir: str | Path = "reports") -> Optional[pd.DataFrame]:
    dir_path = Path(reports_dir)
    if not dir_path.exists():
        logger.warning(f"Reports directory not found: {dir_path}")
        return None

    csv_files = list(dir_path.glob("submission_*.csv")) + list(dir_path.glob("preds_*.csv"))
    if not csv_files:
        logger.warning(f"No completed model submission files found in {dir_path.resolve()}.")
        return None

    predictions: dict[str, np.ndarray] = {}
    benchmark_results: list[dict[str, any]] = []

    for f in sorted(csv_files):
        model_name = f.stem.replace("submission_", "").replace("preds_", "")
        df_pred = pd.read_csv(f)

        if "is_anomaly" not in df_pred.columns:
            continue

        preds = df_pred["is_anomaly"].values.astype(int)
        predictions[model_name] = preds

        dist = compute_anomaly_distribution(preds)
        benchmark_results.append({
            "Paradigm / Model": model_name,
            "Anomaly Count": dist["anomaly_count"],
            "Anomaly Rate (%)": dist["anomaly_rate_pct"],
            "Total Samples": dist["total_samples"],
        })

    if not benchmark_results:
        logger.warning("No valid prediction files with 'is_anomaly' column found.")
        return None

    summary_df = pd.DataFrame(benchmark_results)
    jaccard_df = compute_pairwise_jaccard_matrix(predictions)
    overlap_df = compute_pairwise_overlap_count_matrix(predictions)

    print("\n" + "=" * 75)
    print("                    Anomaly Detection Model Summary")
    print("=" * 75)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("         Pairwise Jaccard Similarity Matrix (Anomaly Set IoU)")
    print("=" * 75)
    print(jaccard_df.round(4).to_string())

    print("\n" + "=" * 75)
    print("             Pairwise Shared Anomaly Counts (Intersection)")
    print("=" * 75)
    formatted_overlap = overlap_df.map(lambda x: f"{x:,}")
    print(formatted_overlap.to_string())
    print("=" * 75 + "\n")

    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Payment Anomaly Detection: Comparison Tool")
    parser.add_argument("--reports-dir", default="reports", help="Directory containing model submission CSVs")
    args = parser.parse_args()

    compare_existing_runs(reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
