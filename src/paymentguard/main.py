import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paymentguard.data.loader import load_dataset
from paymentguard.data.validator import validate_and_clean_data
from paymentguard.evaluation.benchmark import compare_existing_runs
from paymentguard.evaluation.metrics import compute_anomaly_distribution
from paymentguard.models.clustering.multi_view import MultiViewClusteringDetector
from paymentguard.models.ensemble.combiner import MetaEnsembleDetector
from paymentguard.models.isolation_forest.detector import IsolationForestDetector
from paymentguard.models.rule_based.rules import RuleBasedAuditor
from paymentguard.models.semi_supervised.lgbm_classifier import SemiSupervisedLGBMDetector
from paymentguard.models.semi_supervised.triangulator import ConsensusTriangulator
from paymentguard.models.vae.trainer import DeepFinancialVAEDetector
from paymentguard.utils.logger import get_logger

logger = get_logger("PaymentGuard-CLI")


def save_model_artifact(model: any, name: str, models_dir: Path = Path("models")) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"model_{name}.pkl"
    try:
        with open(out_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved trained model artifact to: {out_path}")
    except Exception as e:
        logger.warning(f"Failed to pickle model artifact {name}: {e}")


def load_model_artifact(name: str, models_dir: Path = Path("models")) -> any:
    model_path = models_dir / f"model_{name}.pkl"
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"Loaded existing trained model from: {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Could not load {model_path}: {e}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="PaymentGuard-1M: Anomaly Detection Benchmark CLI")
    parser.add_argument(
        "--data", default="data/transactions.csv", help="Path to input dataset CSV/Parquet"
    )
    parser.add_argument(
        "--model",
        choices=[
            "rule_based",
            "clustering",
            "isolation_forest",
            "vae",
            "semi_supervised",
            "ensemble",
        ],
        default="ensemble",
        help="Anomaly detection model to execute",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Compare existing completed model outputs in reports/",
    )
    parser.add_argument(
        "--sample_frac", type=float, default=None, help="Fraction of dataset to sample (0.0 to 1.0)"
    )
    parser.add_argument("--output", default=None, help="Custom output path for submission CSV")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if model artifact exists",
    )
    args = parser.parse_args()

    if args.benchmark:
        logger.info("Comparing completed model runs in reports/...")
        compare_existing_runs(reports_dir="reports")
        return

    out_file = args.output if args.output is not None else f"reports/submission_{args.model}.csv"
    out_path = Path(out_file)
    logger.info(f"Loading data from: {args.data}...")
    raw_df = load_dataset(args.data, sample_frac=args.sample_frac)
    df = validate_and_clean_data(raw_df)

    t0 = time.perf_counter()

    if args.model == "semi_supervised":
        loaded_model = load_model_artifact(args.model) if not args.force_retrain else None
        if loaded_model is not None:
            model = loaded_model
            preds = model.predict_label(df)
        else:
            base_preds = []
            model_names = []

            report_candidates = [
                ("rule_based", Path("reports/submission_rule_based.csv"), RuleBasedAuditor),
                (
                    "clustering",
                    Path("reports/submission_clustering.csv"),
                    MultiViewClusteringDetector,
                ),
                (
                    "isolation_forest",
                    Path("reports/submission_isolation_forest.csv"),
                    IsolationForestDetector,
                ),
                ("vae", Path("reports/submission_vae.csv"), DeepFinancialVAEDetector),
            ]

            for name, p_path, _detector_cls in report_candidates:
                if p_path.exists():
                    logger.info(f"Reusing existing predictions from {p_path}...")
                    base_preds.append(pd.read_csv(p_path)["is_anomaly"].values)
                    model_names.append(name)

            if len(base_preds) < 2:
                logger.info(
                    "Fewer than 2 base reports found in reports/. Running Rule-Based and Clustering..."
                )
                m1 = RuleBasedAuditor()
                m2 = MultiViewClusteringDetector()
                p1 = m1.fit_predict(df)
                p2 = m2.fit_predict(df)
                base_preds = [p1, p2]
                model_names = ["rule_based", "clustering"]

            min_agree = 2
            logger.info(
                f"Triangulating consensus across {len(base_preds)} models ({', '.join(model_names)}) with min_agreement={min_agree}..."
            )
            triangulator = ConsensusTriangulator(min_agreement=min_agree)
            idx, pseudo_y = triangulator.generate_pseudo_labels(base_preds)
            logger.info(
                f"Generated {len(idx):,} pseudo-labeled samples ({int((pseudo_y == 1).sum()):,} pos, {int((pseudo_y == 0).sum()):,} neg). Training LightGBM..."
            )
            model = SemiSupervisedLGBMDetector()
            model.fit_pseudo(df, idx, pseudo_y)
            save_model_artifact(model, args.model)
            preds = model.predict_label(df)

        elapsed = time.perf_counter() - t0
        dist = compute_anomaly_distribution(preds)
        logger.info(
            f"Predictions: {dist['anomaly_count']:,} anomalies ({dist['anomaly_rate_pct']}%) in {elapsed:.2f}s"
        )
        out_df = df[["order_id"]].copy()
        out_df["is_anomaly"] = preds
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_file, index=False)
        logger.info(f"Saved predictions to: {out_file}")
        return

    elif args.model == "ensemble":
        existing_reports = (
            list(Path("reports").glob("submission_*.csv")) if Path("reports").exists() else []
        )
        if len(existing_reports) >= 2:
            logger.info(f"Combining {len(existing_reports)} existing model runs from reports/...")
            dfs = [
                pd.read_csv(f)["is_anomaly"].values
                for f in existing_reports
                if f.name != "submission_ensemble.csv"
            ]
            if len(dfs) >= 2:
                votes = np.column_stack(dfs)
                preds = (votes.mean(axis=1) >= 0.5).astype(int)
                elapsed = time.perf_counter() - t0
                dist = compute_anomaly_distribution(preds)
                logger.info(
                    f"Ensemble predictions: {dist['anomaly_count']:,} anomalies ({dist['anomaly_rate_pct']}%) in {elapsed:.2f}s"
                )
                out_df = df[["order_id"]].copy()
                out_df["is_anomaly"] = preds
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_df.to_csv(out_file, index=False)
                logger.info(f"Saved predictions to: {out_file}")
                return

        m1 = RuleBasedAuditor()
        m2 = MultiViewClusteringDetector()
        m3 = IsolationForestDetector()
        m4 = DeepFinancialVAEDetector()
        model = MetaEnsembleDetector(detectors=[m1, m2, m3, m4], mode="hard_voting")
        preds = model.fit_predict(df)

    else:
        loaded_model = load_model_artifact(args.model) if not args.force_retrain else None
        if loaded_model is not None:
            model = loaded_model
            preds = model.predict_label(df)
        else:
            if args.model == "rule_based":
                model = RuleBasedAuditor()
            elif args.model == "clustering":
                model = MultiViewClusteringDetector()
            elif args.model == "isolation_forest":
                model = IsolationForestDetector()
            elif args.model == "vae":
                model = DeepFinancialVAEDetector()

            logger.info(f"Fitting model: {args.model}...")
            preds = model.fit_predict(df)
            save_model_artifact(model, args.model)

    elapsed = time.perf_counter() - t0
    dist = compute_anomaly_distribution(preds)
    logger.info(
        f"Predictions complete: {dist['anomaly_count']:,} anomalies ({dist['anomaly_rate_pct']}%) in {elapsed:.2f}s"
    )

    out_df = df[["order_id"]].copy()
    out_df["is_anomaly"] = preds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_file, index=False)
    logger.info(f"Saved predictions to: {out_file}")


if __name__ == "__main__":
    main()
