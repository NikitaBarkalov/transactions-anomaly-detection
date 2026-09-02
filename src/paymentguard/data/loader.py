import argparse
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from paymentguard.utils.logger import get_logger

logger = get_logger("DataLoader")

DEFAULT_HF_REPO = "nektonekks/transactions_dataset"
POSSIBLE_FILENAMES = [
    "transactions.csv",
    "hackathon_int20h_dataset_test.csv",
    "transactions.parquet",
    "data.csv",
]


def download_dataset(
    repo_id: str = DEFAULT_HF_REPO,
    target_path: str | Path = "data/transactions.csv",
    force: bool = False,
) -> Path:
    out_path = Path(target_path)
    if out_path.exists() and not force:
        logger.info(f"Dataset already exists at {out_path.resolve()}.")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = False

    for fname in POSSIBLE_FILENAMES:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{fname}"
        logger.info(f"Attempting download from: {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaymentGuard-Downloader"})
            with urllib.request.urlopen(req) as response, open(out_path, "wb") as out_file:
                total_size = int(response.info().get("Content-Length", 0))
                downloaded_bytes = 0
                chunk_size = 1024 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    if total_size > 0:
                        pct = (downloaded_bytes / total_size) * 100
                        print(
                            f"\rDownloading dataset: {downloaded_bytes / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                print()
            downloaded = True
            logger.info(f"Successfully downloaded dataset to {out_path.resolve()}")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise

    if not downloaded:
        raise RuntimeError(f"Could not find dataset files at Hugging Face repo: {repo_id}")

    return out_path


def load_dataset(
    file_path: str | Path = "data/transactions.csv",
    nrows: int | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
    auto_download: bool = True,
) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        if auto_download and path.name in ("transactions.csv", "transactions.parquet"):
            logger.warning(
                f"File not found at {path}. Initiating automatic download from Hugging Face..."
            )
            download_dataset(target_path=path)
        else:
            raise FileNotFoundError(f"Dataset not found at path: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        if nrows is not None:
            df = df.iloc[:nrows]
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="PaymentGuard Dataset Downloader")
    parser.add_argument(
        "--repo", default=DEFAULT_HF_REPO, help="Hugging Face dataset repository ID"
    )
    parser.add_argument("--output", default="data/transactions.csv", help="Target output file path")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download if file already exists"
    )
    args = parser.parse_args()

    download_dataset(repo_id=args.repo, target_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
