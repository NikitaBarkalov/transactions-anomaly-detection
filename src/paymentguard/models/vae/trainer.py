from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from paymentguard.core.base_detector import BaseAnomalyDetector
from paymentguard.features.pipeline import FeaturePipeline
from paymentguard.models.vae.network import (
    FinancialVAE,
    VAEConfig,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LATENT_DIM,
    DEFAULT_DROPOUT,
    DEFAULT_CLIP_PERCENTILE,
    DEFAULT_FREE_BITS,
    DEFAULT_BETA_START,
    DEFAULT_BETA_END,
    DEFAULT_BETA_WARMUP_EPOCHS,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_GRAD_CLIP,
    DEFAULT_SEED,
)
from paymentguard.models.vae.loss import vae_loss
from paymentguard.models.vae.clipper import PerFeatureClipper


from paymentguard.utils.logger import get_logger

logger = get_logger("DeepFinancialVAE")


class DeepFinancialVAEDetector(BaseAnomalyDetector):
    def __init__(
        self,
        hidden_dims: Optional[list[int]] = None,
        latent_dim: int = DEFAULT_LATENT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        clip_percentile: float = DEFAULT_CLIP_PERCENTILE,
        free_bits: float = DEFAULT_FREE_BITS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
        beta_warmup_epochs: int = DEFAULT_BETA_WARMUP_EPOCHS,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        grad_clip: float = DEFAULT_GRAD_CLIP,
        seed: int = DEFAULT_SEED,
        device: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name="DeepFinancialVAEDetector", config=config or {})
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vae_cfg = VAEConfig(
            hidden_dims=hidden_dims or list(DEFAULT_HIDDEN_DIMS),
            latent_dim=latent_dim,
            dropout=dropout,
            clip_percentile=clip_percentile,
            free_bits=free_bits,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_warmup_epochs=beta_warmup_epochs,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            seed=seed,
        )
        self.model: Optional[FinancialVAE] = None
        self.clipper = PerFeatureClipper(percentile=self.vae_cfg.clip_percentile)
        self.feature_pipeline = FeaturePipeline(use_user_agg=True, use_sequential=True)
        self.score_threshold_: float = 0.0
        self.train_scores_: Optional[np.ndarray] = None

    def _predict_score_from_matrix(self, X_clipped: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        dataset = TensorDataset(torch.from_numpy(X_clipped.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.vae_cfg.batch_size * 4, shuffle=False)

        self.model.eval()
        recon_errors = []

        with torch.no_grad():
            for batch in loader:
                x_b = batch[0].to(self.device)
                recon, _, _ = self.model(x_b)
                mse = ((recon - x_b) ** 2).mean(dim=1).cpu().numpy()
                recon_errors.append(mse)

        return np.concatenate(recon_errors)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> "DeepFinancialVAEDetector":
        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.fit_transform(X)
        else:
            X_matrix = X

        X_clipped = self.clipper.fit_transform(X_matrix)

        self.vae_cfg.input_dim = X_clipped.shape[1]
        self.model = FinancialVAE(self.vae_cfg).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_clipped.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=self.vae_cfg.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.vae_cfg.learning_rate, weight_decay=self.vae_cfg.weight_decay)

        self.model.train()
        epochs = self.vae_cfg.epochs
        logger.info(f"Training VAE on device [{self.device}] for {epochs} epochs (Input Dim: {self.vae_cfg.input_dim}, Latent Dim: {self.vae_cfg.latent_dim}, Batch Size: {self.vae_cfg.batch_size})...")

        for epoch in range(1, epochs + 1):
            beta = min(1.0, epoch / max(1, self.vae_cfg.beta_warmup_epochs)) * self.vae_cfg.beta_end
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            n_batches = 0

            for batch in loader:
                x_b = batch[0].to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(x_b)
                loss, recon_l, kl_l = vae_loss(recon, x_b, mu, logvar, beta, self.vae_cfg.free_bits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.vae_cfg.grad_clip)
                optimizer.step()

                total_loss += float(loss.item())
                total_recon += float(recon_l.item())
                total_kl += float(kl_l.item())
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            avg_recon = total_recon / max(1, n_batches)
            avg_kl = total_kl / max(1, n_batches)
            logger.info(f"  Epoch [{epoch:02d}/{epochs:02d}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {beta:.4f}")

        self.is_fitted = True

        self.train_scores_ = self._predict_score_from_matrix(X_clipped)
        log_scores = np.log(self.train_scores_ + 1e-9)
        mu_log = float(np.mean(log_scores))
        sigma_log = float(np.std(log_scores))
        self.score_threshold_ = float(np.exp(mu_log + 1.50 * sigma_log))
        logger.info(f"VAE training complete. Adaptive log-normal 1.50-sigma anomaly threshold: {self.score_threshold_:.4f}")
        return self

    def predict_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict")

        if isinstance(X, pd.DataFrame):
            _, X_matrix = self.feature_pipeline.transform(X)
        else:
            X_matrix = X

        X_clipped = self.clipper.transform(X_matrix)
        return self._predict_score_from_matrix(X_clipped)

    def fit_predict(self, X: pd.DataFrame | np.ndarray, y: Optional[pd.Series | np.ndarray] = None) -> np.ndarray:
        self.fit(X, y)
        if self.train_scores_ is not None:
            return (self.train_scores_ >= self.score_threshold_).astype(int)
        return self.predict_label(X)

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        scores = self.predict_score(X)
        thr = threshold if threshold is not None else self.score_threshold_
        return (scores >= thr).astype(int)

    def save_checkpoint(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.model.state_dict() if self.model is not None else None,
            "vae_cfg": self.vae_cfg,
            "clipper": self.clipper,
            "feature_pipeline": self.feature_pipeline,
            "score_threshold_": self.score_threshold_,
            "is_fitted": self.is_fitted,
        }
        torch.save(checkpoint, out_path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: Optional[str] = None) -> "DeepFinancialVAEDetector":
        in_path = Path(path)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(in_path, map_location=dev)
        detector = cls(device=dev)
        detector.vae_cfg = checkpoint["vae_cfg"]
        detector.clipper = checkpoint["clipper"]
        detector.feature_pipeline = checkpoint["feature_pipeline"]
        detector.score_threshold_ = checkpoint["score_threshold_"]
        detector.is_fitted = checkpoint["is_fitted"]
        if checkpoint["state_dict"] is not None:
            detector.model = FinancialVAE(detector.vae_cfg).to(detector.device)
            detector.model.load_state_dict(checkpoint["state_dict"])
            detector.model.eval()
        return detector
