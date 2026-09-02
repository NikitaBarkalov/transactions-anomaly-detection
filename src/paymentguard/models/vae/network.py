from dataclasses import dataclass, field

import torch
import torch.nn as nn

DEFAULT_INPUT_DIM = 40
DEFAULT_HIDDEN_DIMS = [256, 128, 64]
DEFAULT_LATENT_DIM = 16
DEFAULT_DROPOUT = 0.2
DEFAULT_CLIP_PERCENTILE = 99.9
DEFAULT_FREE_BITS = 0.1
DEFAULT_BETA_START = 0.0
DEFAULT_BETA_END = 0.01
DEFAULT_BETA_WARMUP_EPOCHS = 5
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SEED = 42


@dataclass
class VAEConfig:
    input_dim: int = DEFAULT_INPUT_DIM
    hidden_dims: list[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_DIMS))
    latent_dim: int = DEFAULT_LATENT_DIM
    dropout: float = DEFAULT_DROPOUT
    clip_percentile: float = DEFAULT_CLIP_PERCENTILE
    free_bits: float = DEFAULT_FREE_BITS
    beta_start: float = DEFAULT_BETA_START
    beta_end: float = DEFAULT_BETA_END
    beta_warmup_epochs: int = DEFAULT_BETA_WARMUP_EPOCHS
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_clip: float = DEFAULT_GRAD_CLIP
    seed: int = DEFAULT_SEED


class _VAEBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinancialVAE(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        enc_layers = []
        in_dim = cfg.input_dim
        for h_dim in cfg.hidden_dims:
            enc_layers.append(_VAEBlock(in_dim, h_dim, cfg.dropout))
            in_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dims[-1], cfg.latent_dim)

        dec_layers = []
        in_dim = cfg.latent_dim
        for h_dim in reversed(cfg.hidden_dims):
            dec_layers.append(_VAEBlock(in_dim, h_dim, cfg.dropout))
            in_dim = h_dim
        dec_layers.append(nn.Linear(cfg.hidden_dims[0], cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=4.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
