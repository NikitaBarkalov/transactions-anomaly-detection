import torch
import torch.nn.functional as F


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = F.huber_loss(recon, x, reduction="mean", delta=1.0)

    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)

    kl_loss = torch.clamp(kl_per_dim, min=free_bits).mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
