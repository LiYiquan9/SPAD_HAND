import torch
import torch.nn.functional as F


@torch.jit.script
def reconst_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "mse",      # "mse" | "mae"
    log: bool = False,      # whether to apply loss function in log space
) -> torch.Tensor:
    """
    Mean squared / absolute error as reconstruction loss.
    c.f. https://arxiv.org/abs/2307.09555
    """

    if log:
        pred, target = torch.log(pred + 1), torch.log(target + 1)
    if mode == "mse":
        return F.mse_loss(pred, target, reduction="sum") / len(pred)
    return F.l1_loss(pred, target, reduction="sum") / len(pred)
    

@torch.jit.script
def eikonal_loss(gradient: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Eikonal loss to regularize SDF.
    c.f. https://arxiv.org/abs/2106.10689
    """
    
    loss = (torch.linalg.vector_norm(gradient, dim=-1) - 1) ** 2
    return (loss * mask).sum() / (mask.sum() + 1e-5)


@torch.jit.script
def tvl1_loss(alpha: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """
    Total variation (TV-L1) prior on piecewise (log-) opacity.
    c.f. https://arxiv.org/abs/1906.07751
    """

    log_alpha = torch.log(eps + alpha)
    diff = torch.abs(log_alpha[..., 1:] - log_alpha[..., :-1])
    tvl1 = torch.mean(torch.sum(diff, dim=-1))
    return tvl1


@torch.jit.script
def beta_loss(opacity: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """
    Beta(0.5, 0.5) prior that encourages ray opacity to be 0 or 1.
    The negative log-likelihood is 0.5 * (log(x) + log(1 - x)).
    c.f. https://arxiv.org/abs/1906.07751
    """

    beta = torch.log(eps + opacity) + torch.log(eps + 1 - opacity)
    return beta.mean() - 2.20727


@torch.jit.script
def entropy_loss(opacity: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """
    Entropy loss that encourages ray opacity to be 0 or 1.
    c.f. https://arxiv.org/abs/2303.12280
    """
    
    opacity = torch.clamp(opacity, min=eps, max=1 - eps)
    return F.binary_cross_entropy(opacity, opacity)