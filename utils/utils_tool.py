import torch
from torch import Tensor, nn
import numpy as np
import warnings
from torch.distributions import Bernoulli, kl_divergence
import torch.nn.functional as F


def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


class spars_loss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8, mode='kl'):
        super(spars_loss, self).__init__()
        assert mode in ['kl', 'js'], "mode must be 'kl' or 'js'"
        self.reduction = reduction
        self.eps = eps
        self.mode = mode

    def forward(self, p):
        """
        Args:
            p (Tensor): shape (B, S, N, N), Bernoulli probabilities from the model
        Returns:
            Divergence loss (scalar or tensor depending on reduction)
        """
        # prior Bernoulli close to 0 (sparse prior)
        Q_exp = torch.ones_like(p) * 1e-6

        # Clamp probabilities for stability
        p = p.clamp(self.eps, 1 - self.eps)
        Q_exp = Q_exp.clamp(self.eps, 1 - self.eps)

        # Define distributions
        P_dist = Bernoulli(probs=p)
        Q_dist = Bernoulli(probs=Q_exp)

        if self.mode == 'kl':
            # KL(P || Q)
            loss = kl_divergence(P_dist, Q_dist)
        else:
            # Jensen–Shannon Divergence
            M = 0.5 * (p + Q_exp)
            M_dist = Bernoulli(probs=M)
            kl_pm = kl_divergence(P_dist, M_dist)
            kl_qm = kl_divergence(Q_dist, M_dist)
            loss = 0.5 * (kl_pm + kl_qm)

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def consistency_loss(
    P: torch.Tensor,
    Q: torch.Tensor,
    reduction: str = 'mean',
    mask: torch.Tensor = None,
    final_reduction: str = 'mean',
    mode: str = 'js'
) -> torch.Tensor:
    """
    Compute divergence-based matching loss (KL or Jensen–Shannon)
    between Bernoulli distributions P and Q.

    Args:
        P (torch.Tensor): [B, S, N, N], predicted Bernoulli probabilities
        Q (torch.Tensor): [B, 1, N, N], reference Bernoulli probabilities
        reduction (str): 'sum' | 'mean' | 'none', reduction over N×N
        mask (torch.Tensor, optional): [N, N] or [B, S, N, N], element-wise mask
        final_reduction (str): 'mean' | 'sum' | 'none', reduction over B and S
        mode (str): 'kl' for Kullback–Leibler divergence,
                    'js' for Jensen–Shannon divergence

    Returns:
        torch.Tensor: scalar or tensor loss depending on reductions
    """
    assert P.shape[0] == Q.shape[0] and P.shape[2:] == Q.shape[2:], "Shape mismatch"
    assert mode in ['kl', 'js'], "mode must be 'kl' or 'js'"

    B, S, N, _ = P.shape
    Q_exp = Q.expand(-1, S, -1, -1)

    # Clamp for numerical stability
    P = P.clamp(1e-6, 1 - 1e-6)
    Q_exp = Q_exp.clamp(1e-6, 1 - 1e-6)

    # Define distributions
    P_dist = Bernoulli(probs=P)
    Q_dist = Bernoulli(probs=Q_exp)

    # --- Compute divergence ---
    if mode == 'kl':
        # KL(P || Q)
        div = kl_divergence(P_dist, Q_dist)  # [B, S, N, N]
    else:
        # Jensen–Shannon divergence
        M = 0.5 * (P + Q_exp)
        M_dist = Bernoulli(probs=M)
        kl_pm = kl_divergence(P_dist, M_dist)
        kl_qm = kl_divergence(Q_dist, M_dist)
        div = 0.5 * (kl_pm + kl_qm)

    # --- Apply mask if provided ---
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        elif mask.shape != div.shape:
            raise ValueError("Mask shape must match [B, S, N, N]")
        div = div * mask

    # --- Reduction over N×N ---
    if reduction == 'sum':
        div_reduced = div.sum(dim=(-2, -1))  # [B, S]
    elif reduction == 'mean':
        if mask is not None:
            denom = mask.sum(dim=(-2, -1)).clamp(min=1e-6)
            div_reduced = div.sum(dim=(-2, -1)) / denom
        else:
            div_reduced = div.mean(dim=(-2, -1))  # [B, S]
    elif reduction == 'none':
        div_reduced = div  # [B, S, N, N]
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

    # --- Final reduction over B, S ---
    if final_reduction == 'mean':
        return div_reduced.mean()
    elif final_reduction == 'sum':
        return div_reduced.sum()
    elif final_reduction == 'none':
        return div_reduced
    else:
        raise ValueError(f"Invalid final_reduction: {final_reduction}")