import torch
from torch import Tensor, nn
import warnings
from torch.distributions import Bernoulli, kl_divergence
import torch.nn.functional as F


class spars_loss(nn.Module):
    def __init__(self, reduction='sum', eps=1e-8):
        """
        KL divergence between Bernoulli(p) and prior Bernoulli(0).

        Args:
            reduction: 'mean', 'sum', or 'none'
            eps: small value for numerical stability
        """
        super(spars_loss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, p):
        """
        Args:
            p: Tensor of shape (B, S, N, N) - probabilities from the model
        Returns:
            KL loss as a scalar (if reduction is 'mean' or 'sum'), or tensor (if 'none')
        """
        Q_exp = torch.ones_like(p)*1e-6
        M = 0.5 * (p + Q_exp)
        P_dist = Bernoulli(probs=p.clamp(1e-6, 1 - 1e-6))
        Q_dist = Bernoulli(probs=Q_exp.clamp(1e-6, 1 - 1e-6))
        M_dist = Bernoulli(probs=M.clamp(1e-6, 1 - 1e-6))
        kl_pm = kl_divergence(P_dist, M_dist)  # [B, S, N, N]
        kl_qm = kl_divergence(Q_dist, M_dist)  # [B, S, N, N]

        jsd = 0.5 * (kl_pm + kl_qm)
        jsd_reduced = jsd.mean(dim=(0, 1))


        if self.reduction == 'mean':
            return jsd_reduced.mean()
        elif self.reduction == 'sum':
            return jsd_reduced.sum()
        else:  # 'none
            return jsd_reduced



def match_loss(X: torch.Tensor, Y: torch.Tensor):
    x_concat = torch.cat([X, Y], dim=1)
    B, S, N, _ = x_concat.shape
    loss = 0
    for i in range(S):
        for i_1 in range(i+1,S):
            P = x_concat[:,i,:,:]
            Q = x_concat[:,i_1,:,:]
            M = 0.5 * (P + Q)
            P_dist = Bernoulli(probs=P.clamp(1e-6, 1 - 1e-6))
            Q_dist = Bernoulli(probs=Q.clamp(1e-6, 1 - 1e-6))
            M_dist = Bernoulli(probs=M.clamp(1e-6, 1 - 1e-6))
            kl_pm = kl_divergence(P_dist, M_dist)
            kl_qm = kl_divergence(Q_dist, M_dist)
            jsd = 0.5 * (kl_pm + kl_qm)
            jsd = torch.mean(jsd,dim=0).sum()
            loss += jsd
    return loss


def decouple_loss(X, Y):
    B, S, N, _ = Y.shape
    X_flat = X.view(B, -1)
    Y_flat = Y.view(B, S, -1)
    dec_loss = 0
    for s in range(S):
        cos_sim = torch.nn.functional.cosine_similarity(X_flat, Y_flat[:, s, :], dim=-1)  # [B]
        dec_loss += cos_sim.mean()
    return dec_loss




if __name__=="__main__":
    a = torch.tensor([[2.0, 0.7]]*10)
    print(gumbel_softmax(a, tau=10))