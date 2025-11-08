import numpy as np
import torch
from torchmetrics.classification import AUROC
import torch.nn.functional as F

def mask_top(matrix,percent):
    flat = matrix.flatten()
    threshold_index = int(len(flat) * percent)
    threshold = np.partition(flat, -threshold_index)[-threshold_index]  # 找到 top 20% 的分界点
    mask = (matrix > threshold).astype(int)
    return mask


def compute_matrix_auroc(pred_matrix: torch.Tensor, label_matrix: torch.Tensor) -> float:
    preds = pred_matrix.flatten()
    targets = label_matrix.flatten()
    auroc_metric = AUROC(task="binary")
    auroc = auroc_metric(preds, targets)
    return auroc.item()

def shd_distance(A: np.ndarray, B: np.ndarray):
    A = A.astype(int)
    B = B.astype(int)
    shd = np.sum(A != B)
    return int(shd)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor,n_nodes) -> float:
    A_flat = A.view(-1)
    B_flat = B.reshape(n_nodes*n_nodes)
    sim = F.cosine_similarity(A_flat.unsqueeze(0), B_flat.unsqueeze(0), dim=1)
    return sim.item()

