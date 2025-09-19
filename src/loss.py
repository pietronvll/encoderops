import math
from typing import Literal

import linear_operator_learning.nn.functional as F
import torch
from torch import Tensor
import numpy as np

def joint_LoRA(X: Tensor, Y: Tensor) -> Tensor:
    assert X.shape == Y.shape
    assert X.ndim == 2
    n_modes = X.shape[1]
    vec_mask, mat_mask = get_joint_nesting_masks(
        weights=np.ones(n_modes) / n_modes, 
    )
    corr_term = -2 * (vec_mask.to(X.device) * X * Y).mean(0).sum()
    # \sum_{i=1}^k \sum_{j=1}^k <f_i, f_j> <g_i, g_j>
    # = tr ( cov (f_{1:k}) * cov(g_{1:k}) )
    M_x = compute_second_moment(X)
    M_y = compute_second_moment(Y)
    metric_term = (mat_mask.to(X.device) * M_x * M_y).sum()
    return corr_term + metric_term

def seq_LoRA(X: Tensor, Y: Tensor) -> Tensor:
    assert X.shape == Y.shape
    assert X.ndim == 2
    corr_term = -2 * (X * Y).mean(0).sum()
    # \sum_{i=1}^k \sum_{j=1}^k <f_i, f_j> <g_i, g_j>
    # = tr ( cov (f_{1:k}) * cov(g_{1:k}) )
    M_f = compute_second_moment(X, seq_nesting=True)
    M_g = compute_second_moment(Y, seq_nesting=True)
    metric_term = (M_f * M_g).sum()
    return corr_term + metric_term

def compute_second_moment(
        f: torch.Tensor,
        g: torch.Tensor | None = None,
        seq_nesting: bool = False
    ) -> torch.Tensor:
    """
    compute (optionally sequentially nested) second-moment matrix
        M_ij = <f_i, g_j>
    with partial stop-gradient handling when seq_nesting is True.

    args
    ----
    f : (n, k) tensor
    g : (n, k) tensor or None
    seq_nesting : bool           
    """
    if g is None:
        g = f
    n = f.shape[0]
    if not seq_nesting:
        return (f.T @ g) / n
    else:
        # partial stop gradient
        # lower-triangular: <f_i, sg[g_j]> for i > j
        lower = torch.tril(f.T @ g.detach(), diagonal=-1)
        # upper-triangular: <sg[f_i], g_j> for i < j
        upper = torch.triu(f.detach().T @ g, diagonal=+1)
        # diagonal:         <f_i, g_i>     (no stop-grad)
        diag  = torch.diag((f * g).sum(dim=0))
        return (lower + diag + upper) / n

def get_joint_nesting_masks(weights: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    vector_mask = list(np.cumsum(list(weights)[::-1])[::-1])
    vector_mask = torch.tensor(np.array(vector_mask)).float()
    matrix_mask = torch.minimum(
        vector_mask.unsqueeze(1), vector_mask.unsqueeze(1).T
    ).float()
    return vector_mask, matrix_mask


def DV_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    """KL contrastive loss based on the Donsker-Varadhan lower bound."""
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    linear_term = torch.mean(X * Y) * dim
    sim_mat = torch.matmul(X, Y.T)
    sim_mat_nodiag = torch.triu(sim_mat, diagonal=1) + torch.tril(sim_mat, diagonal=-1)
    log_term = torch.logsumexp(sim_mat_nodiag, dim=(0, 1)) - math.log(npts * (npts - 1))
    return log_term - linear_term


def NWJ_contrastive_loss(X: Tensor, Y: Tensor) -> Tensor:
    """KL contrastive loss based on the Nguyen-Wainwright-Jordan lower bound."""
    assert X.shape == Y.shape
    assert X.ndim == 2

    npts, dim = X.shape
    linear_term = torch.mean(X * Y) * dim
    sim_mat = torch.matmul(X, Y.T) - 1.0
    sim_mat_nodiag = torch.triu(sim_mat, diagonal=1) + torch.tril(sim_mat, diagonal=-1)
    exp_term = sim_mat_nodiag.exp().mean() * npts / (npts - 1)
    return exp_term - linear_term


class Loss(torch.nn.Module):
    def __init__(
        self, reg: float = 1e-5, loss: Literal["kl_DV", "kl_NWJ", "l2", "dpnets", "vampnets"] = "l2"
    ):
        super().__init__()
        self.reg = reg
        self.loss = loss

    def regularization_term(self, inputs, lagged):
        inputs_norm2 = (torch.linalg.matrix_norm(inputs)) ** 2
        lagged_norm2 = (torch.linalg.matrix_norm(lagged)) ** 2
        return self.reg * (inputs_norm2 + lagged_norm2) / 2

    def forward(self, inputs, lagged):
        return self.__call__(inputs, lagged)

    def noreg(self, inputs, lagged):
        if self.loss == "l2":
            return F.l2_contrastive_loss(inputs, lagged)
        elif self.loss == "kl_DV":
            return DV_contrastive_loss(inputs, lagged)
        elif self.loss == "kl_NWJ":
            return NWJ_contrastive_loss(inputs, lagged)
        elif self.loss == "dpnets":
            return F.dp_loss(inputs, lagged, center_covariances=False)
        elif self.loss == "vampnets":
            return F.vamp_loss(inputs, lagged, center_covariances=False)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def __call__(self, inputs, lagged):
        loss = self.noreg(inputs, lagged)
        reg = self.regularization_term(inputs, lagged)
        return loss + reg
