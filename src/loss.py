import math
from typing import Literal

import linear_operator_learning.nn.functional as F
import torch
from torch import Tensor


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
        self, reg: float = 1e-5, loss: Literal["kl_DV", "kl_NWJ", "l2"] = "l2"
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
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def __call__(self, inputs, lagged):
        loss = self.noreg(inputs, lagged)
        reg = self.regularization_term(inputs, lagged)
        return loss + reg
