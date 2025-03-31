import linear_operator_learning.nn.functional as F
import torch


class RegSpectralLoss(torch.nn.Module):
    def __init__(self, reg: float = 1e-5):
        super().__init__()
        self.reg = reg

    def regularization_term(self, inputs, lagged):
        inputs_norm2 = (torch.linalg.matrix_norm(inputs)) ** 2
        lagged_norm2 = (torch.linalg.matrix_norm(lagged)) ** 2
        return self.reg * (inputs_norm2 + lagged_norm2) / 2

    def forward(self, inputs, lagged):
        return self.__call__(inputs, lagged)

    def noreg(self, inputs, lagged):
        return F.l2_contrastive_loss(inputs, lagged)

    def __call__(self, inputs, lagged):
        loss = F.l2_contrastive_loss(inputs, lagged)
        reg = self.regularization_term(inputs, lagged)
        return loss + reg