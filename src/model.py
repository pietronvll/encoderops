from typing import List

import lightning
import linear_operator_learning as lol
import torch
from mlcolvar.core.nn.graph.schnet import SchNetModel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.configs import ModelArgs
from src.loss import RegSpectralLoss
from src.utils import normalize_linear_layer


class EvolutionOperator(lightning.LightningModule):
    def __init__(self, cutoff: float, atomic_numbers: List[int], model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args
        self.automatic_optimization = False

        encoder = SchNetModel(
            n_out=model_args.latent_dim,
            cutoff=cutoff,
            atomic_numbers=atomic_numbers,
            n_bases=model_args.n_bases,
            n_layers=model_args.n_layers,
            n_filters=model_args.n_filters,
            n_hidden_channels=model_args.n_hidden_channels,
        )
        if model_args.simnorm_dim > 0:
            simnorm = lol.nn.SimNorm(dim=model_args.simnorm_dim)
            self.encoder = torch.nn.Sequential(encoder, simnorm)
        else:
            self.encoder = encoder

        self.linear = torch.nn.Linear(
            model_args.latent_dim,
            model_args.latent_dim,
            bias=False,
        )
        self.loss = RegSpectralLoss(reg=model_args.regularization)

    def forward_nn(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if lagged:
            x = self.linear(x)
        return x

    def training_step(self, train_batch, batch_idx):
        # =================get data===================
        x_t = self._setup_graph_data(train_batch, key="data_list")
        x_lag = self._setup_graph_data(train_batch, key="data_list_lag")

        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        # optimization
        for opt in self.optimizers():
            opt.zero_grad()
        # ===================loss=====================
        loss = self.loss(f_t, f_lag)
        self.manual_backward(loss)
        if self.model_args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.model_args.max_grad_norm
            )
        for opt in self.optimizers():
            opt.step()
        # single scheduler
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step()
        # ====================log=====================
        with torch.no_grad():
            loss_noreg = self.loss.noreg(f_t, f_lag)
        loss_dict = {
            "train_loss": -loss,
            "train_loss_noreg": -loss_noreg,
        }
        self.log_dict(
            dict(loss_dict), on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "learning_rate",
            sch.get_last_lr()[0],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # =================get data===================
        x_t = self._setup_graph_data(batch, key="data_list")
        x_lag = self._setup_graph_data(batch, key="data_list_lag")

        # =================forward====================
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        # ===================loss=====================
        loss = self.loss(f_t, f_lag)
        loss_noreg = self.loss.noreg(f_t, f_lag)

        # Compute effective rank
        T = self.linear.weight
        # Compute the effective rank of T, which is the exponential of the entropy of the singular values
        s = torch.linalg.svdvals(T)
        s /= s.sum()
        effective_rank = torch.exp(-torch.sum(s * torch.log(s)))
        loss_dict = {
            "valid_loss": -loss,
            "valid_loss_noreg": -loss_noreg,
            "effective_rank": effective_rank,
        }
        self.log_dict(loss_dict, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_after_backward(self):
        if self.model_args.normalize_lin:
            normalize_linear_layer(self.linear)

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.

        Returns
        -------
        torch.optim
            Torch optimizer
        """

        encoder_opt = AdamW(
            self.encoder.parameters(),
            lr=self.model_args.encoder_lr,
        )
        linear_opt = Adam(self.linear.parameters(), lr=self.model_args.linear_lr)
        warmup_sch = LinearLR(
            encoder_opt,
        )

        total_epochs = self.model_args.epochs
        warmup_epochs = int(0.1 * total_epochs)  # 10% of total epochs for warmup

        warmup_sch = LinearLR(
            encoder_opt,
            start_factor=1e-5 / self.model_args.encoder_lr,  # Scale from 1e-5
            end_factor=1.0,  # To initial LR
            total_iters=warmup_epochs,
        )

        # Cosine annealing for remaining epochs
        cos_sch = CosineAnnealingLR(
            encoder_opt,
            T_max=total_epochs - warmup_epochs,  # Remaining 90% of epochs
            eta_min=self.model_args.min_encoder_lr,  # Minimum learning rate
        )
        # Sequential scheduler that first runs warmup, then cosine decay
        scheduler = SequentialLR(
            encoder_opt,
            schedulers=[warmup_sch, cos_sch],
            milestones=[warmup_epochs],  # Switch to cosine after warmup completes
        )

        return (
            {
                "optimizer": encoder_opt,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            },
            {"optimizer": linear_opt},
        )

    @staticmethod
    def _setup_graph_data(train_batch, key: str = "data_list"):
        data = train_batch[key]
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        return data
