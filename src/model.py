from dataclasses import asdict
from typing import List

import lightning
import numpy as np
import torch
from linear_operator_learning.nn import SimNorm
from linear_operator_learning.nn.stats import covariance
from loguru import logger
from mlcolvar.core.nn.graph.schnet import SchNetModel
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs import DataArgs, ModelArgs
from src.loss import RegSpectralLoss
from src.utils import effective_rank, lin_svdvals


class EvolutionOperator(lightning.LightningModule):
    def __init__(
        self,
        cutoff: float,
        atomic_numbers: List[int],
        model_args: ModelArgs,
        data_args: DataArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_args = model_args
        self.data_args = data_args
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

        batch_norm = torch.nn.BatchNorm1d(
            num_features=model_args.latent_dim, affine=False
        )

        if model_args.simnorm_dim > 0:
            simnorm = SimNorm(dim=model_args.simnorm_dim)
            self.encoder = torch.nn.Sequential(encoder, batch_norm, simnorm)
        else:
            self.encoder = torch.nn.Sequential(encoder, batch_norm)

        # Register buffers for covariance and cross-covariance matrices
        self.register_buffer("cov", torch.eye(model_args.latent_dim))
        self.register_buffer("cross_cov", torch.eye(model_args.latent_dim))

        self.linear = torch.nn.Linear(
            model_args.latent_dim, model_args.latent_dim, bias=False
        )
        if self.model_args.normalize_lin:
            self.linear = spectral_norm(self.linear)

        self.loss = RegSpectralLoss(reg=model_args.regularization)

    def forward_nn(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if lagged:
            x = self.linear(x)
        return x

    @torch.no_grad()
    def get_timescales(self):
        """
        Create a Wandb scatter plot of the eigenvalues of the transfer operator as currently estimated by self.cov and self.cross_cov
        """
        transfer_operator = self.get_transfer_operator()
        operator_eigs = torch.linalg.eigvals(transfer_operator).numpy(force=True)
        lagtime_ns = self.trainer.train_dataloader.dataset.lagtime_ns
        timescales = np.sort((1 / -np.log(np.abs(operator_eigs))) * lagtime_ns)[::-1]
        return timescales

    @torch.no_grad()
    def get_transfer_operator(self, reg: float = 1e-4):
        reg_cov = (
            reg
            * torch.eye(
                self.model_args.latent_dim, dtype=self.cov.dtype, device=self.cov.device
            )
            + self.cov
        )
        transfer_operator = torch.linalg.solve(reg_cov, self.cross_cov)
        return transfer_operator

    def training_step(self, train_batch, batch_idx):
        # data
        x_t = self._setup_graph_data(train_batch)
        x_lag = self._setup_graph_data(train_batch, key="item_lag")
        # forward
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        # opt
        # opt:zero_grad
        for opt in self.optimizers():
            opt.zero_grad()
        # opt:loss
        loss = self.loss(f_t, f_lag)
        # opt:backard
        self.manual_backward(loss)
        # opt:grad_clip
        if self.model_args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.model_args.max_grad_norm
            )
        # opt:step
        for opt in self.optimizers():
            opt.step()

        # opt:scheduler_step
        if self.model_args.min_encoder_lr is not None:
            sch = self.lr_schedulers()
            if self.trainer.is_last_batch:
                sch.step()
            self.log(
                "learning_rate",
                sch.get_last_lr()[0],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=f_t.shape[0],
            )
        # update covariances with EMA

        with torch.no_grad():
            cov_new = covariance(f_t, center=False)
            f_lag_nolin = self.forward_nn(x_lag, lagged=False)
            cross_cov_new = covariance(f_t, f_lag_nolin, center=False)
            # Gather values from all processes
            if self.trainer.world_size > 1:
                # Use all_reduce with AVG operation to average across all devices
                torch.distributed.all_reduce(cov_new, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(
                    cross_cov_new, op=torch.distributed.ReduceOp.SUM
                )

                # Normalize by world_size since we summed across devices
                cov_new = cov_new / self.trainer.world_size
                cross_cov_new = cross_cov_new / self.trainer.world_size
            alpha = 0.99
            self.cov = alpha * self.cov + (1 - alpha) * cov_new
            self.cross_cov = alpha * self.cross_cov + (1 - alpha) * cross_cov_new

        # log
        with torch.no_grad():
            svals = lin_svdvals(self.linear).sort().values
            timescales = self.get_timescales()
            loss_noreg = self.loss.noreg(f_t, f_lag)

        loss_dict = {
            "train_loss": -loss,
            "train_loss_noreg": -loss_noreg,
            "effective_rank": effective_rank(svals),
            "sigma_1": svals[-1].item(),
            "sigma_2": svals[-2].item(),
            "timescale_1 (ns)": timescales[0],
            "timescale_2 (ns)": timescales[1],
            "timescale_3 (ns)": timescales[2],
        }

        self.log_dict(
            dict(loss_dict),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            prog_bar=True,
            batch_size=f_t.shape[0],
        )
        return loss

    def on_train_start(self):
        if self.global_rank == 0:
            logger.info(f"Checkpoints at {self.trainer.checkpoint_callback.dirpath}")
            for k, v in asdict(self.model_args).items():
                if k not in self.logger.experiment.config.keys():
                    self.logger.experiment.config[k] = v

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.
        """

        encoder_opt = Adam(
            self.encoder.parameters(),
            lr=self.model_args.encoder_lr,
        )
        linear_opt = Adam(self.linear.parameters(), lr=self.model_args.linear_lr)

        configuration = (
            {
                "optimizer": encoder_opt,
            },
            {
                "optimizer": linear_opt,
            },
        )

        if self.model_args.min_encoder_lr is not None:
            scheduler = CosineAnnealingLR(
                encoder_opt,
                T_max=self.model_args.epochs,
                eta_min=self.model_args.min_encoder_lr,  # Minimum learning rate
            )
            configuration[0]["lr_scheduler"] = scheduler

        return configuration

    @staticmethod
    def _setup_graph_data(train_batch, key: str = "item"):
        data = train_batch[key]
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
        return data
