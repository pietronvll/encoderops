from dataclasses import asdict

import lightning
import numpy as np
import torch
from linear_operator_learning.nn import SimNorm
from loguru import logger
from torch.nn.utils.parametrizations import spectral_norm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs import TrainerArgs
from src.loss import RegSpectralLoss
from src.modules import EMACovariance, EuclideanNorm
from src.utils import effective_rank, lin_svdvals


class EvolutionOperator(lightning.LightningModule):
    def __init__(
        self,
        encoder_cls: torch.nn.Module,
        encoder_args: dict,
        trainer_args: TrainerArgs,
    ):
        super().__init__()
        self.trainer_args = trainer_args
        self.encoder_args = encoder_args
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.forecast = trainer_args.forecast
        if self.forecast:
            d = trainer_args.latent_dim+encoder_args['input_shape']
        else:
            d = trainer_args.latent_dim

        self.encoder = encoder_cls(**encoder_args)
        batch_norm = torch.nn.BatchNorm1d(
            num_features=d, affine=False
        )
        self.covariances = EMACovariance(feature_dim=d)

        if trainer_args.normalize_latents == "simnorm":
            assert trainer_args.simnorm_dim > 0
            simnorm = SimNorm(dim=trainer_args.simnorm_dim)
            self.normalizer = torch.nn.Sequential(batch_norm, simnorm)
        elif trainer_args.normalize_latents == "euclidean":
            euclidnorm = EuclideanNorm()
            self.normalizer = torch.nn.Sequential(batch_norm, euclidnorm)
        else:  # None
            self.normalizer = torch.nn.Sequential(batch_norm)

        self.linear = torch.nn.Linear(d, d, bias=False)

        self._global_step = 0
        self._samples = 0
        if self.trainer_args.normalize_lin:
            self.linear = spectral_norm(self.linear)
        self.loss = RegSpectralLoss(reg=trainer_args.regularization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_nn(x)

    def forward_nn(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        x_enc = self.encoder(x)
        if self.forecast:
            x_enc = torch.cat([x_enc, x], dim=-1)
        x_enc = self.normalizer(x_enc)
        if lagged:
            x_enc = self.linear(x_enc)
        return x_enc

    @torch.no_grad()
    def get_timescales(self):
        transfer_operator = self.get_transfer_operator()
        operator_eigs = torch.linalg.eigvals(transfer_operator).numpy(force=True)
        if hasattr(self.trainer.train_dataloader.dataset, "lagtime_ns"):
            lagtime = self.trainer.train_dataloader.dataset.lagtime_ns
        else:
            lagtime = self.trainer.train_dataloader.dataset.lagtime
        timescales = np.sort((1 / -np.log(np.abs(operator_eigs))) * lagtime)[::-1]
        return timescales

    @torch.no_grad()
    def get_transfer_operator(self, reg: float = 1e-4):
        if self.forecast:
            d = self.trainer_args.latent_dim + self.encoder_args["input_shape"]
        else:
            d = self.trainer_args.latent_dim
        regularizer = reg * torch.eye(d, out=torch.empty_like(self.covariances.cov_X))
        reg_cov = regularizer + self.covariances.cov_X
        transfer_operator = torch.linalg.solve(reg_cov, self.covariances.cov_XY)
        return transfer_operator

    def training_step(self, train_batch, batch_idx):
        x_t, x_lag = self.encoder.prepare_batch(train_batch)
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
        if self.trainer_args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), max_norm=self.trainer_args.max_grad_norm
            )
        # opt:step
        for opt in self.optimizers():
            opt.step()

        # opt:scheduler_step
        if self.trainer_args.min_encoder_lr is not None:
            sch = self.lr_schedulers()
            if self.trainer.is_last_batch:
                sch.step()
            self.log(
                "learning_rate",
                sch.get_last_lr()[0],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=f_t.shape[0],
            )
        # update covariances with EMA
        with torch.no_grad():
            f_lag_nolin = self.forward_nn(x_lag, lagged=False)
            self.covariances(f_t, f_lag_nolin)

        # log
        with torch.no_grad():
            svals = lin_svdvals(self.linear).sort().values
            timescales = self.get_timescales()
            loss_noreg = self.loss.noreg(f_t, f_lag)

        self._global_step += 1
        self._samples += f_t.shape[0]
        loss_dict = {
            "samples": self._samples * self.trainer.world_size,
            "step": self._global_step,
            "train_loss": -loss,
            "train_loss_noreg": -loss_noreg,
            "rank": effective_rank(torch.linalg.eigvalsh(self.covariances.cov_X)),
            "sigma_1": svals[-1].item(),
            "sigma_2": svals[-2].item(),
            "tau_1_ns": timescales[0],
            "tau_2_ns": timescales[1],
        }

        self.log_dict(
            dict(loss_dict),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            prog_bar=True,
            batch_size=f_t.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, x_lag = self.encoder.prepare_batch(batch)
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        loss = self.loss(f_t, f_lag)
        loss_noreg = self.loss.noreg(f_t, f_lag)

        loss_dict = {
            "val_loss": -loss,
            "val_loss_noreg": -loss_noreg,
        }

        self.log_dict(
            dict(loss_dict),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            prog_bar=True,
            batch_size=f_t.shape[0],
        )
        return loss

    def on_train_start(self):
        if self.global_rank == 0:
            logger.info(f"Checkpoints at {self.trainer.checkpoint_callback.dirpath}")
            for k, v in asdict(self.trainer_args).items():
                if k not in self.logger.experiment.config.keys():
                    self.logger.experiment.config[k] = v

    def configure_optimizers(self):
        """
        Initialize the optimizer based on self._optimizer_name and self.optimizer_kwargs.
        """

        encoder_opt = AdamW(
            self.encoder.parameters(),
            lr=self.trainer_args.encoder_lr,
        )
        linear_opt = AdamW(
            self.linear.parameters(), lr=self.trainer_args.linear_lr
        )

        configuration = (
            {
                "optimizer": encoder_opt,
            },
            {
                "optimizer": linear_opt,
            },
        )

        if self.trainer_args.min_encoder_lr is not None:
            scheduler = CosineAnnealingLR(
                encoder_opt,
                T_max=self.trainer_args.epochs,
                eta_min=self.trainer_args.min_encoder_lr,  # Minimum learning rate
            )
            configuration[0]["lr_scheduler"] = scheduler

        return configuration
