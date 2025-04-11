from pathlib import Path
from typing import List

import lightning
import torch
from linear_operator_learning.nn import SimNorm
from loguru import logger
from mlcolvar.core.nn.graph.schnet import SchNetModel
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs import DataArgs, ModelArgs
from src.loss import RegSpectralLoss


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

        linear_l = torch.nn.Linear(
            model_args.latent_dim,
            model_args.linear_lora,
            bias=False,
        )
        linear_r = torch.nn.Linear(
            model_args.linear_lora,
            model_args.latent_dim,
            bias=False,
        )
        self.linear = torch.nn.Sequential(linear_l, linear_r)

        self.loss = RegSpectralLoss(reg=model_args.regularization)
        self.embeddings = {"t": [], "lag": []}

    def effective_rank(self):
        # Compute effective rank
        T_1 = self.linear[1].weight
        T_0 = self.linear[0].weight
        T = T_1 @ T_0
        # Compute the effective rank of T, which is the exponential of the entropy of the singular values
        s = torch.linalg.svdvals(T)
        s /= s.sum()
        return torch.exp(-torch.sum(s * torch.log(s)))

    def forward_nn(self, x: torch.Tensor, lagged: bool = False) -> torch.Tensor:
        x = self.encoder(x)
        if lagged:
            x = self.linear(x)
        return x

    def training_step(self, train_batch, batch_idx):
        # data
        x_t = self._setup_graph_data(train_batch, key="data_list")
        x_lag = self._setup_graph_data(train_batch, key="data_list_lag")
        # forward
        f_t = self.forward_nn(x_t)
        f_lag = self.forward_nn(x_lag, lagged=True)
        if self.model_args.save_embeddings:
            self.embeddings["t"].append((f_t.detach().cpu()))
            self.embeddings["lag"].append((f_lag.detach().cpu()))
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
        # opt:linear_norm
        if self.model_args.normalize_lin:
            raise NotImplementedError()
            # normalize_linear_layer(self.linear)
        # opt:scheduler_step
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step()

        # log
        with torch.no_grad():
            loss_noreg = self.loss.noreg(f_t, f_lag)
            effective_rank = self.effective_rank()
        loss_dict = {
            "train_loss": -loss,
            "train_loss_noreg": -loss_noreg,
            "effective_rank": effective_rank,
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

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            logger.info(f"Checkpoints at {self.trainer.checkpoint_callback.dirpath}")
        if not self.model_args.save_embeddings:
            return

        # to_save = {"linear": self.linear.weight.detach().cpu()}
        to_save = {}
        for k, v in self.embeddings.items():
            to_save[k] = torch.cat(v)

        dirpath = Path(self.trainer.checkpoint_callback.dirpath).parent
        act_path = dirpath / f"embeddings/epoch={self.current_epoch}.pt"
        act_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to disk
        torch.save(
            to_save,
            act_path.__str__(),
        )

        # Clear for next epoch
        self.embeddings = {"t": [], "lag": []}

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

        # Cosine annealing for half of the epochs and then constant.
        decay_epochs = self.model_args.epochs // 2
        scheduler = CosineAnnealingLR(
            encoder_opt,
            T_max=decay_epochs,  # Remaining 90% of epochs
            eta_min=self.model_args.min_encoder_lr,  # Minimum learning rate
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
