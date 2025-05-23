import torch

import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import Lorenz63DataModule
from src.modules import MLP

from kooplearn.models import Nonlinear
from kooplearn.models.feature_maps.nn import NNFeatureMap
from kooplearn.data import traj_to_contexts
from kooplearn.nn import VAMPLoss
from torch.utils.data import DataLoader
from kooplearn.nn.data import collate_context_dataset

from dataclasses import asdict

from linear_operator_learning.nn import MLP

class MLPEncoder(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_hidden,
        layer_size,
        output_shape,
        dropout=0.0,
        activation=torch.nn.ReLU,
        iterative_whitening=False,
        bias=True,
    ):
        super(MLPEncoder, self).__init__()

        self.encoder = MLP(
            input_shape=input_shape,
            n_hidden=n_hidden,
            layer_size=layer_size,
            output_shape=output_shape,
            dropout=dropout,
            activation=activation,
            iterative_whitening=iterative_whitening,
            bias=bias,
        )
        
    def forward(self, x):
        x_enc = self.encoder(x)
        x_enc = torch.cat([x_enc, x], dim=-1)
        return x_enc

def main(cfg: Configs):
    datamodule = Lorenz63DataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_data = datamodule.train_dataset.data
    train_ctxs = traj_to_contexts(train_data.astype('float32'), time_lag=cfg.data_args.lagtime, backend='numpy')
    train_dl = DataLoader(
        train_ctxs,
        batch_size = cfg.trainer_args.batch_size,
        shuffle=cfg.data_args,
        collate_fn=collate_context_dataset,
        num_workers=cfg.dataloader_workers,
        persistent_workers=True,
        )
    val_data = datamodule.val_dataset.data
    val_ctxs = traj_to_contexts(val_data.astype('float32'), time_lag=cfg.data_args.lagtime, backend='numpy')
    val_dl = DataLoader(
        val_ctxs,
        batch_size = cfg.trainer_args.batch_size,
        shuffle=cfg.data_args,
        collate_fn=collate_context_dataset,
        num_workers=1,
        persistent_workers=True,
        )

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
        tags=["VAMPNets"],
        name=f"VAMPNets_rep{cfg.trainer_args.seed}",    
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=25, save_top_k=-1, save_last=True
    )
    # Timer
    timer = Timer()
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, timer],
        accelerator="cuda",
        devices=cfg.num_devices,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    num_vars = datamodule.train_dataset.num_variables
    # Model
    encoder_args = {
        "input_shape": num_vars * (cfg.data_args.history_len + 1),
        "output_shape": cfg.trainer_args.latent_dim,
        "dropout": 0.0,
        "activation": torch.nn.ReLU,
        "iterative_whitening": False,
        "bias": True,
    }
    loss_args = {
        'schatten_norm': 2,
        'center_covariances': False
        }
    encoder_args = encoder_args | asdict(cfg.model_args)
    feature_map = NNFeatureMap(
        MLPEncoder,
        VAMPLoss,
        torch.optim.AdamW,
        trainer,
        encoder_kwargs=encoder_args,
        loss_kwargs=loss_args,
        optimizer_kwargs={'lr': cfg.trainer_args.encoder_lr},
        seed=cfg.trainer_args.seed,
    )    
    feature_map.fit(train_dl, val_dl)
    runtime = timer.time_elapsed("train")
    wandb_logger.experiment.log({"runtime": runtime})
    feature_map.save(checkpoint_callback.dirpath + f"/last.pt")

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)