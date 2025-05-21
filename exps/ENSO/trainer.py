# uv run python -m exps.trpcage.trainer trp-cage --help

from dataclasses import asdict

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import SSTDataModule
from src.model import EvolutionOperator
from src.modules import ResNet18


def main(cfg: Configs):
    datamodule = SSTDataModule(cfg.trainer_args, cfg.data_args, cfg.dataloader_workers)
    datamodule.prepare_data()
    datamodule.setup("fit")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=20, save_top_k=-1, save_last=True
    )
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="cuda",
        devices=cfg.num_devices,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=1,
        enable_model_summary=True,
    )
    # Model
    encoder_args = {
        "num_features": cfg.trainer_args.latent_dim,
        "channels_in": cfg.data_args.history_len + 1,
    }
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(ResNet18, encoder_args, cfg.trainer_args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)
