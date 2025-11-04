# uv run python -m exps.ENSO.trainer ENSO_CESM --help

from dataclasses import asdict

import tyro
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import SSTDataModule
from src.model import EvolutionOperator
from src.utils import EpochTimerCallback
from src.modules import  CirTModel

def main(cfg: Configs):
    seed_everything(cfg.trainer_args.seed, workers=True)
    datamodule = SSTDataModule(cfg.trainer_args, cfg.data_args, cfg.dataloader_workers)
    datamodule.prepare_data()
    datamodule.setup("fit")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
    )
    # Add configs
    wandb_logger.experiment.config.update(asdict(cfg))

    checkpoint_all = ModelCheckpoint(
        every_n_epochs=20, save_last=True, save_top_k=-1, filename="{epoch}"
    )
    checkpoint_best = ModelCheckpoint(
        save_top_k=1,  mode="max", filename="best"
    )
    timer = EpochTimerCallback()

    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_all, checkpoint_best, timer],
        accelerator="cuda",
        devices=cfg.num_devices,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=1,
        enable_model_summary=True,
    )
    # Model
    encoder_args = {
        "num_classes": cfg.trainer_args.latent_dim,
        "in_chans": cfg.data_args.history_len + (2 if cfg.data_args.mask else 1),
    }
    encoder_args = encoder_args 
    model = EvolutionOperator(CirTModel, encoder_args, cfg.trainer_args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)
