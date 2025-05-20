# uv run python -m exps.trpcage.trainer trp-cage --help

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import DESRESDataModule


def main(cfg: Configs):
    datamodule = DESRESDataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.setup("fit")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1, save_top_k=-1, save_last=True
    )
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="cuda",
        devices=cfg.num_devices,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    trainer.fit(model, datamodule=datamodule)


def test_trainer(cfg: Configs):
    for lmdb_path in ["/home/novelli/encoderops/lmdb", None]:
        datamodule = DESRESDataModule(
            cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
        )
        datamodule.setup("fit")


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    test_trainer(config)
