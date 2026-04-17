# uv run python -m exps.ENSO.trainer ENSO_CESM --help

from dataclasses import asdict

import tyro
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs import Configs, defaults
from src.data import SSTDataModule
from src.model import EvolutionOperator
from src.utils import EpochTimerCallback, build_run_logger
from src.modules import MaskedCNN

def main(cfg: Configs):
    seed_everything(cfg.trainer_args.seed, workers=True)
    datamodule = SSTDataModule(cfg.trainer_args, cfg.data_args, cfg.dataloader_workers)
    datamodule.prepare_data()
    datamodule.setup("fit")

    run_logger = build_run_logger(
        offline=cfg.offline,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        save_dir="./logs",
        config=asdict(cfg),
    )

    checkpoint_all = ModelCheckpoint(
        every_n_epochs=20, save_last=True, save_top_k=-1, filename="{epoch}"
    )
    checkpoint_best = ModelCheckpoint(
        save_top_k=1, monitor="val_loss_noreg", mode="max", filename="best"
    )
    timer = EpochTimerCallback()

    # Trainer
    trainer = Trainer(
        logger=run_logger,
        callbacks=[checkpoint_all, checkpoint_best, timer],
        accelerator=cfg.accelerator,
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
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(MaskedCNN, encoder_args, cfg.trainer_args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)
