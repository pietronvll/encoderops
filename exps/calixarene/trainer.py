# uv run python -m exps.trpcage.trainer trp-cage --help

from dataclasses import asdict

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import CalixareneDataModule
from src.model import EvolutionOperator
from src.modules import SchNet


def main(cfg: Configs):
    datamodule = CalixareneDataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.prepare_data()
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
    encoder_args = {
        "n_out": cfg.trainer_args.latent_dim,
        "cutoff": cfg.data_args.cutoff_ang,
        "atomic_numbers": datamodule.dataset.z_table.zs,
    }
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(SchNet, encoder_args, cfg.trainer_args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)
