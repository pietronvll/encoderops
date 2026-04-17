# uv run python -m exps.trpcage.trainer trp-cage --help

from dataclasses import asdict

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from src.configs import Configs, defaults
from src.data import DESRESDataModule
from src.model import EvolutionOperator
from src.modules import SchNet
from src.utils import build_run_logger


def main(cfg: Configs):
    datamodule = DESRESDataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.setup("fit")

    run_logger = build_run_logger(
        offline=cfg.offline,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        save_dir="./logs",
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1, save_top_k=-1, save_last=True
    )
    # Trainer
    trainer_kwargs = {
        "logger": run_logger,
        "callbacks": [checkpoint_callback],
        "accelerator": cfg.accelerator,
        "devices": cfg.num_devices,
        "max_epochs": cfg.trainer_args.epochs,
        "log_every_n_steps": 10,
        "enable_model_summary": True,
    }
    if cfg.num_devices > 1:
        trainer_kwargs["strategy"] = (
            "ddp"
            if cfg.trainer_args.share_encoder
            else "ddp_find_unused_parameters_true"
        )
    trainer = Trainer(**trainer_kwargs)
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
