# uv run python -m exps.trpcage.trainer trp-cage --help

from dataclasses import asdict

import torch
import tyro
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import Lorenz63DataModule
from src.model import EvolutionOperator
from src.modules import MLP


def main(cfg: Configs):
    seed_everything(cfg.trainer_args.seed, workers=True)
    datamodule = Lorenz63DataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
        tags=["EvOp"],
        name=f"EvOp_rep{cfg.trainer_args.seed}",    
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
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(MLP, encoder_args, cfg.trainer_args)
    trainer.fit(model, datamodule=datamodule)
    runtime = timer.time_elapsed("train")
    wandb_logger.experiment.log({"runtime": runtime})


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)
