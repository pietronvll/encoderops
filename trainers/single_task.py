# uv run --env-file=.env -- python -m  trainers.chignolin [ARGS]


import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.configs import Config, default_configs
from src.data import CalixareneDataset, DESRESDataset
from src.model import EvolutionOperator
import torch

torch.set_float32_matmul_precision('high')


def main(config: Config):
    # Data Loading
    if config.data_args.protein_id in ["G1", "G2", "G3", "G4", "G5"]:
        if isinstance(config.data_args.traj_id, list):
            datasets = [
                CalixareneDataset(
                    config.data_args.protein_id,
                    traj_id=traj_id,
                    lagtime=config.data_args.lagtime,
                )
                for traj_id in config.data_args.traj_id
            ]
            dataset = ConcatDataset(datasets)
            dataset.cutoff = dataset.datasets[0].cutoff
            dataset.lagtime_ns = dataset.datasets[0].lagtime_ns
            dataset.protein_id = dataset.datasets[0].protein_id
            dataset.z_table = dataset.datasets[0].z_table

        else:
            dataset = CalixareneDataset(
                config.data_args.protein_id,
                traj_id=config.data_args.traj_id,
                lagtime=config.data_args.lagtime,
            )
    else:
        dataset = DESRESDataset(
            config.data_args.protein_id,
            traj_id=config.data_args.traj_id,
            lagtime=config.data_args.lagtime,
        )
    logger.info(
        f"Loaded dataset {dataset.protein_id} | cutoff {dataset.cutoff} Ang | lagtime {dataset.lagtime_ns} ns"
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=config.data_args.batch_size,
        shuffle=True,
        num_workers=config.dataloader_workers,
    )

    # Model Init
    model = EvolutionOperator(
        cutoff=dataset.cutoff,
        atomic_numbers=dataset.z_table.zs,
        model_args=config.model_args,
        data_args=config.data_args,
    )

    wandb_logger = WandbLogger(
        project=f"encoderops-{config.data_args.protein_id}",
        entity="csml",
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
        devices=config.num_devices,
        max_epochs=config.model_args.epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader
    )


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs)
    main(config)
