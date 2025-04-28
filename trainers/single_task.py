# uv run --env-file=.env -- python -m  trainers.chignolin [ARGS]


import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch_geometric.loader import DataLoader

from src.configs import Config, default_configs
from src.data import DESRESDataset
from src.model import EvolutionOperator


def main(config: Config):
    # Data Loading
    dataset = DESRESDataset(
        config.data_args.protein_id,
        traj_id=config.data_args.traj_id,
        lagtime=config.data_args.lagtime,
    )
    logger.info(
        f"Loaded dataset {dataset.protein_id}-{dataset.traj_id} | cutoff {dataset.cutoff} Ang | lagtime {dataset.lagtime_ns} ns"
    )

    train_dataloader = DataLoader(
        dataset, batch_size=config.data_args.batch_size, shuffle=True, num_workers=8
    )

    # Model Init
    model = EvolutionOperator(
        cutoff=dataset.cutoff,
        atomic_numbers=dataset.z_table.zs,
        model_args=config.model_args,
        data_args=config.data_args,
    )

    wandb_logger = WandbLogger(
        project=f"encoderops-{config.data_args.protein_id}-{config.data_args.traj_id}",
        entity="csml",
        save_dir="./logs",
    )
    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="cuda",
        devices=config.num_devices,
        max_epochs=config.model_args.epochs,
        enable_model_summary=True,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs)
    main(config)
