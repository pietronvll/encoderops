import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.configs import MultiTaskConfig, default_configs
from src.data import DESRESDataset, ConcatDESRES
from src.model import EvolutionOperator



def main(config: MultiTaskConfig):
    # Data Loading
    datasets = [
        DESRESDataset(
            data_args.protein_id,
            traj_id=data_args.traj_id,
            lagtime=data_args.lagtime,
        )
        for data_args in config.data_args
    ]
    dataset = ConcatDESRES(datasets)

    logger.info(
        f"Loaded {len(dataset.datasets)} datasets:"
    )
    for ds in dataset.datasets:
        logger.info(f"\n  - {ds.protein_id}-{ds.traj_id} | cutoff {ds.cutoff} Ang | lagtime {ds.lagtime_ns} ns")
    

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
        project=f"encoderops-{config.data_args.protein_id}-{config.data_args.traj_id}",
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
        enable_model_summary=True,
    )

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs)
    main(config)
