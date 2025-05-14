# uv run --env-file=.env -- python -m  trainers.chignolin [ARGS]


import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from mlcolvar.data.graph.atomic import AtomicNumberTable
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

from src.configs import Config, MultiTaskConfig, default_configs
from src.data import CalixareneDataset, DESRESDataset
from src.model import EvolutionOperator

torch.set_float32_matmul_precision("high")


def make_calixarene_dataset(data_args):
    assert data_args.protein_id in ["G1", "G2", "G3", "G4", "G5"]
    traj_ids = [idx for idx in range(data_args.traj_id + 1)]
    logger.info(f"Traj ids for protein {data_args.protein_id}: {traj_ids}")
    datasets = [
        CalixareneDataset(
            data_args.protein_id,
            traj_id=traj_id,
            lagtime=data_args.lagtime,
        )
        for traj_id in traj_ids
    ]
    dataset = ConcatDataset(datasets)
    dataset.cutoff = dataset.datasets[0].cutoff
    dataset.lagtime_ns = dataset.datasets[0].lagtime_ns
    dataset.protein_id = dataset.datasets[0].protein_id
    dataset.z_table = dataset.datasets[0].z_table
    return dataset


def main(config: Config | MultiTaskConfig):
    # Data Loading
    if isinstance(config, MultiTaskConfig):
        datasets = [
            make_calixarene_dataset(data_args) for data_args in config.data_args
        ]
        dataset = ConcatDataset(datasets)
        dataset.cutoff = dataset.datasets[0].cutoff
        dataset.lagtime_ns = dataset.datasets[0].lagtime_ns
        # Merge atomic numbers
        atomic_numbers = sorted(list(set(sum([ds.z_table.zs for ds in datasets], []))))
        z_table = AtomicNumberTable(atomic_numbers)
        for ds in datasets:
            ds.z_table = z_table
        dataset.z_table = z_table
        dataset.protein_id = "-".join([ds.protein_id for ds in datasets])
    else:
        if config.data_args.protein_id in ["G1", "G2", "G3", "G4", "G5"]:
            dataset = make_calixarene_dataset(config.data_args)
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
        batch_size=config.data_args.batch_size
        if isinstance(config, Config)
        else config.batch_size,
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
        project=f"encoderops-{dataset.protein_id}",
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

    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(default_configs)
    main(config)
