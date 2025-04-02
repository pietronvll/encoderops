# uv run --env-file=.env -- python -m  trainers.chignolin [ARGS]
import os
import pickle
from pathlib import Path

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader

from src.configs import Config, default_configs
from src.data import get_dataset
from src.model import EvolutionOperator


def main(config: Config):
    # Data Loading
    data_path = Path(os.environ["CHIG_DATA_PATH"])
    trajectory_files = [str(traj) for traj in data_path.glob("*.dcd")]
    top = next(data_path.glob("*.pdb")).__str__()
    name = next(data_path.glob("*.pdb")).stem
    lagtime = config.data_args.lagtime
    prepro_data_path = (
        Path(__file__).parent.parent / f"preprocessed_data/{name}-lag{lagtime}.pkl"
    )

    # If the file exists then load it with pickle, otherwise call get_dataset and save it
    if prepro_data_path.exists():
        with open(prepro_data_path, "rb") as f:
            dataset = pickle.load(f)

    else:
        dataset = get_dataset(trajectory_files, top, config.data_args)
        prepro_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prepro_data_path, "wb") as f:
            pickle.dump(dataset, f)

    train_dataloader = DataLoader(
        dataset, batch_size=config.data_args.batch_size, shuffle=True
    )

    # Model Init
    model = EvolutionOperator(
        cutoff=dataset.metadata["cutoff"],
        atomic_numbers=dataset.metadata["z_table"],
        model_args=config.model_args,
        data_args=config.data_args,
    )

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(every_n_epochs=5, save_top_k=-1)
    # Trainer
    trainer = Trainer(
        logger=WandbLogger(project="encoderops_chignolin", entity="csml"),
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
