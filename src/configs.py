from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainerArgs:
    latent_dim: int
    encoder_lr: float
    linear_lr: float
    epochs: int
    batch_size: int
    max_grad_norm: float | None
    normalize_lin: bool
    regularization: float
    min_encoder_lr: float | None = None
    normalize_latents: Literal["simnorm", "euclidean"] | None = "simnorm"
    simnorm_dim: int = 4


@dataclass
class SchNetModelArgs:
    n_bases: int = 16
    n_layers: int = 3
    n_filters: int = 32
    n_hidden_channels: int = 64


@dataclass
class DESRESDataArgs:
    protein_id: str
    "Protein ID"
    traj_id: int = 0
    "Trajectory ID"
    lagtime: int = 1
    "Lagtime (in number of frames) used to generate lagged data"
    cutoff_ang: float = 7.0
    "Cutoff distance in Angstroms for defining neighbors in the graph"
    lmdb_path: str | None = None
    "Path to the LMDB database. If None, tries to read the 'LMDB_PATH' environment variable"
    remove_isolated_nodes: bool = False
    "Whether to remove isolated nodes from the graph"


@dataclass
class Configs:
    trainer_args: TrainerArgs
    model_args: SchNetModelArgs
    data_args: DESRESDataArgs
    wandb_project: str
    wandb_entity: str | None = None
    offline: bool = False
    num_devices: int = -1
    dataloader_workers: int = 8


defaults = {
    "trp-cage": (
        "TRP-cage - Single Task",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                epochs=45,
                batch_size=64,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-5,
                min_encoder_lr=1e-4,
            ),
            model_args=SchNetModelArgs(),
            data_args=DESRESDataArgs(
                protein_id="2JOF",
                lagtime=500,  # 100ns
            ),
            wandb_project="encoderops-2JOF",
        ),
    ),
}
