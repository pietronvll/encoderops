from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class TrainerArgs:
    latent_dim: int
    "Dimension of the latent space"
    encoder_lr: float
    "Learning rate for the encoder"
    linear_lr: float
    "Learning rate for the linear (transfer operator) layer"
    epochs: int
    "Number of training epochs"
    batch_size: int
    "Batch size for training"
    max_grad_norm: float | None
    "Maximum gradient norm for gradient clipping. If None, no clipping is performed"
    normalize_lin: bool
    "Whether to apply spectral normalization to the linear layer"
    regularization: float
    "Regularization strength for the spectral loss"
    min_encoder_lr: float | None = None
    "Minimum learning rate for the encoder, used in cosine annealing scheduler. If None, no scheduler is used."
    normalize_latents: Literal["simnorm", "euclidean"] | None = "simnorm"
    "Normalization method for the latent space. Can be 'simnorm', 'euclidean', or None"
    simnorm_dim: int = 4
    "Dimension for the SimNorm normalization, if used"


@dataclass
class SchNetModelArgs:
    n_bases: int = 16
    "Number of radial basis functions"
    n_layers: int = 3
    "Number of interaction layers"
    n_filters: int = 32
    "Number of filters in the interaction layers"
    n_hidden_channels: int = 64
    "Number of hidden channels in the interaction layers"


@dataclass
class MLPModelArgs:
    n_hidden: int = 2
    "Number of hidden layers"
    layer_size: int = 16
    "Size of each hidden layer"


@dataclass
class ResNet18ModelArgs:
    drop_rate: float = 0.2
    "Dropout rate"


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
    data_path: str | None = None
    "Path to the LMDB database. If None, tries to read the 'DATA_PATH' environment variable."
    remove_isolated_nodes: bool = False
    "Whether to remove isolated nodes from the graph"


@dataclass
class CalixareneDataArgs:
    molecule_ids: Tuple[str, ...]
    "Molecule IDs"
    traj_ids: Tuple[int, ...] = field(default_factory=lambda: tuple(range(2)))
    "Trajectory IDs"
    lagtime: int = 1
    "Lagtime (in number of frames) used to generate lagged data"
    cutoff_ang: float = 7.0
    "Cutoff distance in Angstroms for defining neighbors in the graph"
    data_path: str | None = None
    "Path to the Calixarene data. If None, tries to read the 'DATA_PATH' environment variable."
    keep_mdtraj: bool = False
    "Whether to keep the MDTraj trajectory in memory"


@dataclass
class Lorenz63DataArgs:
    lagtime: int = 1
    "Lagtime (in number of frames) used to generate lagged data"
    history_len: int = 1
    "Number of frames to use as history"
    data_path: str | None = None
    "Path to the data file. If None, tries to read the 'DATA_PATH' environment variable"


@dataclass
class SSTDataArgs:
    lagtime: int = 1
    "Lagtime (in number of months) used to generate lagged data"
    history_len: int = 0
    "Number of frames to use as history"
    data_path: str | None = None
    "Path to the data file. If None, tries to read the 'DATA_PATH' environment variable"


@dataclass
class Configs:
    trainer_args: TrainerArgs
    model_args: SchNetModelArgs | MLPModelArgs | ResNet18ModelArgs
    data_args: DESRESDataArgs | Lorenz63DataArgs | CalixareneDataArgs | SSTDataArgs
    wandb_project: str
    wandb_entity: str | None = None
    offline: bool = False
    num_devices: int = -1
    dataloader_workers: int = 8


defaults = {
    "trp-cage": (
        "TRP-cage Experiment",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                epochs=20,
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
    "l63": (
        "Lorenz63",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=8,
                encoder_lr=1e-3,
                linear_lr=1e-3,
                epochs=100,
                batch_size=32,
                max_grad_norm=None,
                normalize_lin=False,
                regularization=0.0,
                min_encoder_lr=1e-4,
                normalize_latents=None,
            ),
            model_args=MLPModelArgs(),
            data_args=Lorenz63DataArgs(lagtime=10, history_len=0),
            wandb_project="encoderops-lorenz63",
        ),
    ),
    "G2": (
        "Calixarene-G2 system",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-4,
                epochs=25,
                batch_size=32,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-5,
                normalize_latents=None,
            ),
            model_args=SchNetModelArgs(),
            data_args=CalixareneDataArgs(molecule_ids=("G2",), lagtime=500),
            wandb_project="encoderops-calixarene-G2",
        ),
    ),
    "G13": (
        "Calixarene-G1+G3 system",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-4,
                epochs=25,
                batch_size=128,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-5,
                normalize_latents=None,
            ),
            model_args=SchNetModelArgs(),
            data_args=CalixareneDataArgs(molecule_ids=("G1", "G3"), lagtime=500),
            wandb_project="encoderops-calixarene-G1+3",
        ),
    ),
    "ENSO": (
        "Enso Modes",
        Configs(
            trainer_args=TrainerArgs(
                latent_dim=128,
                encoder_lr=1e-3,
                linear_lr=1e-3,
                epochs=1000,
                batch_size=64,
                max_grad_norm=1e-5,
                normalize_lin=False,
                regularization=1e-4,
                normalize_latents="euclidean",
                # simnorm_dim=2,
            ),
            model_args=ResNet18ModelArgs(),
            data_args=SSTDataArgs(history_len=0),
            wandb_project="encoderops-ENSO",
            num_devices=1,
        ),
    ),
}
