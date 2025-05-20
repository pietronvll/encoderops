from dataclasses import dataclass
from typing import Literal


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
    "Minimum learning rate for the encoder, used in cosine annealing scheduler. If None, no scheduler is used"
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
class Lorenz63DataArgs:
    lagtime: int = 1
    "Lagtime (in number of frames) used to generate lagged data"
    history_len: int = 1
    "Number of frames to use as history"
    data_path: str | None = None
    "Path to the data file. If None, tries to read the 'DATA_PATH' environment variable"


@dataclass
class Configs:
    trainer_args: TrainerArgs
    model_args: SchNetModelArgs | MLPModelArgs
    data_args: DESRESDataArgs | Lorenz63DataArgs
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
                linear_lr=1e-2,
                epochs=100,
                batch_size=512,
                max_grad_norm=None,
                normalize_lin=True,
                regularization=0.0,
                min_encoder_lr=1e-4,
            ),
            model_args=MLPModelArgs(),
            data_args=Lorenz63DataArgs(lagtime=10, history_len=1),
            wandb_project="encoderops-lorenz63",
        ),
    ),
}
