from dataclasses import dataclass


@dataclass
class ModelArgs:
    latent_dim: int
    encoder_lr: float
    linear_lr: float
    epochs: int
    max_grad_norm: float | None
    normalize_lin: bool
    regularization: float
    min_encoder_lr: float | None = None
    simnorm_dim: int = 4
    n_bases: int = 16
    n_layers: int = 3
    n_filters: int = 32
    n_hidden_channels: int = 64


@dataclass
class DataArgs:
    protein_id: str
    traj_id: int = 0
    lagtime: int = 1
    batch_size: int = 64
    remove_isolated_nodes: bool = False


@dataclass
class Config:
    model_args: ModelArgs
    data_args: DataArgs
    num_devices: int = 1
    dataloader_workers: int = 8


@dataclass
class MultiTaskConfig:
    model_args: ModelArgs
    data_args: list[DataArgs]
    batch_size: int
    num_devices: int = 1
    dataloader_workers: int = 8


default_configs = {
    "chignolin-prod": (
        "Chignolin production configs",
        Config(
            model_args=ModelArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-4,
                epochs=5,
                max_grad_norm=None,
                normalize_lin=True,
                regularization=1e-5,
            ),
            data_args=DataArgs(
                protein_id="CLN025",
            ),
        ),
    ),
    "villin-prod": (
        "Villin production configs",
        Config(
            model_args=ModelArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-4,
                epochs=250,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-5,
            ),
            data_args=DataArgs(
                protein_id="2F4K",
                lagtime=1,
                batch_size=64,
            ),
            num_devices=1,
        ),
    ),
    "multitask": (
        "MultiTask - Chignolin Trp-cage Villin",
        MultiTaskConfig(
            model_args=ModelArgs(
                latent_dim=64,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-4,
                epochs=5,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-5,
            ),
            data_args=[
                DataArgs(protein_id="CLN025"),
                DataArgs(protein_id="2JOF"),
                DataArgs(protein_id="2F4K"),
            ],
            batch_size=32,
            num_devices=2,
            dataloader_workers=10,
        ),
    ),
}
