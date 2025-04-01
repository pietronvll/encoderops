from dataclasses import dataclass


@dataclass
class ModelArgs:
    latent_dim: int
    encoder_lr: float
    linear_lr: float
    min_encoder_lr: float
    epochs: int
    max_grad_norm: float | None
    normalize_lin: bool
    regularization: float
    simnorm_dim: int = 8
    n_bases: int = 12
    n_layers: int = 2
    n_filters: int = 32
    n_hidden_channels: int = 32


@dataclass
class DataArgs:
    lagtime: int = 100
    batch_size: int = 64
    cutoff: float = 8.0
    remove_isolated_nodes: bool = False
    system_selection: str | None = None


@dataclass
class Config:
    model_args: ModelArgs
    data_args: DataArgs


default_configs = {
    "chignolin-dev": (
        "Chignolin dev configs",
        Config(
            model_args=ModelArgs(
                latent_dim=128,
                encoder_lr=1e-3,
                linear_lr=1e-3,
                min_encoder_lr=1e-5,
                epochs=50,
                max_grad_norm=None,
                normalize_lin=True,
                regularization=1e-5,
            ),
            data_args=DataArgs(
                lagtime=500, cutoff=6.0, system_selection="all and not type H"
            ),
        ),
    ),
    "chignolin-prod": (
        "Chignolin production configs",
        Config(
            model_args=ModelArgs(
                latent_dim=512,
                encoder_lr=1e-3,
                linear_lr=1e-3,
                min_encoder_lr=1e-5,
                epochs=100,
                max_grad_norm=0.1,
                normalize_lin=True,
                regularization=1e-5,
            ),
            data_args=DataArgs(
                lagtime=100, cutoff=8.0, system_selection="all and not type H"
            ),
        ),
    ),
}
