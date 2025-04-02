from dataclasses import dataclass


@dataclass
class ModelArgs:
    latent_dim: int
    linear_lora: int
    encoder_lr: float
    linear_lr: float
    min_encoder_lr: float
    epochs: int
    max_grad_norm: float | None
    normalize_lin: bool
    regularization: float
    simnorm_dim: int = 4
    n_bases: int = 16
    n_layers: int = 3
    n_filters: int = 32
    n_hidden_channels: int = 64
    save_embeddings: bool = False


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
    num_devices: int = 1


default_configs = {
    "chignolin-dev": (
        "Chignolin dev configs",
        Config(
            model_args=ModelArgs(
                latent_dim=16,
                linear_lora=4,
                encoder_lr=1e-3,
                linear_lr=1e-2,
                min_encoder_lr=1e-5,
                epochs=50,
                max_grad_norm=None,
                normalize_lin=False,
                regularization=1e-5,
                save_embeddings=True,
            ),
            data_args=DataArgs(lagtime=500, cutoff=6.0, system_selection="name CA"),
        ),
    ),
    "chignolin-prod": (
        "Chignolin production configs",
        Config(
            model_args=ModelArgs(
                latent_dim=64,
                linear_lora=16,
                encoder_lr=1e-2,
                linear_lr=1e-2,
                min_encoder_lr=1e-3,
                epochs=500,
                max_grad_norm=0.2,
                normalize_lin=False,
                regularization=1e-4,
                save_embeddings=False,
            ),
            data_args=DataArgs(
                lagtime=50,
                cutoff=7.0,
                # system_selection="all and not type H",
                system_selection="name CA",
                batch_size=128,
            ),
            num_devices=4,
        ),
    ),
}
