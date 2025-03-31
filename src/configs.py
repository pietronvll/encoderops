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