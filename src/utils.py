import torch


def normalize_linear_layer(layer):
    with torch.no_grad():
        T = layer.weight
        # Normalize T with respect its operator norm
        T_norm = torch.linalg.matrix_norm(T, ord=2)
        layer.weight.div_(T_norm)