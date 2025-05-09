import torch


def lin_svdvals(layer: torch.nn.Linear | torch.nn.Sequential) -> torch.Tensor:
    if isinstance(layer, torch.nn.Linear):
        T = layer.weight
    elif isinstance(layer, torch.nn.Sequential):
        linears = [m for m in layer]
        for lin_idx, linear in enumerate(linears):
            if not isinstance(linear, torch.nn.Linear):
                raise ValueError(
                    f"Sequential layer must contain only linear layers, while the {lin_idx + 1}-th is of type {type(linear)}"
                )
        T = torch.linalg.multi_dot([linear.weight for linear in linears])
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")
    return torch.linalg.svdvals(T)


def effective_rank(s: torch.Tensor) -> float:
    # Compute the effective rank of T, which is the exponential of the entropy of the singular values
    s = torch.clamp(s, min=1e-8) # Clip negative terms
    norm_s = s / s.sum()
    return torch.exp(-torch.sum(norm_s * torch.log(norm_s))).item()


def normalize_linear_layer(layer) -> float:
    with torch.no_grad():
        # Check if the layer is a linear layer
        if isinstance(layer, torch.nn.Linear):
            T = layer.weight
            # Normalize T with respect its operator norm
            T_norm = torch.linalg.matrix_norm(T, ord=2)
            layer.weight.div_(T_norm)
            return T_norm.item()

        elif isinstance(layer, torch.nn.Sequential):
            # Collect all linear layers in the Sequential
            linears = [m for m in layer]
            for lin_idx, linear in enumerate(linears):
                if not isinstance(linear, torch.nn.Linear):
                    raise ValueError(
                        f"Sequential layer must contain only linear layers, while the {lin_idx + 1}-th is of type {type(linear)}"
                    )

            # Compute the product of all linear layers' weights (T = T1 @ T2 @ ... @ Tn)
            weights = [linear.weight for linear in linears]
            T = torch.linalg.multi_dot(weights)

            # Compute the spectral norm of the product
            T_norm = torch.linalg.matrix_norm(T, ord=2)

            # Normalize each linear layer by T_norm^(1/num_linears)
            scale = T_norm ** (1.0 / len(linears))
            for linear in linears:
                linear.weight.div_(scale)

            return T_norm.item()
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")
