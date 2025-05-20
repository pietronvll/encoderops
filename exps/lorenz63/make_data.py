from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro
import xarray as xr
from kooplearn.datasets import Lorenz63
from loguru import logger


def generate_dataset(
    local_dir: str | None = None,
    n_train: int = 10_000,
    n_val: int = 1_000,
    dt: float = 0.001,
) -> None:
    if local_dir is None:
        local_dir = str(Path().cwd().resolve())
        logger.info(
            f"No local directory provided, using current directory ({local_dir})"
        )

    data_path = Path(local_dir) / "data"
    data_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    raw_data = Lorenz63(dt=dt).sample(X0=np.ones(3), T=n_train + 10000 + n_val)
    time = np.arange(len(raw_data)) * dt
    mean = np.mean(raw_data, axis=0)
    norm = np.max(np.abs(raw_data), axis=0)
    # Data rescaling
    data = (raw_data - mean) / norm

    dataset = {
        "train": data[: n_train + 1],
        "val": data[-n_val - 1 :],
        "train_time": time[: n_train + 1],
        "val_time": time[-n_val - 1 :],
    }

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        dataset["train"][:, 0],
        dataset["train"][:, 1],
        dataset["train"][:, 2],
        lw=0.5,
        label="Training data",
    )
    ax.plot(
        dataset["val"][:, 0],
        dataset["val"][:, 1],
        dataset["val"][:, 2],
        lw=2,
        label="Test data",
    )
    ax.set_title("Lorenz System Trajectory")
    ax.set_xlabel("X")
    ax.set_xticks(
        np.linspace(-1, 1, 5),
    )
    ax.set_xticklabels(np.linspace(-1, 1, 5))
    ax.set_ylabel("Y")
    ax.set_yticks(np.linspace(-1, 1, 5))
    ax.set_zlabel("Z")
    ax.set_zticks(np.linspace(-0.5, 0.5, 5))

    plt.legend()
    fig.savefig(data_path / "dataset.png")
    plt.close(fig)

    # Create xarray Dataset
    # combined_data = np.concatenate([dataset["train"], dataset["val"]], axis=0)
    # combined_time = np.concatenate([dataset["train_time"], dataset["val_time"]])
    split = np.array(
        ["train"] * (n_train + 1) + ["buffer"] * 9999 + ["val"] * (n_val + 1)
    )

    ds = xr.Dataset(
        data_vars={
            "trajectory": (["time", "dim"], data),
        },
        coords={
            "time": time,
            "split": ("time", split),
            "dim": ["x", "y", "z"],
        },
        attrs={
            "description": "Lorenz63 trajectory normalized by max-abs after centering",
            "dt": dt,
            "mean": mean.tolist(),
            "norm": norm.tolist(),
        },
    )

    ds.to_netcdf(data_path / "lorenz63_dataset.nc")
    logger.info(f"Dataset saved to {data_path / 'lorenz63_dataset.nc'}")


if __name__ == "__main__":
    tyro.cli(generate_dataset)
