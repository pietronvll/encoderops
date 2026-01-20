import os
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro
from loguru import logger

from src.data import mdtraj_load, traj_to_confs


def main(
    protein_id: str,
    traj_id: int = 0,
    system_selection: str | None = "all and not type H",
):
    data_path = Path(os.environ["DATA_PATH"])
    protein_path = (
        data_path
        / f"DESRES-Trajectory_{protein_id}-{traj_id}-protein/{protein_id}-{traj_id}-protein"
    )
    trajectory_files = sorted([str(traj) for traj in protein_path.glob("*.dcd")])
    top = next(protein_path.glob("*.pdb")).__str__()
    name = next(protein_path.glob("*.pdb")).stem
    traj = mdtraj_load(trajectory_files, top, 1)
    if system_selection is not None:
        system_atoms = traj.top.select(system_selection)
        logger.info(f"System selection: {system_selection}")
        traj = traj.atom_slice(system_atoms)
    configs, z_table, _ = traj_to_confs(traj)
    metadata = {
        "system_selection": system_selection,
        "lagtime_ns": 0.2,
    }
    database_path = Path(__file__).parent.parent / "h5_data"
    logger.info(database_path.__str__())
    if not database_path.exists():
        database_path.mkdir(parents=True)
    h5_path = database_path / f"{name}.h5"
    store_to_h5(h5_path, configs, z_table, metadata)


@torch.no_grad()
def store_to_h5(h5_path, configs, z_table, metadata):
    with h5py.File(h5_path, "w") as f:
        # Store configurations as a dataset
        configs_array = np.array(configs)
        f.create_dataset("configurations", data=configs_array, compression="gzip")
        
        # Store z_table
        f.create_dataset("z_table", data=z_table, compression="gzip")
        
        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
        
        logger.info(f"Data stored to {h5_path}")


if __name__ == "__main__":
    tyro.cli(main)
