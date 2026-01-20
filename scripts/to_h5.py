import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro
from loguru import logger
from tqdm import tqdm

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
    with h5py.File(h5_path, "w", libver='latest') as f:
        # Store metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
        
        # Store z_table as pickled binary
        z_table_binary = pickle.dumps(z_table)
        f.create_dataset(
            "z_table",
            data=np.frombuffer(z_table_binary, dtype=np.uint8),
            compression="gzip",
            compression_opts=4
        )
        
        # Pickle all configurations and concatenate them with size prefixes
        logger.info("Pickling all configurations...")
        pickled_configs = []
        config_sizes = []
        for config in tqdm(configs, desc="Pickling configurations"):
            pickled_config = pickle.dumps(config)
            pickled_configs.append(pickled_config)
            config_sizes.append(len(pickled_config))
        
        # Concatenate all pickled data
        logger.info("Concatenating pickled data...")
        all_data = b"".join(pickled_configs)
        
        # Store all configurations in a single dataset
        logger.info("Writing to HDF5...")
        configs_group = f.create_group("configurations")
        configs_group.attrs["__len__"] = len(configs)
        
        # Store all pickled data as one large array
        configs_group.create_dataset(
            "data",
            data=np.frombuffer(all_data, dtype=np.uint8),
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        
        # Store offsets for quick access
        offsets = np.cumsum([0] + config_sizes[:-1], dtype=np.uint64)
        configs_group.create_dataset("offsets", data=offsets, compression="gzip")
        configs_group.create_dataset("sizes", data=np.array(config_sizes, dtype=np.uint32), compression="gzip")
        
        logger.info(f"Data stored to {h5_path}")


if __name__ == "__main__":
    tyro.cli(main)
