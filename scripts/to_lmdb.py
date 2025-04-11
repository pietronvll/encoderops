import json
import os
import pickle
from pathlib import Path

import lmdb
import tyro
from loguru import logger
from mlcolvar.data.graph.utils import _create_dataset_from_configuration
from tqdm import tqdm

from src.data import mdtraj_load, traj_to_confs


def main(
    protein_id: str,
    traj_id: int = 0,
    cutoff: float = 7.0,
    system_selection: str = "all and not type H",
):
    data_path = Path(os.environ["DATA_PATH"])
    protein_path = (
        data_path
        / f"DESRES-Trajectory_{protein_id}-{traj_id}-protein/{protein_id}-{traj_id}-protein"
    )
    trajectory_files = [str(traj) for traj in protein_path.glob("*.dcd")]
    top = next(protein_path.glob("*.pdb")).__str__()
    name = next(protein_path.glob("*.pdb")).stem
    traj = mdtraj_load(trajectory_files, top, 1)
    configs, z_table, atom_names = traj_to_confs(
        traj, system_selection=system_selection
    )
    metadata = {
        "cutoff": cutoff,
        "system_selection": system_selection,
    }
    database_path = protein_path.parent / "lmdb"
    logger.info(database_path.__str__())
    if not database_path.exists():
        database_path.mkdir(parents=True)
    metadata_path = database_path / "metadata.json"
    json.dump(metadata, open(metadata_path, "w"))
    lmdb_path = database_path / f"{name}.lmdb"
    map_size = 10_995_116_277_760  # 1 TB
    env = lmdb.open(lmdb_path.__str__(), map_size=map_size, subdir=False)
    store_to_lmdb(env, configs, z_table, cutoff)


def store_to_lmdb(env, configs, z_table, cutoff):
    """Store dictionary data to LMDB"""
    with env.begin(write=True) as txn:
        length = len(configs)
        # Store key metadata
        txn.put(b"__len__", pickle.dumps(length))
        for idx, c in tqdm(list(enumerate(configs))):
            value = _create_dataset_from_configuration(
                config=c,
                z_table=z_table,
                cutoff=cutoff,
                buffer=0.0,
            )
            pickled_value = pickle.dumps(value)
            item_key = f"item_{idx}".encode()
            txn.put(item_key, pickled_value)

            del value
            del pickled_value



if __name__ == "__main__":
    tyro.cli(main)
