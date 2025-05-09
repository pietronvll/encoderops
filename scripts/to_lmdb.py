import json
import os
import pickle
from pathlib import Path

import torch
import tyro
from loguru import logger
from tqdm import tqdm

import lmdb
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
    trajectory_files = [str(traj) for traj in protein_path.glob("*.dcd")]
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
    database_path = Path(__file__).parent.parent / "lmdb"
    # database_path = protein_path.parent / "lmdb"
    logger.info(database_path.__str__())
    if not database_path.exists():
        database_path.mkdir(parents=True)
    metadata_path = database_path / f"metadata-{name}.json"
    json.dump(metadata, open(metadata_path, "w"))
    lmdb_path = database_path / f"{name}.lmdb"
    map_size = 10_995_116_277_760  # 1 TB
    env = lmdb.open(lmdb_path.__str__(), map_size=map_size, subdir=False)
    store_to_lmdb(env, configs, z_table)


@torch.no_grad()
def store_to_lmdb(env, configs, z_table):
    with env.begin(write=True) as txn:
        length = len(configs)
        txn.put(b"__len__", pickle.dumps(length))
        txn.put(b"z_table", pickle.dumps(z_table))

    for idx, c in tqdm(list(enumerate(configs))):
        with env.begin(write=True) as txn:
            item_key = f"item_{idx}".encode()
            pickled_value = pickle.dumps(c)
            txn.put(item_key, pickled_value)
            del pickled_value  # Explicitly delete the pickled data


if __name__ == "__main__":
    tyro.cli(main)
