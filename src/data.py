import os
import pickle
from functools import lru_cache

import lmdb
import mdtraj
from mlcolvar.utils.io import (
    _configures_from_trajectory,
    _names_from_top,
    _z_table_from_top,
)
from torch.utils.data import Dataset


def traj_to_confs(traj: mdtraj.Trajectory, system_selection: str | None = None):
    configs = _configures_from_trajectory(traj, system_selection=system_selection)
    z_table = _z_table_from_top([traj.top])
    atom_names = _names_from_top([traj.top])
    return configs, z_table, atom_names


def mdtraj_load(trajectory_files: list[str], top: str, stride: int = 1000):
    traj = mdtraj.load(trajectory_files, top=top, stride=stride)
    traj.top = mdtraj.core.trajectory.load_topology(top)
    return traj


class DESRESDataset(Dataset):
    def __init__(self, protein_id: str, traj_id: int = 0, lagtime: int = 1):
        super().__init__()
        dataset_path = (
            os.environ("DATA_PATH")
            / f"DESRES-Trajectory_{protein_id}-{traj_id}-protein/{protein_id}-{traj_id}-protein/lmdb/{protein_id}-{traj_id}-protein.lmdb"
        )
        assert dataset_path.exists()
        map_size = 10_995_116_277_760  # 1 TB
        self.env = lmdb.open(self.lmdb_path, map_size=map_size, subdir=False)
        self.lagtime = lagtime

    @lru_cache(maxsize=10000)  # Cache recently used items
    def _get_lmdb_item(self, key, idx):
        """Get a specific item from LMDB with caching"""
        if self.env is None:
            raise RuntimeError("LMDB environment not initialized")

        item_key = f"{key}_{idx}".encode()
        with self.env.begin(write=False) as txn:
            data_binary = txn.get(item_key)
            if data_binary is None:
                raise KeyError(f"Key '{key}' at index {idx} not found in LMDB")
            return pickle.loads(data_binary)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._dictionary[index]
        else:
            slice_dict = {}
            for key, val in self._dictionary.items():
                try:
                    slice_dict[key] = val[index]
                except:
                    slice_dict[key] = list(itemgetter(*index)(val))
            return slice_dict

    def __len__(self):
        value = next(iter(self._dictionary.values()))
        return len(value)
