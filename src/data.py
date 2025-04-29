import json
import os
import pickle
from pathlib import Path

import mdtraj
from mlcolvar.data.graph.utils import _create_dataset_from_configuration
from mlcolvar.utils.io import (
    _configures_from_trajectory,
    _names_from_top,
    _z_table_from_top,
)
from torch.utils.data import Dataset

import lmdb


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
    def __init__(
        self, protein_id: str, traj_id: int = 0, lagtime: int = 1, cutoff: float = 7.0
    ):
        super().__init__()
        lmdb_path = Path(os.environ["LMDB_PATH"])
        dataset_path = lmdb_path / f"{protein_id}-{traj_id}-protein.lmdb"
        metadata_path = lmdb_path / f"metadata-{protein_id}-{traj_id}-protein.json"
        map_size = 10_995_116_277_760  # 1 TB
        self.env = lmdb.open(
            dataset_path.__str__(),
            map_size=map_size,
            subdir=False,
            readonly=True,
            lock=False,
        )
        self.lagtime = lagtime
        self._metadata = json.load(open(metadata_path, "r"))
        self.protein_id = protein_id
        self.traj_id = traj_id
        self.length = self._load_length()
        self.z_table = self._load_z_table()
        self.cutoff = cutoff

    def _load_length(self):
        item_key = "__len__".encode()
        with self.env.begin(write=False) as txn:
            data_binary = txn.get(item_key)
            if data_binary is None:
                raise KeyError("Key '__len__' not found in LMDB")
            return pickle.loads(data_binary) - self.lagtime

    def _load_z_table(self):
        item_key = "z_table".encode()
        with self.env.begin(write=False) as txn:
            data_binary = txn.get(item_key)
            if data_binary is None:
                raise KeyError("Key 'z_table' not found in LMDB")
            return pickle.loads(data_binary)

    @property
    def lagtime_ns(self):
        return self.lagtime * self._metadata["lagtime_ns"]

    @property
    def system_selection(self):
        return self._metadata["system_selection"]

    def __len__(self):
        return self.length - self.lagtime

    def _get_lmdb_item(self, idx):
        """Get a specific item from LMDB with caching"""
        if self.env is None:
            raise RuntimeError("LMDB environment not initialized")

        item_key = f"item_{idx}".encode()
        item_lagged_key = f"item_{idx + self.lagtime}".encode()
        with self.env.begin(write=False) as txn:
            data_binary = txn.get(item_key)
            data_lagged_binary = txn.get(item_lagged_key)
            if (data_binary is None) or (data_lagged_binary is None):
                raise KeyError(
                    f"Item at index {idx}/{idx + self.lagtime} not found in LMDB"
                )
        return self.convert_to_pyg(data_binary), self.convert_to_pyg(data_lagged_binary)

    def convert_to_pyg(self, config_binary):
        config = pickle.loads(config_binary)
        pyg_data = _create_dataset_from_configuration(
            config=config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            buffer=0.0,
        )
        return pyg_data

    def __getitem__(self, index):
        result_dict = {}
        if isinstance(index, slice):
            # Handle slice
            index = range(*index.indices(self.__len__()))
            raise NotImplementedError
        elif isinstance(index, (list, tuple)):
            # Handle list of indices
            raise NotImplementedError

        data = self._get_lmdb_item(index)
        result_dict = {
            "item": data[0],
            "item_lag": data[1],
        }
        return result_dict
