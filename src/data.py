import bisect
import json
import os
import pickle
from pathlib import Path

import mdtraj
import torch.distributed
from lightning import LightningDataModule
from loguru import logger
from mlcolvar.data.graph.utils import _create_dataset_from_configuration
from mlcolvar.utils.io import (
    _configures_from_trajectory,
    _names_from_top,
    _z_table_from_top,
)
from torch.utils.data import ConcatDataset, Dataset
from torch_geometric.loader import DataLoader

import lmdb
from src.configs import DESRESDataArgs, TrainerArgs


class DESRESDataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainerArgs,
        data_args: DESRESDataArgs,
        num_workers: int,
    ):
        super().__init__()
        self.args = args
        self.data_args = data_args
        if self.data_args.lmdb_path is None:
            lmdb_path = Path(os.environ["LMDB_PATH"])
        else:
            lmdb_path = Path(self.data_args.lmdb_path)
        self.lmdb_path = lmdb_path  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = DESRESDataset(
            protein_id=self.data_args.protein_id,
            lmdb_path=self.lmdb_path,
            traj_id=self.data_args.traj_id,
            lagtime=self.data_args.lagtime,
            cutoff_ang=self.data_args.cutoff_ang,
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


class DESRESDataset(Dataset):
    def __init__(
        self,
        protein_id: str,
        lmdb_path: Path,
        traj_id: int = 0,
        lagtime: int = 1,
        cutoff_ang: float = 7.0,
    ):
        super().__init__()
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
        self.cutoff = cutoff_ang
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.info(
                    f"Loaded {self.protein_id}-{self.traj_id} | lagtime {self.lagtime_ns} ns | cutoff {self.cutoff} angs"
                )

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


def traj_to_confs(traj: mdtraj.Trajectory, system_selection: str | None = None):
    configs = _configures_from_trajectory(traj, system_selection=system_selection)
    z_table = _z_table_from_top([traj.top])
    atom_names = _names_from_top([traj.top])
    return configs, z_table, atom_names


def mdtraj_load(trajectory_files: list[str], top: str, stride: int = 1000):
    traj = mdtraj.load(trajectory_files, top=top, stride=stride)
    traj.top = mdtraj.core.trajectory.load_topology(top)
    return traj


class CalixareneDataset(Dataset):
    def __init__(
        self,
        protein_id: str,  # I know it is not a protein, but I'm inheriting this class from DESRES data.
        traj_id: int = 0,
        lagtime: int = 1,
        cutoff: float = 7.0,
        system_selection: str | None = "all and not type H",
        _keep_mdtraj: bool = False,
    ):
        super().__init__()
        traj_path = (
            Path(os.environ["CALIX_PATH"]) / f"{protein_id}/traj/traj_com_{traj_id}.trr"
        )
        top_path = Path(os.environ["CALIX_PATH"]) / f"{protein_id}/data/no_water.gro"

        self.lagtime = lagtime
        self.protein_id = protein_id
        self.traj_id = traj_id
        self.cutoff = cutoff
        traj = mdtraj_load([traj_path], top_path, 1)
        if system_selection is not None:
            system_atoms = traj.top.select(system_selection)
            logger.info(f"System selection: {system_selection}")
            traj = traj.atom_slice(system_atoms)
        if _keep_mdtraj:
            self.traj = traj
        self.configs, self.z_table, _ = traj_to_confs(traj)
        self._metadata = {
            "system_selection": system_selection,
            "lagtime_ns": 0.001,
        }

    @property
    def lagtime_ns(self):
        return self.lagtime * self._metadata["lagtime_ns"]

    @property
    def system_selection(self):
        return self._metadata["system_selection"]

    def __len__(self):
        return len(self.configs) - self.lagtime

    def _get_item(self, idx):
        config = self.configs[idx]
        config_lagged = self.configs[idx + self.lagtime]

        return self.convert_to_pyg(config), self.convert_to_pyg(config_lagged)

    def convert_to_pyg(self, config):
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

        data = self._get_item(index)
        result_dict = {
            "item": data[0],
            "item_lag": data[1],
        }
        return result_dict


class ConcatDESRES(ConcatDataset):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
