import json
import os
import pickle
import cftime
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import mdtraj
import numpy as np
import torch.distributed
import xarray as xr
from lightning import LightningDataModule
from loguru import logger
from mlcolvar.data.graph.atomic import AtomicNumberTable
from mlcolvar.data.graph.utils import _create_dataset_from_configuration
from mlcolvar.utils.io import (
    _configures_from_trajectory,
    _names_from_top,
    _z_table_from_top,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

import lmdb
from src.configs import (
    CalixareneDataArgs,
    DESRESDataArgs,
    Lorenz63DataArgs,
    SSTDataArgs,
    TrainerArgs,
)
from src.utils import FastTensorDataLoader


def traj_to_confs(traj: mdtraj.Trajectory, system_selection: str | None = None):
    configs = _configures_from_trajectory(traj, system_selection=system_selection)
    z_table = _z_table_from_top([traj.top])
    atom_names = _names_from_top([traj.top])
    return configs, z_table, atom_names


def mdtraj_load(trajectory_files: list[str], top: str, stride: int = 1000):
    traj = mdtraj.load(trajectory_files, top=top, stride=stride)
    traj.top = mdtraj.core.trajectory.load_topology(top)
    return traj


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
        self.data_path = self.parse_datapath(self.data_args.data_path)
        self.num_workers = num_workers

    def parse_datapath(self, data_path):
        if data_path is None:
            data_path = Path(os.environ["DATA_PATH"])
        else:
            data_path = Path(data_path)
        return data_path  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.

    def setup(self, stage):
        self.dataset = DESRESDataset(
            protein_id=self.data_args.protein_id,
            data_path=self.data_path,
            traj_id=self.data_args.traj_id,
            lagtime=self.data_args.lagtime,
            cutoff_ang=self.data_args.cutoff_ang,
        )

    def state_dict(self):
        state = {"data_args": asdict(self.data_args), "num_workers": self.num_workers}
        return state

    def load_state_dict(self, state):
        self.data_args = DESRESDataArgs(**state["data_args"])
        self.num_workers = state["num_workers"]
        self.data_path = self.parse_datapath(self.data_args.data_path)

    def train_dataloader(self):
        return PyGDataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


class DESRESDataset(Dataset):
    def __init__(
        self,
        protein_id: str,
        data_path: Path,
        traj_id: int = 0,
        lagtime: int = 1,
        cutoff_ang: float = 7.0,
    ):
        super().__init__()
        dataset_path = data_path / f"{protein_id}-{traj_id}-protein.lmdb"
        metadata_path = data_path / f"metadata-{protein_id}-{traj_id}-protein.json"
        map_size = 10_995_116_277_760  # 1 TB
        print(dataset_path)
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
        else:
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


class CalixareneDataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainerArgs,
        data_args: CalixareneDataArgs,
        num_workers: int,
    ):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.data_path = self.parse_datapath(self.data_args.data_path)
        self.num_workers = num_workers

    def parse_datapath(self, data_path):
        if data_path is None:
            data_path = Path(os.environ["DATA_PATH"])
        else:
            data_path = Path(data_path)
        return data_path  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.

    def prepare_data(self):
        if not (self.data_path / "calixarene").exists():
            logger.info("Downloading Calixarene dataset")
            import huggingface_hub as hf

            hf.snapshot_download(
                repo_id="pnovelli/encoderops",
                allow_patterns="calixarene/**",
                repo_type="dataset",
                local_dir=self.data_path,
            )

    def setup(self, stage):
        datasets = []
        atomic_numbers = []
        for molecule_id in self.data_args.molecule_ids:
            for traj_id in self.data_args.traj_ids:
                ds = CalixareneDataset(
                    molecule_id=molecule_id,
                    data_path=self.data_path,
                    traj_id=traj_id,
                    lagtime=self.data_args.lagtime,
                    cutoff_ang=self.data_args.cutoff_ang,
                    keep_mdtraj=self.data_args.keep_mdtraj,
                )
                datasets.append(ds)
                atomic_numbers.extend(ds.z_table.zs)
        atomic_numbers = sorted(list(set(atomic_numbers)))
        z_table = AtomicNumberTable(atomic_numbers)
        molecule_ids = "-".join(self.data_args.molecule_ids)
        for ds in datasets:
            ds.z_table = z_table
        self.dataset = ConcatDataset(datasets)
        self.dataset.lagtime = self.data_args.lagtime
        self.dataset.lagtime_ns = self.dataset.datasets[0].lagtime_ns
        self.dataset.z_table = z_table
        self.dataset.molecule_ids = molecule_ids

    def state_dict(self):
        state = {"data_args": asdict(self.data_args), "num_workers": self.num_workers}
        return state

    def load_state_dict(self, state):
        self.data_args = CalixareneDataArgs(**state["data_args"])
        self.num_workers = state["num_workers"]
        self.data_path = self.parse_datapath(self.data_args.data_path)

    def train_dataloader(self):
        return PyGDataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


class CalixareneDataset(Dataset):
    def __init__(
        self,
        molecule_id: str,
        data_path: Path,
        traj_id: int = 0,
        lagtime: int = 1,
        cutoff_ang: float = 7.0,
        system_selection: str | None = "all and not type H",
        keep_mdtraj: bool = False,
    ):
        super().__init__()
        traj_path = data_path / f"calixarene/{molecule_id}/traj/traj_com_{traj_id}.trr"
        top_path = data_path / f"calixarene/{molecule_id}/data/no_water.gro"

        self.lagtime = lagtime
        self.molecule_id = molecule_id
        self.traj_id = traj_id
        self.cutoff = cutoff_ang
        traj = mdtraj_load([traj_path], top_path, 1)

        if system_selection is not None:
            system_atoms = traj.top.select(system_selection)
            traj = traj.atom_slice(system_atoms)
        if keep_mdtraj:
            self.traj = traj
        self.configs, self.z_table, _ = traj_to_confs(traj)
        self._metadata = {
            "system_selection": system_selection,
            "lagtime_ns": 0.001,
        }

        is_rank_0 = True
        if torch.distributed.is_initialized():
            is_rank_0 = torch.distributed.get_rank() == 0

        if is_rank_0:
            logger.info(
                f"Loaded {self.molecule_id}-{self.traj_id} | lagtime {self.lagtime_ns} ns | cutoff {self.cutoff} angs"
            )

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


class Lorenz63DataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainerArgs,
        data_args: Lorenz63DataArgs,
        num_workers: int = 4,
    ):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.data_path = self.parse_datapath(
            self.data_args.data_path
        )  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.
        self.num_workers = num_workers

    def parse_datapath(self, data_path):
        if data_path is None:
            data_path = Path(os.environ["DATA_PATH"])
        else:
            data_path = Path(data_path)
        return data_path  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.

    def prepare_data(self):
        if not (self.data_path / "lorenz63/lorenz63_dataset.nc").exists():
            logger.info("Downloading Lorenz63 dataset")
            import huggingface_hub as hf

            hf.hf_hub_download(
                repo_id="pnovelli/encoderops",
                filename="lorenz63/lorenz63_dataset.nc",
                repo_type="dataset",
                local_dir=self.data_path,
            )

    def state_dict(self):
        state = {"data_args": asdict(self.data_args), "num_workers": self.num_workers}
        return state

    def load_state_dict(self, state):
        self.data_args = Lorenz63DataArgs(**state["data_args"])
        self.num_workers = state["num_workers"]
        self.data_path = self.parse_datapath(self.data_args.data_path) 

    def setup(self, stage):
        self.train_dataset = Lorenz63Dataset(
            lagtime=self.data_args.lagtime,
            history_len=self.data_args.history_len,
            data_path=self.data_path / "lorenz63/lorenz63_dataset.nc",
            split="train",
        )
        self.val_dataset = Lorenz63Dataset(
            lagtime=self.data_args.lagtime,
            history_len=self.data_args.history_len,
            data_path=self.data_path / "lorenz63/lorenz63_dataset.nc",
            split="val",
        )
        self.test_dataset = Lorenz63Dataset(
            lagtime=self.data_args.lagtime,
            history_len=self.data_args.history_len,
            data_path=self.data_path / "lorenz63/lorenz63_dataset.nc",
            split="test",
        )

    def train_dataloader(self):
        idx_X_arr, idx_Y_arr = zip(*self.train_dataset.indices)  # shape: (N, history_len+1)
        idx_X_arr = np.array(idx_X_arr)
        idx_Y_arr = np.array(idx_Y_arr)

        X = torch.from_numpy(self.train_dataset.data[idx_X_arr]).float()
        Y = torch.from_numpy(self.train_dataset.data[idx_Y_arr]).float()

        X = X.reshape((-1, *X.shape[2:]))  # (N, H, dim)
        Y = Y.reshape((-1, *Y.shape[2:]))  # (N, H, dim)
        return FastTensorDataLoader(
            X,
            Y,
            batch_size=self.args.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        idx_X_arr, idx_Y_arr = zip(*self.val_dataset.indices)  # shape: (N, history_len+1)
        idx_X_arr = np.array(idx_X_arr)
        idx_Y_arr = np.array(idx_Y_arr)

        X = torch.from_numpy(self.val_dataset.data[idx_X_arr]).float()
        Y = torch.from_numpy(self.val_dataset.data[idx_Y_arr]).float()

        X = X.reshape((-1, *X.shape[2:]))  # (N, H, dim)
        Y = Y.reshape((-1, *Y.shape[2:]))  # (N, H, dim)
        return FastTensorDataLoader(
            X,
            Y,
            batch_size=len(self.val_dataset),
            shuffle=False
        )

    def test_dataloader(self):
        idx_X_arr, idx_Y_arr = zip(*self.test_dataset.indices)  # shape: (N, history_len+1)
        idx_X_arr = np.array(idx_X_arr)
        idx_Y_arr = np.array(idx_Y_arr)

        X = torch.from_numpy(self.test_dataset.data[idx_X_arr]).float()
        Y = torch.from_numpy(self.test_dataset.data[idx_Y_arr]).float()

        X = X.reshape((-1, *X.shape[2:]))  # (N, H, dim)
        Y = Y.reshape((-1, *Y.shape[2:]))  # (N, H, dim)
        return FastTensorDataLoader(
            X,
            Y,
            batch_size=len(self.test_dataset),
            shuffle=False
        )


class SSTDataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainerArgs,
        data_args: SSTDataArgs,
        num_workers: int,
    ):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.data_path = self.parse_datapath(
            self.data_args.data_path
        )  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.
        self.num_workers = num_workers

    def parse_datapath(self, data_path):
        if data_path is None:
            data_path = Path(os.environ["DATA_PATH"])
        else:
            data_path = Path(data_path)
        return data_path  # Preprocessed offline for the moment. Maybe move to prepare_data if asked to.

    def prepare_data(self):
        if self.data_args.data_source == "ORAS5":
            data_source = "SST/sst_monthly.nc"
        elif self.data_args.data_source == "CESM":
            data_source = "SST/cesm_sst_regridded_1.5deg_850-2005.nc"
        if not (self.data_path / data_source).exists():
            logger.info(f"Downloading {self.data_args.data_source} dataset")
            import huggingface_hub as hf

            hf.hf_hub_download(
                repo_id="CSML-IIT/encoderops",
                filename=data_source,
                repo_type="dataset",
                local_dir=self.data_path,
            )

    def state_dict(self):
        state = {"data_args": asdict(self.data_args), "num_workers": self.num_workers}
        return state

    def load_state_dict(self, state):
        self.data_args = Lorenz63DataArgs(**state["data_args"])
        self.num_workers = state["num_workers"]
        self.data_path = self.parse_datapath(self.data_args.data_path)

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                augmentations=self.data_args.augmentations,
                random_roll=self.data_args.random_roll,
                vertical_flip_probability=self.data_args.vertical_flip_probability,
                data_path=self.data_path,
                land_mask=self.data_args.mask,
                split="train",
                data_source=self.data_args.data_source,
            )
            self.val_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                data_path=self.data_path,
                augmentations=False,
                land_mask=self.data_args.mask,
                split="val",
                data_source=self.data_args.data_source,
            )
        elif stage == "test":
            self.test_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                data_path=self.data_path,
                augmentations=False,
                land_mask=self.data_args.mask,
                split="test",
                data_source=self.data_args.data_source,
            )
        elif stage == "full":
            self.full_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                data_path=self.data_path,
                augmentations=False,
                land_mask=self.data_args.mask,
                split="full",
                data_source=self.data_args.data_source,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class SSTAugmentations(torch.nn.Module):
    def __init__(
        self, random_roll: bool = True, vertical_flip_probability: float = 0.5
    ):
        super().__init__()
        self.vertical_flip_probability = vertical_flip_probability
        self.random_roll = random_roll

    def forward(self, image: torch.Tensor, image_lag: torch.Tensor):
        if self.random_roll:
            W: int = image.shape[-1]
            shift = torch.randint(W - 1, (1,)).item()

            image = torch.roll(image, shifts=shift, dims=-1)
            image_lag = torch.roll(image_lag, shifts=shift, dims=-1)

        vertical_flip = torch.rand(1).item() < self.vertical_flip_probability
        if vertical_flip:
            image = torch.flip(image, dims=(-2,))
            image_lag = torch.flip(image_lag, dims=(-2,))
        return image, image_lag


class SSTDataset(Dataset):
    def __init__(
        self,
        lagtime: int = 1,
        history_len: int = 0,
        augmentations: bool = False,
        random_roll: bool = True,
        vertical_flip_probability: float = 0.5,
        data_path: str | Path | None = None,
        split: Literal["train", "val", "test", "full"] = "train",
        land_mask: bool = True,
        data_source: Literal["ORAS5", "CESM"] = "ORAS5",
        detrend: bool = False,
    ):
        # If data_path is not specified, read it from the environment variable "DATA_PATH"
        self.lagtime = lagtime
        self.history_len = history_len
        self.split = split
        self.augmentations = augmentations
        self.transforms = SSTAugmentations(
            random_roll=random_roll, vertical_flip_probability=vertical_flip_probability
        )
        self.land_mask = land_mask
        self.data_source = data_source

        if data_path is None:
            try:
                data_path = os.environ["DATA_PATH"]
            except KeyError:
                raise ValueError(
                    "data_path environment variable is not set, and data_path is not provided."
                )

        if self.data_source == "ORAS5":
            dataset_path = Path(data_path) / "SST/sst_monthly.nc"
            split_years = {
                "full": list(range(1979, 2024)),
                "train": list(range(1979, 2017)),
                "val": list(range(2017, 2024)),
                "test": list(range(2017, 2024))
                }
        elif self.data_source == "CESM":
            dataset_path = Path(data_path) / "SST/cesm_sst_regridded_1.5deg_850-2005.nc"
            split_years = {
                "full": list(range(850, 2006)),
                "train": list(range(850, 1900)),
                "val": list(range(1900, 2006)),
                "test": list(range(1900, 2006))
                }
        else:
            raise ValueError(f"Unsupported data_source: {self.data_source}")

        ds = xr.open_dataset(dataset_path, use_cftime=True)
        anomalies = self._compute_oni(
            ds,
            method="centered",
            output="anomalies",
            latitude_range=(180.0, -180.0),
            longitude_range=(0.0, 360.0),
            detrend=detrend,
        )
        oni = self._compute_oni(ds, method="centered")

        if split == "full":
            self.ds = ds
            self.anomalies = anomalies
            self.oni = oni
        else:
            time_mask = ds.time.dt.year.isin(split_years[split])
            self.ds = ds.sel(time=time_mask)
            self.anomalies = anomalies.sel(time=time_mask)
            self.oni_full = oni
            self.oni = oni.sel(time=time_mask)

        self.data = np.squeeze(self.anomalies.SST.values)

        self.time = self.ds.time.values
        if self.land_mask:
            self.anomalies["mask"] = (anomalies.SST != 0).astype("float32")
            self.mask = self.anomalies.mask.values[0]
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.info(f"Dataset loaded with {self.num_samples} samples.")
        else:
            logger.info(f"Dataset loaded with {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    @property
    def num_samples(self):
        return len(self.ds.time) - self.history_len - self.lagtime

    def _compute_oni(
        self,
        sst_ds: xr.DataArray | xr.Dataset,
        method: str = "fixed",
        output: str = "oni",
        fixed_base_period: tuple = (1991, 2020),
        latitude_range: tuple[float, float] | None = None,
        longitude_range: tuple[float, float] | None = None,
        detrend: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """
        Compute ONI index or SST anomalies based on a given method.

        Parameters:
        ----------
        method : str
            'fixed' for a fixed 30-year base period, 'centered' for moving base periods.
        output : str
            'oni' for ONI (with 3-month running mean), 'anomalies' for raw anomalies.
        fixed_base_period : tuple
            Start and end date of the fixed base period (only if method='fixed').
        centered_periods : list
            List of tuples for (5-year block start, 5-year block end, base period string).
        latitude_range : tuple
            Latitude range to average over (default Niño 3.4: (5, -5)).
        longitude_range : tuple
            Longitude range to average over (default Niño 3.4: (190, 240)).

        Returns:
        -------
        xr.DataArray
            ONI index (with rolling mean) or SST anomalies time series.
        """

        if latitude_range is None:
            latitude_range = (5, -5)
        if longitude_range is None:
            longitude_range = (190, 240)

        # Step 0: Prepare
        sst_region = sst_ds.sel(
            latitude=slice(*latitude_range), longitude=slice(*longitude_range)
        )

        # Latitude weights for area mean
        weights = np.cos(np.deg2rad(sst_region.latitude))
        weights.name = "weights"

        # Fixed climatology method
        if method == "fixed":
            base_start, base_end = fixed_base_period
            base_start = cftime.DatetimeNoLeap(int(base_start), 1, 1)
            base_end = cftime.DatetimeNoLeap(int(base_end), 12, 31)
            # Select base period
            sst_base = sst_region.sel(time=slice(base_start, base_end))
            climatology = sst_base.groupby("time.month").mean("time")

            # Compute anomalies
            anomalies = sst_region.groupby("time.month") - climatology

        # Centered climatology method
        elif method == "centered":
            centered_periods = self._generate_centered_periods(sst_ds)
            anomalies_list = []

            for start_year, end_year, base_start, base_end in centered_periods:
                base_start = cftime.DatetimeNoLeap(base_start, 1, 1)
                base_end = cftime.DatetimeNoLeap(base_end, 12, 31)
                # Select base climatology
                sst_base = sst_region.sel(time=slice(base_start, base_end))
                climatology = sst_base.groupby("time.month").mean("time")

                # Select block data
                start_year = cftime.DatetimeNoLeap(start_year, 1, 1)
                end_year = cftime.DatetimeNoLeap(end_year, 12, 31)
                sst_block = sst_region.sel(time=slice(start_year, end_year))

                # Compute anomalies
                anomalies_block = sst_block.groupby("time.month") - climatology

                anomalies_list.append(anomalies_block)

            anomalies = xr.concat(anomalies_list, dim="time")

        else:
            raise ValueError("Method must be 'fixed' or 'centered'.")

        # Step 4: Return output
        if output == "anomalies":
            if detrend:
                def detrend_along_time(da: xr.DataArray) -> xr.DataArray:
                    from scipy.signal import detrend
                    return xr.apply_ufunc(
                        detrend,
                        da,
                        input_core_dims=[["time"]],
                        output_core_dims=[["time"]],
                        vectorize=True,
                        dask="parallelized",
                        output_dtypes=[da.SST.dtype],
                    )
                anomalies = detrend_along_time(anomalies)
            anomalies = anomalies.drop_vars("month")
            return anomalies
        elif output == "oni":
            # Spatial mean (area weighted)
            anomaly_mean = anomalies.weighted(weights).mean(
                dim=["latitude", "longitude"]
            )
            oni = anomaly_mean.rolling(time=3, center=True, min_periods=1).mean()
            return oni
        else:
            raise ValueError("Output must be 'oni' or 'anomalies'.")

    def _generate_centered_periods(
        self,
        sst_ds: xr.Dataset,
        block_size: int = 5,
        climatology_window: int = 30
    ) -> list[tuple[int, int, str]]:
        """
        Generate (start, end, base_period) tuples for centered ONI computation.
        
        Parameters
        ----------
        sst_ds : xr.DataArray
            SST dataset.
        block_size : int
            Width of the target block (in years).
        climatology_window : int
            Width of the base period (in years).

        Returns
        -------
        List of tuples: (block_start_year, block_end_year, base_period_str)
        """
        years = np.unique(sst_ds.time.dt.year.values)
        start_year = years.min()
        end_year = years.max()

        half_clim = climatology_window // 2
        blocks = []

        for mid_year in range(start_year, end_year, block_size):
            block_start = mid_year
            block_end = mid_year + block_size -1
            base_start = mid_year - half_clim + 1
            base_end = mid_year + half_clim
            blocks.append((int(block_start), int(block_end), int(base_start), int(base_end)))

        return blocks

    def _load_sample(self, idx: int):
        x_selectors = [idx - h + self.history_len for h in range(self.history_len + 1)]
        y_selectors = [x_id + self.lagtime for x_id in x_selectors]
        x = self.data[x_selectors]
        y = self.data[y_selectors]

        if self.land_mask:
            # mask = self.mask[..., 1:, 1:]  # skip empty vertical line at x = 0, y = 0
            mask = self.mask
            x = np.concatenate((x, mask), axis=0)
            y = np.concatenate((y, mask), axis=0)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.augmentations:
            x, y = self.transforms(x, y)

        times = self.time[x_selectors]
        times_lag = self.time[y_selectors]

        return (
            x,
            y,
            times,
            times_lag,
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, (list, tuple)):
            raise NotImplementedError

        if idx < 0:
            idx = len(self) + idx

        if idx >= len(self):
            raise IndexError("Index out of range")
        x, y, t, t_lag = self._load_sample(idx)
        return {
            "x": x,
            "y": y,
            "time": [str(t_i) for t_i in t],
            "time_lag": [str(t_i) for t_i in t_lag],
        }


class Lorenz63Dataset(Dataset):
    def __init__(
        self,
        lagtime: int = 1,
        history_len: int = 0,
        data_path: str | Path | None = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        # If data_path is not specified, read it from the environment variable "DATA_PATH"
        self.lagtime = lagtime
        self.history_len = history_len
        self.split = split

        if data_path is None:
            try:
                data_path = os.environ["DATA_PATH"]
                data_path = Path(data_path) / "lorenz63/lorenz63_dataset.nc"
            except KeyError:
                raise ValueError(
                    "data_path environment variable is not set, and data_path is not provided."
                )
        ds = xr.open_dataset(data_path, engine="netcdf4")
        self.ds = ds.sel(time=ds.split == split)
        self.data = self.ds["trajectory"].values.astype("float32")
        self.time = self.ds["time"].values.astype("float32")

        # Precompute valid (x, y) index ranges
        self.indices = [
            ([(i - h + self.history_len) for h in range(self.history_len + 1)],
             [(i - h + self.history_len + self.lagtime) for h in range(self.history_len + 1)])
            for i in range(len(self.ds.time) - self.history_len - self.lagtime)
        ]

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.info(
                    f"Dataset loaded with {self.num_samples} samples and {self.num_variables} variables."
                )
        else:
            logger.info(
                f"Dataset loaded with {self.num_samples} samples and {self.num_variables} variables."
            )

    def __len__(self):
        return len(self.indices)

    @property
    def num_variables(self):
        return self.ds.sizes["dim"]

    @property
    def num_samples(self):
        return len(self.ds.time) - self.history_len - self.lagtime

    def _load_sample(self, idx: int):
        x_selectors, y_selectors = self.indices[idx]
        x = self.data[x_selectors]
        y = self.data[y_selectors]

        x = torch.from_numpy(x.reshape((-1, *x.shape[2:])))
        y = torch.from_numpy(y.reshape((-1, *y.shape[2:])))
        return x, y

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, (list, tuple)):
            raise NotImplementedError

        x, y = self._load_sample(idx)
        return x, y
    