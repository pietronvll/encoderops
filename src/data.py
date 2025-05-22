import json
import os
import pickle
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
            data_path=self.data_path,
            split="train",
        )
        self.val_dataset = Lorenz63Dataset(
            lagtime=self.data_args.lagtime,
            history_len=self.data_args.history_len,
            data_path=self.data_path,
            split="val",
        )
        self.test_dataset = Lorenz63Dataset(
            lagtime=self.data_args.lagtime,
            history_len=self.data_args.history_len,
            data_path=self.data_path,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataloader,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=self.num_workers,
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
        if not (self.data_path / "SST/sst_monthly.nc").exists():
            logger.info("Downloading Temperature dataset")
            import huggingface_hub as hf

            hf.hf_hub_download(
                repo_id="pnovelli/encoderops",
                filename="SST/sst_monthly.nc",
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
                data_path=self.data_path,
                split="train",
            )
            self.val_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                data_path=self.data_path,
                split="val",
            )
        elif stage == "test":
            self.full_dataset = SSTDataset(
                lagtime=self.data_args.lagtime,
                history_len=self.data_args.history_len,
                data_path=self.data_path,
                split="full",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
        )


class SSTDataset(Dataset):
    def __init__(
        self,
        lagtime: int = 1,
        history_len: int = 0,
        data_path: str | Path | None = None,
        split: Literal["train", "val", "full"] = "train",
    ):
        # If data_path is not specified, read it from the environment variable "DATA_PATH"
        self.lagtime = lagtime
        self.history_len = history_len
        self.split = split

        if data_path is None:
            try:
                data_path = os.environ["DATA_PATH"]
            except KeyError:
                raise ValueError(
                    "data_path environment variable is not set, and data_path is not provided."
                )

        dataset_path = Path(data_path) / "SST/sst_monthly.nc"
        ds = xr.open_dataset(dataset_path)
        ds = self._compute_oni(
            ds,
            method="fixed",
            output="anomalies",
            latitude_range=(180.0, -180.0),
            longitude_range=(0.0, 360.0),
        )
        oni = self._compute_oni(ds, method="fixed")
        oni_full = self._compute_oni(
            ds,
            method="fixed",
            output="oni",
            latitude_range=(180.0, -180.0),
            longitude_range=(0.0, 360.0),
        )
        split_years = {"train": list(range(1979, 2017)), "val": list(range(2017, 2024))}
        if split == "full":
            self.ds = ds
            self.oni = oni
            self.oni_full = oni_full
        else:
            time_mask = ds.time.dt.year.isin(split_years[split])
            self.ds = ds.sel(time=time_mask)
            self.oni = oni.sel(time=time_mask)
            self.oni_full = oni_full.sel(time=time_mask)
        self.data = np.squeeze(self.ds.__xarray_dataarray_variable__.values)
        self.time = self.ds.time.values
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
        sst_ds: xr.DataArray,
        method: str = "fixed",
        output: str = "oni",
        fixed_base_period: tuple = ("1991-01-01", "2020-12-31"),
        centered_periods: list | None = None,
        latitude_range: tuple[float, float] | None = None,
        longitude_range: tuple[float, float] | None = None,
    ) -> xr.DataArray:
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

            # Select base period
            sst_base = sst_region.sel(time=slice(base_start, base_end))
            climatology = sst_base.groupby("time.month").mean("time")

            # Compute anomalies
            anomalies = sst_region.groupby("time.month") - climatology

        # Centered climatology method
        elif method == "centered":
            if centered_periods is None:
                centered_periods = [
                    (1979, 1983, "1964-1993"),
                    (1984, 1988, "1969-1998"),
                    (1989, 1993, "1974-2003"),
                    (1994, 1998, "1979-2008"),
                    (1999, 2003, "1984-2013"),
                    (2004, 2008, "1989-2018"),
                    (2009, 2013, "1994-2023"),
                    (2014, 2018, "1999-2028"),  # Note: beyond your data, careful here
                    (2019, 2023, "2004-2033"),  # Note: beyond your data
                ]

            anomalies_list = []

            for start_year, end_year, base_period in centered_periods:
                base_start, base_end = base_period.split("-")

                # Select base climatology
                sst_base = sst_region.sel(
                    time=slice(f"{base_start}-01-01", f"{base_end}-12-31")
                )
                climatology = sst_base.groupby("time.month").mean("time")

                # Select block data
                sst_block = sst_region.sel(
                    time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
                )

                # Compute anomalies
                anomalies_block = sst_block.groupby("time.month") - climatology

                anomalies_list.append(anomalies_block)

            anomalies = xr.concat(anomalies_list, dim="time")

        else:
            raise ValueError("Method must be 'fixed' or 'centered'.")

        # Step 4: Return output
        if output == "anomalies":
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

    def _load_sample(self, idx: int):
        x_selectors = [idx - h + self.history_len for h in range(self.history_len + 1)]
        y_selectors = [x_id + self.lagtime for x_id in x_selectors]
        x = self.data[x_selectors]
        y = self.data[y_selectors]

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

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
            except KeyError:
                raise ValueError(
                    "data_path environment variable is not set, and data_path is not provided."
                )

        dataset_path = Path(data_path) / "lorenz63/lorenz63_dataset.nc"
        ds = xr.open_dataset(dataset_path)
        self.ds = ds.sel(time=ds.split == split)
        self.data = ds["trajectory"].values
        self.time = ds["time"].values
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
        return self.num_samples

    @property
    def num_variables(self):
        return self.ds.sizes["dim"]

    @property
    def num_samples(self):
        return len(self.ds.time) - self.history_len - self.lagtime

    def _load_sample(self, idx: int):
        x_selectors = [idx - h + self.history_len for h in range(self.history_len + 1)]
        y_selectors = [x_id + self.lagtime for x_id in x_selectors]
        x = self.data[x_selectors]
        y = self.data[y_selectors]

        x = x.reshape((-1, *x.shape[2:]))
        y = y.reshape((-1, *y.shape[2:]))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y, str(self.time[idx + self.history_len])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, (list, tuple)):
            raise NotImplementedError

        x, y, t = self._load_sample(idx)
        return {"x": x, "y": y, "time": t}
