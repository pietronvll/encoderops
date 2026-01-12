# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)
import asyncio
import math
import os
import urllib.request
from collections import defaultdict
from dataclasses import asdict
from os.path import join as opj
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from loguru import logger
from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration
from mlcolvar.data.graph.utils import _create_dataset_from_configuration
from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from src.configs import MDCATHDataArgs, TrainerArgs


def load_pdb_list(pdb_list):
    """Load PDB list from a file or return list directly."""
    if isinstance(pdb_list, list):
        return pdb_list
    elif isinstance(pdb_list, str) and os.path.isfile(pdb_list):
        print(f"Reading PDB list from {pdb_list}")
        with open(pdb_list, "r") as file:
            return [line.strip() for line in file]
    elif pdb_list is None:
        return None
    raise ValueError("Invalid PDB list. Please provide a list or a path to a file.")


class MDCATH(Dataset):
    def __init__(self, data_args: MDCATHDataArgs):
        """mdCATH dataset class for MD trajectories with temporal lag support.

        Parameters:
        -----------
        data_args: MDCATHDataArgs
        """
        super().__init__()
        self.url = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/"
        self.data_args = data_args
        self.root = self._parse_datapath(self.data_args.data_path)

        self.root.mkdir(parents=True, exist_ok=True)

        self.source_file = self.data_args.source_file
        self.file_basename = self.data_args.file_basename
        self.lagtime = self.data_args.lagtime
        self.numAtoms = self.data_args.numAtoms
        self.numNoHAtoms = self.data_args.numNoHAtoms
        self.numResidues = self.data_args.numResidues
        self.remove_hydrogen_atoms = self.data_args.remove_hydrogen_atoms

        self.temperatures = self.data_args.temperatures
        if isinstance(self.temperatures, str):
            self.temperatures = [self.temperatures]
        self.skip_frames = self.data_args.skip_frames
        self.pdb_list = load_pdb_list(self.data_args.pdb_list)
        self.min_gyration_radius = self.data_args.min_gyration_radius
        self.max_gyration_radius = self.data_args.max_gyration_radius
        self.alpha_beta_coil = self.data_args.alpha_beta_coil
        self.numFrames = self.data_args.numFrames
        self.solid_ss = self.data_args.solid_ss
        self.cutoff = self.data_args.cutoff_ang

        # Initialize dataset
        self._ensure_source_file()
        self._filter_and_prepare_data()
        self.download()
        self._setup_idx()

        # Initialize z_table once for the entire dataset
        # For now, we'll use common elements up to 100
        self.z_table = AtomicNumberTable(list(range(1, 101)))

        # Calculate total size
        self.total_size_mb = self.calculate_dataset_size()

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self._log_info()
        else:
            self._log_info()

    def _parse_datapath(self, data_path: str | None) -> Path:
        if data_path is None:
            root = os.environ.get("MDCATH_DATA_PATH")
            if root is None:
                raise ValueError(
                    "data_path unspecified, and not found in environment variables."
                )
            else:
                root = Path(root)  # ty:ignore[invalid-assignment]
        else:
            root = Path(data_path)  # ty:ignore[invalid-assignment]
        logger.info(f"Loading data from {root}")
        return root

    def _log_info(self):
        print(f"Total number of domains: {len(self.processed.keys())}")
        print(f"Total number of conformers: {self.num_conformers}")
        print(
            f"Total valid samples (accounting for lagtime={self.lagtime}): {len(self)}"
        )
        print(f"Total size of dataset: {self.total_size_mb} MB")
        print(f"Cutoff: {self.cutoff} angstroms")

    def _ensure_source_file(self):
        """Ensure the source file is downloaded before processing."""
        source_path = self.root / self.source_file
        if not source_path.exists():
            assert self.source_file == "mdcath_source.h5", (
                "Only 'mdcath_source.h5' is supported as source file for download."
            )
            print(f"Downloading source file {self.source_file}")
            urllib.request.urlretrieve(
                opj(self.url, self.source_file), str(source_path)
            )

    def download(self, max_concurrent: int = 4):
        """Download required HDF5 files for selected PDB IDs."""

        overall_progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )

        file_progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )

        progress_group = Group(overall_progress, file_progress)

        overall_task_id = overall_progress.add_task(
            "overall", filename="Overall Progress", total=len(self.processed)
        )

        async def _download_file(pdb_id, semaphore):
            async with semaphore:
                file_name = f"{self.file_basename}_{pdb_id}.h5"
                file_path = self.root / file_name
                if not file_path.exists():
                    if self.file_basename != "mdcath_dataset":
                        raise AssertionError(
                            "Only 'mdcath_dataset' is supported as file_basename for download."
                        )

                    task_id = file_progress.add_task(
                        "download", filename=file_name, total=None
                    )

                    def hook(count, block_size, total_size):
                        file_progress.update(
                            task_id,
                            total=total_size if total_size != -1 else None,
                            completed=count * block_size,
                        )

                    try:
                        await asyncio.to_thread(
                            urllib.request.urlretrieve,
                            opj(self.url, "data", file_name),
                            str(file_path),
                            hook,
                        )
                    finally:
                        file_progress.remove_task(task_id)
                overall_progress.advance(overall_task_id)

        async def _download_all():
            semaphore = asyncio.Semaphore(max_concurrent)
            with Live(progress_group):
                await asyncio.gather(
                    *[
                        _download_file(pdb_id, semaphore)
                        for pdb_id in self.processed.keys()
                    ]
                )

        # Run the asynchronous download
        asyncio.run(_download_all())

    def calculate_dataset_size(self):
        """Calculate total dataset size in MB."""
        total_size_bytes = 0
        for pdb_id in self.processed.keys():
            file_name = f"{self.file_basename}_{pdb_id}.h5"
            file_path = self.root / file_name
            if file_path.exists():
                total_size_bytes += file_path.stat().st_size
        return round(total_size_bytes / (1024 * 1024), 4)

    def _filter_and_prepare_data(self):
        """Filter trajectories based on specified criteria."""
        source_info_path = self.root / self.source_file

        self.processed = defaultdict(list)
        self.num_conformers = 0

        with h5py.File(source_info_path, "r") as file:
            domains = file.keys() if self.pdb_list is None else self.pdb_list

            for pdb_id in domains:
                if pdb_id not in file:
                    continue

                pdb_group = file[pdb_id]

                # Apply atom and residue filters
                if (
                    self.numAtoms is not None
                    and pdb_group.attrs["numProteinAtoms"] > self.numAtoms
                ):
                    continue
                if (
                    self.numResidues is not None
                    and pdb_group.attrs["numResidues"] > self.numResidues
                ):
                    continue
                if (
                    self.numNoHAtoms is not None
                    and "numNoHAtoms" in pdb_group.attrs
                    and pdb_group.attrs["numNoHAtoms"] > self.numNoHAtoms
                ):
                    continue

                self._process_temperatures(pdb_id, pdb_group)

    def _process_temperatures(self, pdb_id: str, pdb_group):
        """Process all temperature/replica combinations for a PDB."""
        for temp in self.temperatures:
            if temp not in pdb_group:
                continue
            for replica in pdb_group[temp].keys():
                self._evaluate_replica(pdb_id, temp, replica, pdb_group)

    def _evaluate_replica(self, pdb_id: str, temp: str, replica: str, pdb_group):
        """Evaluate if a replica meets filtering criteria."""
        replica_group = pdb_group[temp][replica]

        conditions = [
            self.numFrames is not None
            and replica_group.attrs["numFrames"] < self.numFrames,
            self.min_gyration_radius is not None
            and replica_group.attrs["min_gyration_radius"] < self.min_gyration_radius,
            self.max_gyration_radius is not None
            and replica_group.attrs["max_gyration_radius"] > self.max_gyration_radius,
            self._evaluate_structure(pdb_group, temp, replica),
        ]

        if any(conditions):
            return

        # Calculate number of valid frames (accounting for skip_frames and lagtime)
        total_frames = replica_group.attrs["numFrames"]
        num_frames = math.ceil(total_frames / self.skip_frames)

        # We need at least lagtime+1 frames to form pairs
        if num_frames <= self.lagtime:
            return

        self.processed[pdb_id].append((temp, replica, num_frames))
        self.num_conformers += num_frames

    def _evaluate_structure(self, pdb_group, temp: str, replica: str) -> bool:
        """Check if secondary structure meets criteria."""
        if self.solid_ss is None:
            return False

        replica_group = pdb_group[temp][replica]
        alpha = replica_group.attrs["alpha"]
        beta = replica_group.attrs["beta"]
        solid_ss = (alpha + beta) / pdb_group.attrs["numResidues"] * 100
        return solid_ss < self.solid_ss

    def _setup_idx(self):
        """Build index mapping from sample index to (pdb, file, temp, replica, frame)."""
        self.idx = []

        for pdb_id, group_info in self.processed.items():
            file_path = self.root / f"{self.file_basename}_{pdb_id}.h5"

            for temp, replica, num_frames in group_info:
                # Only include frames where we can form valid pairs with lagtime
                valid_frames = num_frames - self.lagtime

                for frame_idx in range(valid_frames):
                    self.idx.append((pdb_id, str(file_path), temp, replica, frame_idx))

    def _load_frame(
        self, file_path: str, pdb_id: str, temp: str, replica: str, frame_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load atomic numbers, coordinates, forces, and box for a specific frame."""
        actual_frame_idx = frame_idx * self.skip_frames
        slice_idxs = np.s_[actual_frame_idx : actual_frame_idx + 1]

        with h5py.File(file_path, "r") as f:
            z = f[pdb_id]["z"][:]
            coords = np.zeros((z.shape[0], 3))
            forces = np.zeros((z.shape[0], 3))

            group = f[f"{pdb_id}/{temp}/{replica}"]
            group["coords"].read_direct(coords, slice_idxs)
            group["forces"].read_direct(forces, slice_idxs)

            # Load box information
            box = group["box"][:]  # shape: (3, 3)

            # coords and forces shape (num_atoms, 3)
            assert coords.shape[0] == forces.shape[0], (
                f"Number of frames mismatch between coords and forces: {group['coords'].shape[0]} vs {group['forces'].shape[0]}"
            )
            assert coords.shape[0] == z.shape[0], (
                f"Number of atoms mismatch between coords and z: {group['coords'].shape[1]} vs {z.shape[0]}"
            )

        return z, coords, forces, box

    def __len__(self) -> int:
        """Return number of valid (frame, frame+lag) pairs."""
        return len(self.idx)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        """Get a pair of configurations separated by lagtime.

        Returns:
        --------
        dict with keys:
            - 'item': PyTorch Geometric Data object for current frame
            - 'item_lag': PyTorch Geometric Data object for lagged frame
        """
        if isinstance(idx, (slice, list, tuple)):
            raise NotImplementedError("Only integer indexing is supported")

        pdb_id, file_path, temp, replica, frame_idx = self.idx[idx]

        data = {}
        for frame_idx_load, key in zip(
            [frame_idx, frame_idx + self.lagtime], ["item", "item_lag"]
        ):
            # Load frame
            z, coords, forces, box = self._load_frame(
                file_path, pdb_id, temp, replica, frame_idx_load
            )
            if self.remove_hydrogen_atoms:
                mask = z != 1
                z = z[mask]
                coords = coords[mask]
                forces = forces[mask]

            # Create Configuration object
            # Note: mdCATH H5 files store coords in Angstroms but box in nanometers
            # Convert box to Angstroms to match coordinate units
            config = Configuration(
                atomic_numbers=z,
                positions=coords,  # H5 stores coords in Angstroms
                cell=box * 10,  # H5 stores box in nm, convert to Angstroms
                pbc=(True, True, True),  # Assuming periodic boundary conditions
                node_labels=forces,  # Using forces as node labels
                graph_labels=None,
                weight=1.0,
                system=None,
                environment=None,
            )

            # Convert Configuration to PyTorch Geometric Data
            pyg_data = _create_dataset_from_configuration(
                config=config,
                z_table=self.z_table,
                cutoff=self.cutoff,
                buffer=0.0,
            )

            # Add metadata
            pyg_data.info = f"{pdb_id}_{temp}_{replica}_{frame_idx_load}"
            pyg_data.pdb_id = pdb_id
            pyg_data.temp = temp
            pyg_data.replica = replica

            data[key] = pyg_data
        return data

    def get_trajectory_info(self) -> Dict:
        """Get information about all trajectories in the dataset."""
        info = {
            "num_domains": len(self.processed),
            "num_trajectories": sum(len(v) for v in self.processed.values()),
            "domains": {},
        }

        for pdb_id, group_info in self.processed.items():
            info["domains"][pdb_id] = [
                {"temp": temp, "replica": replica, "num_frames": num_frames}
                for temp, replica, num_frames in group_info
            ]

        return info


class MDCATHDataModule(LightningDataModule):
    def __init__(
        self,
        args: TrainerArgs,
        data_args: MDCATHDataArgs,
        num_workers: int,
    ):
        super().__init__()
        self.args = args
        self.data_args = data_args
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = MDCATH(
            data_args=self.data_args,
        )

    def state_dict(self):
        state = {"data_args": asdict(self.data_args), "num_workers": self.num_workers}
        return state

    def load_state_dict(self, state_dict):
        self.data_args = MDCATHDataArgs(**state_dict["data_args"])
        self.num_workers = state_dict["num_workers"]

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
