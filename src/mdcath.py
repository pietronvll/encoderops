# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import math
import os
import urllib.request
from collections import defaultdict
from os.path import join as opj
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_pdb_list(pdb_list):
    """Load PDB list from a file or return list directly."""
    if isinstance(pdb_list, list):
        return pdb_list
    elif isinstance(pdb_list, str) and os.path.isfile(pdb_list):
        print(f"Reading PDB list from {pdb_list}")
        with open(pdb_list, "r") as file:
            return [line.strip() for line in file]
    raise ValueError("Invalid PDB list. Please provide a list or a path to a file.")


class MDCATH(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        lagtime: int = 1,
        source_file: str = "mdcath_source.h5",
        file_basename: str = "mdcath_dataset",
        numAtoms: Optional[int] = 5000,
        numNoHAtoms: Optional[int] = None,
        numResidues: Optional[int] = 1000,
        temperatures: List[str] = None,
        skip_frames: int = 1,
        pdb_list: Optional[Union[List[str], str]] = None,
        min_gyration_radius: Optional[float] = None,
        max_gyration_radius: Optional[float] = None,
        alpha_beta_coil: Optional[Tuple] = None,
        solid_ss: Optional[float] = None,
        numFrames: Optional[int] = None,
        cutoff_ang: float = 7.0,
    ):
        """mdCATH dataset class for MD trajectories with temporal lag support.

        Parameters:
        -----------
        root: str or Path
            Root directory where the dataset should be stored.
        lagtime: int
            Number of frames between current and lagged observation. Default is 1.
        source_file: str
            Name of the source file with protein structure information. Default is "mdcath_source.h5".
        file_basename: str
            Base name of the hdf5 files. Default is "mdcath_dataset".
        numAtoms: int
            Max number of atoms in the protein structure.
        numNoHAtoms: int
            Max number of non-hydrogen atoms. Default is None.
        numResidues: int
            Max number of residues in the protein structure.
        temperatures: list
            List of temperatures (in Kelvin) to include. Default is ["348"].
            Available: ['320', '348', '379', '413', '450']
        skip_frames: int
            Number of frames to skip in the trajectory. Default is 1.
        pdb_list: list or str
            List of PDB IDs or path to file with PDB IDs. If None, all available PDBs loaded.
        min_gyration_radius: float
            Minimum gyration radius (in nm). Default is None.
        max_gyration_radius: float
            Maximum gyration radius (in nm). Default is None.
        alpha_beta_coil: tuple
            Minimum percentage of alpha-helix, beta-sheet and coil residues. Default is None.
        solid_ss: float
            Minimum percentage of solid secondary structure (alpha + beta)/total * 100. Default is None.
        numFrames: int
            Minimum number of frames in trajectory. Default is None.
        cutoff_ang: float
            Cutoff distance in angstroms for neighbor calculations. Default is 7.0.
        """
        super().__init__()

        self.url = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.source_file = source_file
        self.file_basename = file_basename
        self.lagtime = lagtime
        self.numAtoms = numAtoms
        self.numNoHAtoms = numNoHAtoms
        self.numResidues = numResidues
        self.temperatures = temperatures if temperatures else ["348"]
        self.temperatures = [str(temp) for temp in self.temperatures]
        self.skip_frames = skip_frames
        self.pdb_list = load_pdb_list(pdb_list) if pdb_list is not None else None
        self.min_gyration_radius = min_gyration_radius
        self.max_gyration_radius = max_gyration_radius
        self.alpha_beta_coil = alpha_beta_coil
        self.numFrames = numFrames
        self.solid_ss = solid_ss
        self.cutoff = cutoff_ang

        # Initialize dataset
        self._ensure_source_file()
        self._filter_and_prepare_data()
        self.download()
        self._setup_idx()

        # Calculate total size
        self.total_size_mb = self.calculate_dataset_size()

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self._log_info()
        else:
            self._log_info()

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

    def download(self):
        """Download required HDF5 files for selected PDB IDs."""
        for pdb_id in self.processed.keys():
            file_name = f"{self.file_basename}_{pdb_id}.h5"
            file_path = self.root / file_name
            if not file_path.exists():
                assert self.file_basename == "mdcath_dataset", (
                    "Only 'mdcath_dataset' is supported as file_basename for download."
                )
                print(f"Downloading {file_name}")
                urllib.request.urlretrieve(
                    opj(self.url, "data", file_name), str(file_path)
                )

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

            for pdb_id in tqdm(domains, desc="Processing mdcath source"):
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load atomic numbers, coordinates, and forces for a specific frame."""
        actual_frame_idx = frame_idx * self.skip_frames
        slice_idxs = np.s_[actual_frame_idx : actual_frame_idx + 1]

        with h5py.File(file_path, "r") as f:
            z = f[pdb_id]["z"][:]
            coords = np.zeros((z.shape[0], 3))
            forces = np.zeros((z.shape[0], 3))

            group = f[f"{pdb_id}/{temp}/{replica}"]
            group["coords"].read_direct(coords, slice_idxs)
            group["forces"].read_direct(forces, slice_idxs)

            # coords and forces shape (num_atoms, 3)
            assert coords.shape[0] == forces.shape[0], (
                f"Number of frames mismatch between coords and forces: {group['coords'].shape[0]} vs {group['forces'].shape[0]}"
            )
            assert coords.shape[0] == z.shape[0], (
                f"Number of atoms mismatch between coords and z: {group['coords'].shape[1]} vs {z.shape[0]}"
            )

        return z, coords, forces

    def __len__(self) -> int:
        """Return number of valid (frame, frame+lag) pairs."""
        return len(self.idx)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a pair of configurations separated by lagtime.

        Returns:
        --------
        dict with keys:
            - 'item': dict with 'z', 'pos', 'neg_dy', 'info' for current frame
            - 'item_lag': dict with 'z', 'pos', 'neg_dy', 'info' for lagged frame
        """
        if isinstance(index, (slice, list, tuple)):
            raise NotImplementedError("Only integer indexing is supported")

        pdb_id, file_path, temp, replica, frame_idx = self.idx[index]

        # Load current frame
        z_t0, coords_t0, forces_t0 = self._load_frame(
            file_path, pdb_id, temp, replica, frame_idx
        )

        # Load lagged frame
        z_tlag, coords_tlag, forces_tlag = self._load_frame(
            file_path, pdb_id, temp, replica, frame_idx + self.lagtime
        )

        # Create data dictionaries
        item = {
            "z": torch.tensor(z_t0, dtype=torch.long),
            "pos": torch.tensor(coords_t0, dtype=torch.float32),
            "neg_dy": torch.tensor(forces_t0, dtype=torch.float32),
            "info": f"{pdb_id}_{temp}_{replica}_{frame_idx}",
            "pdb_id": pdb_id,
            "temp": temp,
            "replica": replica,
        }

        item_lag = {
            "z": torch.tensor(z_tlag, dtype=torch.long),
            "pos": torch.tensor(coords_tlag, dtype=torch.float32),
            "neg_dy": torch.tensor(forces_tlag, dtype=torch.float32),
            "info": f"{pdb_id}_{temp}_{replica}_{frame_idx + self.lagtime}",
            "pdb_id": pdb_id,
            "temp": temp,
            "replica": replica,
        }

        return {
            "item": item,
            "item_lag": item_lag,
        }

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
