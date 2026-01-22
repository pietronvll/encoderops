import os
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro
from loguru import logger

from src.data import mdtraj_load, traj_to_confs

# Temperature values taken from: https://www.science.org/action/downloadSupplement?doi=10.1126%2Fscience.1208351&file=1208351-lindorff-larsen.som.pdf
PROTEIN_TEMPERATURE_DICT = {
    "1FME": 325,        # 1FME is BBA
    "2F4K": 360,        # 2F4K is Villin
    "2JOF": 290,        # 2JOF is Trp-cage
    "CLN025": 340,      # CLN025 is Chignolin
    "GTT": 360,         # GTT is WW-domain
    "NTL9": 355,        # NTL9 is NTL9
}


def main(
    protein_id: str,
    system_selection: str | None = "all and not type H",
    subsampling_time_ns: float = 1.0,
):
    data_path = Path(os.environ["DATA_PATH"])
    
    # Find all trajectories for this protein
    trajectory_dirs = sorted([
        d for d in data_path.glob(f"DESRES-Trajectory_{protein_id}-*-protein")
        if d.is_dir()
    ])
    
    if not trajectory_dirs:
        logger.error(f"No trajectories found for protein {protein_id}")
        return
    
    logger.info(f"Found {len(trajectory_dirs)} trajectory(ies) for {protein_id}")
    
    # Get temperature for this protein
    if protein_id not in PROTEIN_TEMPERATURE_DICT:
        logger.error(f"Protein {protein_id} not in PROTEIN_TEMPERATURE_DICT")
        return
    
    temperature = PROTEIN_TEMPERATURE_DICT[protein_id]
    
    # Output file
    database_path = Path(__file__).parent.parent / "h5_data"
    database_path.mkdir(parents=True, exist_ok=True)
    h5_path = database_path / f"{protein_id}.h5"
    
    # Process all trajectories and store in single file
    store_trajectories_to_h5(
        h5_path,
        protein_id,
        temperature,
        trajectory_dirs,
        system_selection,
        subsampling_time_ns,
    )


@torch.no_grad()
def store_trajectories_to_h5(
    h5_path,
    protein_id,
    temperature,
    trajectory_dirs,
    system_selection,
    subsampling_time_ns,
):
    """Store all trajectories of a protein into a hierarchical HDF5 file.
    
    Structure:
    protein_id.h5
    └── protein_id/
        └── z
        └── temperature/
            ├── 0/
            │   ├── box
            │   └── coords
            └── 1/
                ├── box
                └── coords

    """
    sampling_rate_ns = 0.2
    stride = int(subsampling_time_ns / sampling_rate_ns)
    
    z_data = None  # To store z once
    
    with h5py.File(h5_path, "w", libver="latest") as f:
        # Create protein group
        protein_group = f.create_group(protein_id)
        
        # Create temperature group
        temp_group = protein_group.create_group(str(temperature))
        
        # Process each trajectory
        for traj_dir in trajectory_dirs:
            # Extract trajectory ID from directory name
            # Directory format: DESRES-Trajectory_PROTEIN-TRAJ_ID-protein
            traj_id = int(traj_dir.name.split("-")[2])
            
            logger.info(f"Processing trajectory {traj_id} from {traj_dir.name}")
            
            # The actual trajectory data is in a nested subdirectory with the same name
            # Structure: DESRES-Trajectory_PROTEIN-TRAJ_ID-protein/PROTEIN-TRAJ_ID-protein/
            inner_dir = traj_dir / traj_dir.name.replace("DESRES-Trajectory_", "")
            
            # Load trajectory
            trajectory_files = sorted([str(f) for f in inner_dir.glob("*.dcd")])
            top = next(inner_dir.glob("*.pdb")).__str__()
            
            traj = mdtraj_load(trajectory_files, top, 1)
            
            if system_selection is not None:
                system_atoms = traj.top.select(system_selection)
                logger.info(f"System selection: {system_selection}")
                traj = traj.atom_slice(system_atoms)
            
            # Get configurations
            configs, _, _ = traj_to_confs(traj)
            
            # Extract atomic numbers (same for all frames)
            z_data = np.array(configs[0].atomic_numbers, dtype=np.int64)
            
            # Apply subsampling only to coordinates
            if stride > 1:
                logger.info(f"Subsampling with stride={stride}")
                coords = np.array([config.positions for config in configs], dtype=np.float32)
                
                coords = coords[::stride]
            else:
                coords = np.array([config.positions for config in configs], dtype=np.float32)
                
            # Convert box to float32
            box = configs[0].cell
            box = np.array(box, dtype=np.float32)
            
            # Create trajectory group
            traj_group = temp_group.create_group(str(traj_id))
            
            # Store box (all frames)
            traj_group.create_dataset(
                "box",
                data=box,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            
            # Store coords (subsampled)
            traj_group.create_dataset(
                "coords",
                data=coords,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            
            logger.info(f"Stored trajectory {traj_id}: coords shape {coords.shape}, box shape {box.shape}")
        
        # Store z at root level (shared across all trajectories)
        if z_data is not None:
            protein_group.create_dataset(
                "z",
                data=z_data,
                compression="gzip",
            )
            logger.info(f"Stored z at protein_group level: shape {z_data.shape}")
    
    logger.info(f"Data stored to {h5_path}")


if __name__ == "__main__":
    tyro.cli(main)
