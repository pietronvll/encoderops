"""Compute total atoms in MDCATH dataset from mdcath_source.h5.

This script loops through the master file and computes:
- Total atoms across all configurations (atoms × frames)
- Breakdown by temperature
- Dataset statistics
"""

import json
from pathlib import Path

import h5py
from tqdm import tqdm


def compute_total_atoms(
    source_file: str = "datasets/mdcath/mdcath_source.h5",
    output_file: str = "exps/mdcath/mdcath_stats.json",
    remove_hydrogen: bool = True,
):
    """Compute total atoms in the MDCATH dataset.

    Args:
        source_file: Path to mdcath_source.h5
        remove_hydrogen: If True, use numNoHAtoms; else use numProteinAtoms
    """
    stats = {
        "total_atoms": 0,  # Sum of (atoms_per_structure × frames)
        "total_frames": 0,
        "total_domains": 0,
        "by_temperature": {},
        "atom_field": "numNoHAtoms" if remove_hydrogen else "numProteinAtoms",
    }

    temperatures = ["320", "348", "379", "413", "450"]
    for temp in temperatures:
        stats["by_temperature"][temp] = {
            "total_atoms": 0,
            "total_frames": 0,
            "domains_with_temp": 0,
        }

    with h5py.File(source_file, "r") as f:
        domains = list(f.keys())
        stats["total_domains"] = len(domains)

        for domain in tqdm(domains, desc="Processing domains"):
            pdb_group = f[domain]

            # Get atom count for this structure
            if remove_hydrogen:
                num_atoms = pdb_group.attrs.get("numNoHAtoms", pdb_group.attrs["numProteinAtoms"])
            else:
                num_atoms = pdb_group.attrs["numProteinAtoms"]

            for temp in temperatures:
                if temp not in pdb_group:
                    continue

                stats["by_temperature"][temp]["domains_with_temp"] += 1

                for replica in pdb_group[temp].keys():
                    replica_group = pdb_group[temp][replica]
                    num_frames = int(replica_group.attrs["numFrames"])

                    atoms_in_replica = int(num_atoms) * num_frames

                    stats["total_atoms"] += atoms_in_replica
                    stats["total_frames"] += num_frames
                    stats["by_temperature"][temp]["total_atoms"] += atoms_in_replica
                    stats["by_temperature"][temp]["total_frames"] += num_frames

    # Add derived statistics
    stats["avg_atoms_per_frame"] = stats["total_atoms"] / stats["total_frames"] if stats["total_frames"] > 0 else 0

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nMDCATH Dataset Statistics")
    print("=" * 50)
    print(f"Total domains: {stats['total_domains']}")
    print(f"Total frames: {stats['total_frames']:,}")
    print(f"Total atoms (atoms × frames): {stats['total_atoms']:,}")
    print(f"Avg atoms per frame: {stats['avg_atoms_per_frame']:.1f}")
    print(f"\nBy temperature:")
    for temp, temp_stats in stats["by_temperature"].items():
        print(f"  {temp}K: {temp_stats['total_atoms']:,} atoms, {temp_stats['total_frames']:,} frames, {temp_stats['domains_with_temp']} domains")
    print(f"\nSaved to: {output_file}")

    return stats


if __name__ == "__main__":
    compute_total_atoms()
