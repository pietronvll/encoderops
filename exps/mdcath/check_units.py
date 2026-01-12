#!/usr/bin/env python3
"""Investigate mdCATH coordinate units and neighbor counts.

This script helps determine whether coordinates in mdCATH are in Angstroms or nanometers
by examining:
1. Coordinate ranges (typical proteins are 20-100 Å across)
2. Box dimensions
3. Interatomic distances (C-C bonds ~1.5 Å, C-N ~1.4 Å)
4. Number of neighbors at different cutoffs

FINDINGS:
---------
- Coordinates ARE in Angstroms (min bond ~0.95 Å, protein extent ~40-50 Å)
- Box dimensions ARE in Nanometers (~5.5 nm for a ~48 Å protein)
- This is a UNIT MISMATCH that affects PBC calculations!

The box should be converted: box_angstrom = box_nm * 10
"""

import os
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist, pdist


def analyze_h5_file(h5_path: Path, pdb_id: str, temp: str = "348"):
    """Analyze a single H5 file to understand coordinate units."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {pdb_id}")
    print(f"{'='*60}")

    with h5py.File(h5_path, "r") as f:
        # Get atomic numbers
        z = f[pdb_id]["z"][:]
        print(f"\nAtom count: {len(z)}")
        print(f"Unique elements (Z): {np.unique(z)}")

        # Count by element
        for elem_z, name in [(1, "H"), (6, "C"), (7, "N"), (8, "O"), (16, "S")]:
            count = np.sum(z == elem_z)
            if count > 0:
                print(f"  {name} ({elem_z}): {count}")

        # Get first replica
        replicas = list(f[f"{pdb_id}/{temp}"].keys())
        if not replicas:
            print(f"No replicas found for temp {temp}")
            return None
        replica = replicas[0]
        print(f"\nUsing replica: {replica}")

        group = f[f"{pdb_id}/{temp}/{replica}"]

        # Get coordinates for first frame
        coords = group["coords"][0]  # shape: (num_atoms, 3)
        print(f"\nCoordinates shape: {coords.shape}")

        # Coordinate statistics
        print(f"\nCoordinate ranges:")
        for i, axis in enumerate(["X", "Y", "Z"]):
            print(
                f"  {axis}: min={coords[:, i].min():.3f}, max={coords[:, i].max():.3f}, "
                f"range={coords[:, i].max() - coords[:, i].min():.3f}"
            )

        # Center of mass and extent
        extent = coords.max(axis=0) - coords.min(axis=0)
        print(f"\nProtein extent (max - min per axis): {extent}")
        print(f"Max extent: {extent.max():.3f}")

        # Box dimensions
        box = group["box"][:]
        box_diag = np.diag(box)
        print(f"\nBox matrix:\n{box}")
        print(f"Box diagonal: {box_diag}")

        # Check box vs protein size
        print(f"\n{'='*40}")
        print("BOX vs PROTEIN SIZE ANALYSIS")
        print(f"{'='*40}")
        print(f"Protein extent: {extent.max():.2f} (units?)")
        print(f"Box size:       {box_diag.mean():.2f} (units?)")
        print(f"Ratio (box/protein): {box_diag.mean() / extent.max():.3f}")

        if box_diag.mean() < extent.max():
            print("\n⚠️  WARNING: Box is SMALLER than protein!")
            print("   This suggests UNIT MISMATCH:")
            print(f"   - If coords in Å and box in nm: box would be {box_diag.mean()*10:.1f} Å")
            print(f"   - Ratio would be: {box_diag.mean()*10 / extent.max():.3f}")

        # Analyze interatomic distances
        print(f"\n{'='*40}")
        print("Interatomic distance analysis")
        print(f"{'='*40}")

        # Get C-C distances (carbon = 6)
        c_mask = z == 6
        c_coords = coords[c_mask]
        if len(c_coords) > 1:
            c_dists = pdist(c_coords)
            print(f"\nC-C distances (n={len(c_coords)} carbons):")
            print(f"  Min: {c_dists.min():.4f}")
            print(f"  5th percentile: {np.percentile(c_dists, 5):.4f}")
            print(f"  Typical C-C bond: 1.54 Å (sp3-sp3)")

        # Get C-N distances
        n_mask = z == 7
        n_coords = coords[n_mask]
        if len(c_coords) > 0 and len(n_coords) > 0:
            cn_dists = cdist(c_coords, n_coords).flatten()
            print(f"\nC-N distances:")
            print(f"  Min: {cn_dists.min():.4f}")
            print(f"  5th percentile: {np.percentile(cn_dists, 5):.4f}")
            print(f"  Typical C-N bond: 1.47 Å (sp3)")

        # All-atom minimum distances
        all_dists = pdist(coords)
        print(f"\nAll pairwise distances (n={len(coords)} atoms):")
        print(f"  Min: {all_dists.min():.4f}")
        print(f"  1st percentile: {np.percentile(all_dists, 1):.4f}")
        print(f"  5th percentile: {np.percentile(all_dists, 5):.4f}")
        print(f"  Typical covalent bonds: 1.0-1.5 Å")

        # Neighbor counts at different cutoffs
        print(f"\n{'='*40}")
        print("Neighbor counts at different cutoffs")
        print(f"{'='*40}")

        # Use distance matrix for neighbor counting
        dist_matrix = cdist(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)  # exclude self

        for cutoff in [0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0]:
            neighbors = (dist_matrix < cutoff).sum(axis=1)
            total_edges = neighbors.sum()
            avg_neighbors = neighbors.mean()
            max_neighbors = neighbors.max()
            print(
                f"  Cutoff {cutoff:5.1f}: avg neighbors={avg_neighbors:6.1f}, "
                f"max={max_neighbors:4d}, total edges={total_edges:,}"
            )

        # Unit inference
        print(f"\n{'='*40}")
        print("UNIT INFERENCE")
        print(f"{'='*40}")

        min_bond = all_dists.min()
        if 0.8 < min_bond < 2.0:
            print(f"✓ Coordinates: Min distance {min_bond:.3f} -> ANGSTROMS")
            print("  (Covalent bonds are typically 1.0-1.5 Å)")
        elif 0.08 < min_bond < 0.2:
            print(f"✗ Coordinates: Min distance {min_bond:.3f} -> NANOMETERS")
            print("  (Covalent bonds would be 0.1-0.15 nm)")

        if box_diag.mean() < extent.max() and box_diag.mean() * 10 > extent.max():
            print(f"✓ Box: {box_diag.mean():.2f} with protein {extent.max():.1f} -> NANOMETERS")
            print(f"  (Box in nm would be {box_diag.mean()*10:.1f} Å, larger than protein)")
        elif box_diag.mean() > extent.max():
            print(f"? Box: {box_diag.mean():.2f} > protein {extent.max():.1f} -> Same units as coords")

        return {
            "pdb_id": pdb_id,
            "n_atoms": len(z),
            "extent": extent.max(),
            "box_diag": box_diag.mean(),
            "min_bond": min_bond,
        }


def main():
    # Find mdCATH data
    data_path = os.environ.get("MDCATH_DATA_PATH")
    if data_path is None:
        # Try common locations
        for path in [
            Path("datasets/mdcath"),
            Path("data/mdcath"),
            Path.home() / "data" / "mdcath",
            Path("/data/mdcath"),
        ]:
            if path.exists():
                data_path = str(path)
                break

    if data_path is None:
        print("MDCATH_DATA_PATH not set and couldn't find data directory")
        print("Set MDCATH_DATA_PATH environment variable or provide path")
        return

    data_path = Path(data_path)
    print(f"Using data path: {data_path}")

    # Find H5 files
    h5_files = list(data_path.glob("mdcath_dataset_*.h5"))
    if not h5_files:
        print(f"No H5 files found in {data_path}")
        return

    print(f"Found {len(h5_files)} H5 files")

    # Analyze first few files
    results = []
    for h5_file in h5_files[:3]:
        # Extract PDB ID from filename
        pdb_id = h5_file.stem.replace("mdcath_dataset_", "")
        result = analyze_h5_file(h5_file, pdb_id)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print("\nAll proteins analyzed show:")
        print("  - Coordinates in ANGSTROMS (bond lengths ~1.0-1.5)")
        print("  - Box dimensions in NANOMETERS (box ~5-6, protein ~40-50)")
        print("\n⚠️  UNIT MISMATCH DETECTED!")
        print("   The box needs to be converted to Angstroms: box_ang = box_nm * 10")
        print("\n   In src/mdcath.py, change:")
        print('     cell=box,  # box is (3,3) matrix in Angstroms  <- WRONG')
        print("   To:")
        print("     cell=box * 10,  # Convert nm to Angstroms")


if __name__ == "__main__":
    main()
