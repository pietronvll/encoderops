#!/usr/bin/env python3
"""Plot throughput scaling results from MDCATH benchmarks."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(results_dir: Path) -> list[dict]:
    """Load all throughput results and compute median throughput and memory per config."""
    # Group measurements by (nodes, gpus_per_node)
    throughput_measurements: dict[tuple[int, int], list[float]] = defaultdict(list)
    memory_measurements: dict[tuple[int, int], list[float]] = defaultdict(list)

    for f in results_dir.glob("throughput_*nodes_*gpus_*.json"):
        # Parse filename: throughput_1nodes_4gpus_20260112_115324.json
        parts = f.stem.split("_")
        nodes = int(parts[1].replace("nodes", ""))
        gpus_per_node = int(parts[2].replace("gpus", ""))
        key = (nodes, gpus_per_node)

        with open(f) as fp:
            data = json.load(fp)

        # Collect all measurements from batch logs
        for entry in data["batch_logs"]:
            throughput_measurements[key].append(entry["throughput_atoms_per_sec"])
            # Memory data may not be present in older logs
            if "gpu_memory_max_allocated_gb" in entry:
                memory_measurements[key].append(entry["gpu_memory_max_allocated_gb"])

    # Compute median and IQR for each config
    results = []
    for (nodes, gpus_per_node), throughputs in throughput_measurements.items():
        total_gpus = nodes * gpus_per_node
        median = statistics.median(throughputs)
        if len(throughputs) >= 2:
            q1 = statistics.median([t for t in throughputs if t <= median])
            q3 = statistics.median([t for t in throughputs if t >= median])
        else:
            q1 = q3 = median

        # Memory stats (if available)
        key = (nodes, gpus_per_node)
        mem_data = memory_measurements.get(key, [])
        if mem_data:
            mem_median = statistics.median(mem_data)
            mem_max = max(mem_data)
        else:
            mem_median = mem_max = None

        results.append(
            {
                "nodes": nodes,
                "gpus_per_node": gpus_per_node,
                "total_gpus": total_gpus,
                "throughput": median,
                "throughput_q1": q1,
                "throughput_q3": q3,
                "n_samples": len(throughputs),
                "gpu_memory_median_gb": mem_median,
                "gpu_memory_max_gb": mem_max,
            }
        )

    return sorted(results, key=lambda x: x["total_gpus"])


def load_stats(stats_path: Path, temperature: str = "348") -> dict:
    """Load dataset statistics."""
    with open(stats_path) as f:
        stats = json.load(f)
    return {
        "total_atoms": stats["by_temperature"][temperature]["total_atoms"],
        "total_frames": stats["by_temperature"][temperature]["total_frames"],
    }


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "benchmark_results"
    stats_path = script_dir / "mdcath_stats.json"

    results = load_results(results_dir)
    stats = load_stats(stats_path)

    if not results:
        print("No results found in", results_dir)
        return

    # Extract data
    gpus = [r["total_gpus"] for r in results]
    throughputs = [r["throughput"] for r in results]
    throughputs_q1 = [r["throughput_q1"] for r in results]
    throughputs_q3 = [r["throughput_q3"] for r in results]
    n_samples = [r["n_samples"] for r in results]

    # Calculate error bars (asymmetric: median - q1, q3 - median)
    yerr_low = [t - q1 for t, q1 in zip(throughputs, throughputs_q1)]
    yerr_high = [q3 - t for t, q3 in zip(throughputs, throughputs_q3)]

    # Calculate GPU-hours per epoch
    total_atoms = stats["total_atoms"]
    gpu_hours = [(total_atoms / t) / 3600 * g for t, g in zip(throughputs, gpus)]
    gpu_hours_q1 = [
        (total_atoms / q3) / 3600 * g for q3, g in zip(throughputs_q3, gpus)
    ]
    gpu_hours_q3 = [
        (total_atoms / q1) / 3600 * g for q1, g in zip(throughputs_q1, gpus)
    ]
    gpu_hours_yerr_low = [h - h1 for h, h1 in zip(gpu_hours, gpu_hours_q1)]
    gpu_hours_yerr_high = [h3 - h for h, h3 in zip(gpu_hours, gpu_hours_q3)]

    # Ideal scaling (linear from first point)
    ideal_throughput = [throughputs[0] * g / gpus[0] for g in gpus]

    # GPU memory data
    gpu_memory = [r["gpu_memory_max_gb"] for r in results]
    has_memory_data = any(m is not None for m in gpu_memory)

    # Create figure
    n_plots = 3 if has_memory_data else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # Plot 1: Throughput vs GPUs
    ax1 = axes[0]
    ax1.errorbar(
        gpus,
        throughputs,
        yerr=[yerr_low, yerr_high],
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=4,
        label="Measured (median)",
    )
    # ax1.plot(gpus, ideal_throughput, "--", color="gray", alpha=0.7, label="Ideal scaling")
    ax1.set_xlabel("Number of GPUs", fontsize=12)
    ax1.set_ylabel("Throughput (atoms/sec)", fontsize=12)
    ax1.set_title("Throughput Scaling", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(gpus)

    # Plot 2: GPU-hours per epoch
    ax2 = axes[1]
    ax2.errorbar(
        gpus,
        gpu_hours,
        yerr=[gpu_hours_yerr_low, gpu_hours_yerr_high],
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=4,
        color="tab:orange",
    )
    ax2.set_xlabel("Number of GPUs", fontsize=12)
    ax2.set_ylabel("GPU-hours per epoch", fontsize=12)
    ax2.set_title("Cost per Epoch (348K)", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(gpus)

    # Plot 3: GPU Memory Usage (if data available)
    if has_memory_data:
        ax3 = axes[2]
        # Filter out None values for plotting
        valid_gpus = [g for g, m in zip(gpus, gpu_memory) if m is not None]
        valid_memory = [m for m in gpu_memory if m is not None]
        ax3.bar(valid_gpus, valid_memory, color="tab:green", alpha=0.7, width=0.6)
        ax3.set_xlabel("Number of GPUs", fontsize=12)
        ax3.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
        ax3.set_title("GPU Memory Usage (per GPU)", fontsize=14)
        ax3.grid(True, alpha=0.3, axis="y")
        ax3.set_xticks(valid_gpus)
        # Add value labels on bars
        for g, m in zip(valid_gpus, valid_memory):
            ax3.annotate(
                f"{m:.1f}",
                xy=(g, m),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Add scaling efficiency annotation
    if len(results) > 1:
        efficiency = (throughputs[-1] / throughputs[0]) / (gpus[-1] / gpus[0]) * 100
        ax1.annotate(
            f"Scaling efficiency: {efficiency:.1f}%",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    output_path = results_dir / "scaling_plot.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    # Print summary table
    print("\nSummary:")
    if has_memory_data:
        print(
            f"{'GPUs':>6} {'Nodes':>6} {'Median (atoms/s)':>18} {'Peak Mem (GB)':>14} {'Samples':>8}"
        )
        print("-" * 56)
        for r in results:
            mem_str = f"{r['gpu_memory_max_gb']:.1f}" if r["gpu_memory_max_gb"] else "N/A"
            print(
                f"{r['total_gpus']:>6} {r['nodes']:>6} {r['throughput']:>18,.1f} {mem_str:>14} {r['n_samples']:>8}"
            )
    else:
        print(f"{'GPUs':>6} {'Nodes':>6} {'Median (atoms/s)':>18} {'Samples':>8}")
        print("-" * 42)
        for r in results:
            print(
                f"{r['total_gpus']:>6} {r['nodes']:>6} {r['throughput']:>18,.1f} {r['n_samples']:>8}"
            )

    plt.show()


if __name__ == "__main__":
    main()
