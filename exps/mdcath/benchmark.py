"""Benchmark script for measuring MDCATH training throughput across GPU configurations.

Usage:
    uv run python -m exps.mdcath.benchmark --num_gpus 4 --num_nodes 2

For SLURM submission, use the submit_benchmark.py script.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import Callback

from src.configs import Configs, MDCATHDataArgs, TrainerArgs, defaults
from src.mdcath import MDCATHDataModule
from src.model import EvolutionOperator
from src.modules import SchNet

# Default PDB list for benchmarking (small subset for quick tests)
DEFAULT_PDB_LIST = [
    "12asA00",
    "153lA00",
    "16pkA02",
    "1a02F00",
    "1a05A00",
    "1a0aA00",
]


class ThroughputCallback(Callback):
    """Callback to measure throughput in atoms/second with periodic file writes."""

    def __init__(
        self,
        output_file: Path | None = None,
        log_every_n_batches: int = 100,
    ):
        self.output_file = output_file
        self.log_every_n_batches = log_every_n_batches
        self.epoch_start_time = None
        self.total_atoms = 0
        self.total_batches = 0
        self.batch_logs: list[dict] = []
        self.epoch_logs: list[dict] = []

    def _is_main_process(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _write_to_file(self):
        if self.output_file is None or not self._is_main_process():
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(
                {"batch_logs": self.batch_logs, "epoch_logs": self.epoch_logs},
                f,
                indent=2,
            )

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = perf_counter()
        self.total_atoms = 0
        self.total_batches = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.total_batches += 1
        # Count atoms in current batch (each sample has item and item_lag)
        # batch.item.z contains atomic numbers, shape: (total_atoms_in_batch,)
        atoms_in_batch = 0
        if hasattr(batch, "item") and hasattr(batch.item, "z"):
            # Both item and item_lag have the same number of atoms
            atoms_in_batch = batch.item.z.shape[0] * 2  # x2 for item + item_lag
            self.total_atoms += atoms_in_batch

        # Periodic logging
        if self.total_batches % self.log_every_n_batches == 0 and self.epoch_start_time is not None:
            elapsed = perf_counter() - self.epoch_start_time
            # Gather atoms from all ranks for accurate throughput
            if dist.is_initialized():
                total_atoms_tensor = torch.tensor(
                    self.total_atoms,
                    dtype=torch.long,
                    device=trainer.strategy.root_device,
                )
                dist.all_reduce(total_atoms_tensor, op=dist.ReduceOp.SUM)
                global_atoms = total_atoms_tensor.item()
            else:
                global_atoms = self.total_atoms

            throughput = global_atoms / elapsed if elapsed > 0 else 0

            if self._is_main_process():
                log_entry = {
                    "epoch": trainer.current_epoch,
                    "batch": self.total_batches,
                    "elapsed_sec": round(elapsed, 2),
                    "atoms_processed": global_atoms,
                    "throughput_atoms_per_sec": round(throughput, 1),
                }
                self.batch_logs.append(log_entry)
                self._write_to_file()
                print(
                    f"[Batch {self.total_batches}] "
                    f"elapsed={elapsed:.1f}s, "
                    f"atoms={global_atoms:,}, "
                    f"throughput={throughput:,.0f} atoms/sec"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is None:
            return

        epoch_time = perf_counter() - self.epoch_start_time

        # Gather total atoms from all ranks
        if dist.is_initialized():
            total_atoms_tensor = torch.tensor(
                self.total_atoms,
                dtype=torch.long,
                device=trainer.strategy.root_device,
            )
            dist.all_reduce(total_atoms_tensor, op=dist.ReduceOp.SUM)
            global_total_atoms = total_atoms_tensor.item()
        else:
            global_total_atoms = self.total_atoms

        self.global_total_atoms = global_total_atoms
        self.epoch_time = epoch_time
        self.throughput = global_total_atoms / epoch_time if epoch_time > 0 else 0

        if self._is_main_process():
            epoch_entry = {
                "epoch": trainer.current_epoch,
                "epoch_time_sec": round(epoch_time, 2),
                "total_atoms": global_total_atoms,
                "total_batches": self.total_batches,
                "throughput_atoms_per_sec": round(self.throughput, 1),
            }
            self.epoch_logs.append(epoch_entry)
            self._write_to_file()
            print(
                f"\n[Epoch {trainer.current_epoch} complete] "
                f"time={epoch_time:.1f}s, "
                f"atoms={global_total_atoms:,}, "
                f"throughput={self.throughput:,.0f} atoms/sec\n"
            )


def benchmark(
    num_gpus: int = 1,
    num_nodes: int = 1,
    epochs: int = 1,
    batch_size: int = 128,
    dataloader_workers: int = 8,
    pdb_list: list[str] | None = None,
    temperature: str = "348",
    output_dir: str = "exps/mdcath/benchmark_results",
):
    """
    Benchmark MDCATH training throughput.

    Args:
        num_gpus: Number of GPUs per node
        num_nodes: Number of nodes
        epochs: Number of epochs to run
        batch_size: Batch size per GPU
        dataloader_workers: Number of dataloader workers per GPU
        pdb_list: List of PDB IDs to use (default: small subset for testing)
        temperature: Single temperature to use (default: 348K)
        output_dir: Directory to save benchmark results
    """
    # Use default PDB list if none provided
    if pdb_list is None:
        pdb_list = DEFAULT_PDB_LIST

    # Load base configuration
    _, base_config = defaults["MDCATH"]

    # Override data_args with single temperature
    data_args: MDCATHDataArgs = base_config.data_args
    data_args.temperatures = temperature
    data_args.pdb_list: list[str] = pdb_list

    cfg = Configs(
        trainer_args=TrainerArgs(
            latent_dim=base_config.trainer_args.latent_dim,
            encoder_lr=base_config.trainer_args.encoder_lr,
            linear_lr=base_config.trainer_args.linear_lr,
            min_encoder_lr=base_config.trainer_args.min_encoder_lr,
            epochs=epochs,
            batch_size=batch_size,
            max_grad_norm=base_config.trainer_args.max_grad_norm,
            normalize_lin=base_config.trainer_args.normalize_lin,
            regularization=base_config.trainer_args.regularization,
            normalize_latents=base_config.trainer_args.normalize_latents,
            loss=base_config.trainer_args.loss,
            seed=base_config.trainer_args.seed,
            forecast=base_config.trainer_args.forecast,
        ),
        model_args=base_config.model_args,
        data_args=data_args,
        wandb_project=base_config.wandb_project,
        wandb_entity=base_config.wandb_entity,
        offline=True,
        num_devices=num_gpus,
        num_nodes=num_nodes,
        dataloader_workers=dataloader_workers,
    )

    # Setup data module
    datamodule = MDCATHDataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.setup("fit")

    # Setup output path for intermediate results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    throughput_log_file = output_path / f"throughput_{num_nodes}nodes_{num_gpus}gpus_{timestamp}.json"

    # Setup callback
    throughput_callback = ThroughputCallback(
        output_file=throughput_log_file,
        log_every_n_batches=100,
    )

    # Setup trainer (minimal, no logging)
    trainer = Trainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[throughput_callback],
        strategy="ddp" if num_gpus * num_nodes > 1 else "auto",
        accelerator="cuda",
        devices=num_gpus,
        num_nodes=num_nodes,
        max_epochs=epochs,
        log_every_n_steps=50,
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    # Setup model
    encoder_args = {
        "n_out": cfg.trainer_args.latent_dim,
        "cutoff": cfg.data_args.cutoff_ang,
        "atomic_numbers": datamodule.dataset.z_table.zs,
    }
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(SchNet, encoder_args, cfg.trainer_args)

    # Run benchmark
    start_time = perf_counter()
    trainer.fit(model, datamodule=datamodule)
    total_time = perf_counter() - start_time

    # Collect results (only on rank 0)
    is_main = not dist.is_initialized() or dist.get_rank() == 0

    if is_main:
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_gpus": num_gpus,
                "num_nodes": num_nodes,
                "total_gpus": num_gpus * num_nodes,
                "batch_size": batch_size,
                "epochs": epochs,
                "dataloader_workers": dataloader_workers,
                "temperature": temperature,
                "pdb_list": pdb_list,
            },
            "dataset": {
                "num_samples": len(datamodule.dataset),
                "num_domains": len(datamodule.dataset.processed),
            },
            "results": {
                "total_time_sec": total_time,
                "time_per_epoch_sec": total_time / epochs,
                "total_atoms_processed": throughput_callback.global_total_atoms,
                "throughput_atoms_per_sec": throughput_callback.throughput,
                "epoch_time_sec": throughput_callback.epoch_time,
            },
        }

        # Load full dataset stats for epoch time estimation
        stats_file = Path("exps/mdcath/mdcath_stats.json")
        if stats_file.exists():
            with open(stats_file) as f:
                full_stats = json.load(f)

            # Estimate time for full epoch on single temperature
            single_temp_atoms = full_stats["by_temperature"][temperature]["total_atoms"]
            estimated_epoch_time_sec = (
                single_temp_atoms / throughput_callback.throughput
            )
            estimated_epoch_time_hours = estimated_epoch_time_sec / 3600
            gpu_hours_per_epoch = estimated_epoch_time_hours * (num_gpus * num_nodes)

            results["full_dataset_estimates"] = {
                "single_temp_total_atoms": single_temp_atoms,
                "estimated_epoch_time_sec": estimated_epoch_time_sec,
                "estimated_epoch_time_hours": estimated_epoch_time_hours,
                "gpu_hours_per_epoch": gpu_hours_per_epoch,
            }

        # Save results
        filename = f"benchmark_{num_nodes}nodes_{num_gpus}gpus_{timestamp}.json"
        result_file = output_path / filename

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(
            f"Configuration: {num_gpus} GPUs x {num_nodes} nodes = {num_gpus * num_nodes} total GPUs"
        )
        print(f"Batch size: {batch_size} per GPU")
        print(
            f"Dataset: {len(datamodule.dataset)} samples, {len(datamodule.dataset.processed)} domains"
        )
        print("\nTiming:")
        print(f"  Total time: {total_time:.2f} sec")
        print(f"  Time per epoch: {total_time / epochs:.2f} sec")
        print("\nThroughput:")
        print(f"  Atoms processed: {throughput_callback.global_total_atoms:,}")
        print(f"  Throughput: {throughput_callback.throughput:,.0f} atoms/sec")

        if "full_dataset_estimates" in results:
            est = results["full_dataset_estimates"]
            print(f"\nFull Dataset Estimates (single temperature {temperature}K):")
            print(f"  Total atoms: {est['single_temp_total_atoms']:,}")
            print(
                f"  Estimated epoch time: {est['estimated_epoch_time_hours']:.2f} hours"
            )
            print(f"  GPU-hours per epoch: {est['gpu_hours_per_epoch']:.2f}")

        print(f"\nResults saved to: {result_file}")
        print(f"Throughput log: {throughput_log_file}")
        print("=" * 60 + "\n")

        return results

    return None


if __name__ == "__main__":
    tyro.cli(benchmark)
