# Benchmark script for measuring MDCATH training time across different GPU/node configurations
# Usage: uv run python -m exps.mdcath.benchmark --num_gpus 4 --num_nodes 2 [additional config args]

from dataclasses import asdict
from time import perf_counter
from typing import Optional

import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults, TrainerArgs
from src.mdcath import MDCATHDataModule
from src.model import EvolutionOperator
from src.modules import SchNet


class BenchmarkCallback(Callback):
    """Callback to log benchmark metrics to wandb."""
    
    def __init__(self):
        self.epoch_start_time = None
        self.total_batches = 0
        self.total_samples = 0
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of training epoch."""
        self.epoch_start_time = perf_counter()
        self.total_batches = 0
        self.total_samples = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        self.total_batches += 1
        if "item" in batch and "z" in batch["item"]:
            self.total_samples += batch["item"]["z"].shape[0]
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of training epoch."""
        if self.epoch_start_time is not None:
            epoch_time = perf_counter() - self.epoch_start_time
            throughput_atoms_per_sec = (self.total_samples) / epoch_time if epoch_time > 0 else 0
            
            # Log benchmark metrics
            metrics = {
                "benchmark/epoch_time_sec": epoch_time,
                "benchmark/total_samples": self.total_samples,
                "benchmark/total_batches": self.total_batches,
                "benchmark/throughput_atoms_per_sec": throughput_atoms_per_sec,
                "benchmark/samples_per_batch": self.total_samples / self.total_batches if self.total_batches > 0 else 0,
                "benchmark/num_gpus": trainer.world_size if trainer.world_size > 0 else 1,
            }
            
            pl_module.log_dict(metrics, on_step=False, on_epoch=True)


@tyro.extras.subcommand_cli_with_default
def benchmark(
    num_gpus: int = tyro.field(
        default=1,
        help="Number of GPUs to use per node"
    ),
    num_nodes: int = tyro.field(
        default=1,
        help="Number of nodes to use"
    ),
    config_name: str = tyro.field(
        default="MDCATH",
        help="Configuration name from defaults (e.g., 'MDCATH')"
    ),
    epochs: int = tyro.field(
        default=1,
        help="Number of epochs to train for"
    ),
    batch_size: int = tyro.field(
        default=128,
        help="Batch size for training"
    ),
    benchmark_name: Optional[str] = tyro.field(
        default=None,
        help="Name for the benchmark run (optional tag for wandb)"
    ),
    offline: bool = tyro.field(
        default=False,
        help="Run in offline mode"
    ),
    dataloader_workers: int = tyro.field(
        default=8,
        help="Number of dataloader workers"
    ),
):
    """
    Benchmark MDCATH training with specified GPU and node configuration.
    
    This script trains the SchNet model on MDCATH data and logs wall-clock time,
    throughput, and other metrics to wandb for benchmarking purposes.
    """
    
    # Load base configuration
    if config_name not in defaults:
        raise ValueError(f"Config '{config_name}' not found in defaults. Available: {list(defaults.keys())}")
    
    _, base_config = defaults[config_name]
    
    # Override with benchmark parameters
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
        data_args=base_config.data_args,
        wandb_project=base_config.wandb_project,
        wandb_entity=base_config.wandb_entity,
        offline=offline,
        num_devices=num_gpus,
        num_nodes=num_nodes,
        dataloader_workers=dataloader_workers,
    )
    
    # Setup data module
    datamodule = MDCATHDataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.setup("fit")
    
    # Setup wandb logger with benchmark tags
    wandb_tags = ["benchmark", f"gpus-{num_gpus}", f"nodes-{num_nodes}"]
    if benchmark_name:
        wandb_tags.append(benchmark_name)
    
    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
        tags=wandb_tags,
        notes=f"Benchmark: {num_gpus} GPUs x {num_nodes} nodes, batch_size={batch_size}, epochs={epochs}",
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1, save_top_k=-1, save_last=True
    )
    benchmark_callback = BenchmarkCallback()
    
    # Setup trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, benchmark_callback],
        strategy="ddp" if cfg.trainer_args.loss in ["kl_DV", "kl_NWJ", "l2"] else "ddp_find_unused_parameters_true",
        accelerator="cuda",
        devices=cfg.num_devices,
        num_nodes=cfg.num_nodes,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    
    # Setup model
    encoder_args = {
        "n_out": cfg.trainer_args.latent_dim,
        "cutoff": cfg.data_args.cutoff_ang,
        "atomic_numbers": datamodule.dataset.z_table.zs,
    }
    encoder_args = encoder_args | asdict(cfg.model_args)
    model = EvolutionOperator(SchNet, encoder_args, cfg.trainer_args)
    
    # Log configuration to wandb
    wandb_logger.log_hyperparams({
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "batch_size": batch_size,
        "epochs": epochs,
        "dataloader_workers": dataloader_workers,
        "latent_dim": cfg.trainer_args.latent_dim,
        "encoder_lr": cfg.trainer_args.encoder_lr,
        "linear_lr": cfg.trainer_args.linear_lr,
    })
    
    # Time the full training
    start_time = perf_counter()
    trainer.fit(model, datamodule=datamodule)
    total_time = perf_counter() - start_time
    
    # Log final summary metrics
    summary_metrics = {
        "benchmark/total_training_time_sec": total_time,
        "benchmark/time_per_epoch_sec": total_time / cfg.trainer_args.epochs if cfg.trainer_args.epochs > 0 else 0,
    }
    wandb_logger.log_metrics(summary_metrics, step=0)
    
    print(f"\n{'='*60}")
    print(f"Benchmark Summary:")
    print(f"{'='*60}")
    print(f"Configuration: {num_gpus} GPUs x {num_nodes} nodes")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Time per epoch: {total_time / cfg.trainer_args.epochs if cfg.trainer_args.epochs > 0 else 0:.2f} seconds")
    print(f"{'='*60}\n")
    
    return total_time


if __name__ == "__main__":
    benchmark()
