import torch

import torch
import tyro
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.configs import Configs, defaults
from src.data import Lorenz63DataModule
from src.modules import MLP

from kooplearn.models import Nonlinear
from kooplearn.models.feature_maps.nn import NNFeatureMap
from kooplearn.data import traj_to_contexts
from kooplearn.nn import VAMPLoss
from torch.utils.data import DataLoader
from kooplearn.nn.data import collate_context_dataset

from dataclasses import asdict
from time import perf_counter
import pickle

def main(cfg: Configs):
    datamodule = Lorenz63DataModule(
        cfg.trainer_args, cfg.data_args, cfg.dataloader_workers
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_data = datamodule.train_dataset.data
    train_ctxs = traj_to_contexts(train_data.astype('float32'), time_lag=cfg.data_args.lagtime, backend='numpy')
    train_dl = DataLoader(train_ctxs, batch_size = cfg.trainer_args.batch_size, shuffle=cfg.data_args, collate_fn=collate_context_dataset, num_workers=4)

    wandb_logger = WandbLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        offline=cfg.offline,
        save_dir="./logs",
        tags=["VAMPNets"],
        name=f"VAMPNets_{cfg.trainer_args.seed}",    
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=100, save_top_k=-1, save_last=True
    )
    # Trainer
    trainer = Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="cuda",
        devices=cfg.num_devices,
        max_epochs=cfg.trainer_args.epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    num_vars = datamodule.train_dataset.num_variables
    # Model
    encoder_args = {
        "input_shape": num_vars * (cfg.data_args.history_len + 1),
        "output_shape": cfg.trainer_args.latent_dim,
        "dropout": 0.0,
        "activation": torch.nn.ReLU,
        "iterative_whitening": False,
        "bias": True,
    }
    loss_args = {
        'schatten_norm': 2,
        'center_covariances': False
        }
    encoder_args = encoder_args | asdict(cfg.model_args)
    feature_map = NNFeatureMap(
        MLP,
        VAMPLoss,
        torch.optim.AdamW,
        trainer,
        encoder_kwargs=encoder_args,
        loss_kwargs=loss_args,
        optimizer_kwargs={'lr': 1e-4},
        seed=cfg.trainer_args.seed,
    )    
    start = perf_counter()
    feature_map.fit(train_dl)
    nn_model = Nonlinear(feature_map, reduced_rank = False, rank=cfg.trainer_args.latent_dim+num_vars).fit(train_ctxs)
    runtime = perf_counter() - start
    # nn_model.save(f'L63_VAMPNets_rep{cfg.trainer_args.seed}')
    with open(f'L63_VAMPNets_rep{cfg.trainer_args.seed}.pkl', 'wb') as f:
        pickle.dump(feature_map, f)
    wandb_logger.experiment.log({"runtime": runtime})

if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(defaults)
    main(config)