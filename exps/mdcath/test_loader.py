from time import perf_counter

from tqdm import tqdm

from src.configs import MDCATHDataArgs, TrainerArgs
from src.mdcath import MDCATHDataModule

if __name__ == "__main__":
    pdb_list = [
        "12asA00",
        "153lA00",
        "16pkA02",
        "1a02F00",
        "1a05A00",
        "1a0aA00",
    ]
    data_configs = MDCATHDataArgs(pdb_list=pdb_list)
    trainer_args = TrainerArgs(
        latent_dim=64,
        encoder_lr=1e-2,
        linear_lr=1e-2,
        epochs=45,
        batch_size=64,
        max_grad_norm=0.2,
        normalize_lin=False,
        regularization=1e-5,
    )

    datamodule = MDCATHDataModule(trainer_args, data_configs, num_workers=4)
    datamodule.setup("fit")
    atoms_processed = 0
    start = perf_counter()
    for batch in tqdm(datamodule.train_dataloader()):
        atoms_processed += batch["item"]["z"].shape[0]
    print(
        f"Throughput: {(atoms_processed / 1e6) / (perf_counter() - start):.2f}Matoms/s"
    )

    # # Measure throughput:
    # start = perf_counter()
    # for idx in tqdm(range(len(mdcath))):
    #     x = mdcath[idx]
    #     if idx == 0:
    #         for k, v in x.items():
    #             print(f"{k}: {v['pos'].shape}")

    #     # [TODO] check if Hydrogen atoms are present in the dataset and if so remove them from the loader

    #     # if ((idx + 1) % 1000) == 0:
    #     #     print(f"Throughput: {1000 / (perf_counter() - start):.2f}samples/s")
    #     #     start = perf_counter()
