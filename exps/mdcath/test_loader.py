import os
from pathlib import Path
from time import perf_counter

from loguru import logger
from tqdm import tqdm

from src.mdcath import MDCATH

if __name__ == "__main__":
    data_path = Path(os.environ["DATA_PATH"]) / "mdcath"
    logger.info(f"Using data path {data_path}")
    domains = ["12asA00", "153lA00", "16pkA02", "1a02F00"]
    mdcath = MDCATH(root=data_path, lagtime=15, pdb_list=domains)
    # Measure throughput:
    start = perf_counter()
    for idx in tqdm(range(len(mdcath))):
        x = mdcath[idx]
        if idx == 0:
            for k, v in x.items():
                print(f"{k}: {v['pos'].shape}")

        # [TODO] check if Hydrogen atoms are present in the dataset and if so remove them from the loader

        # if ((idx + 1) % 1000) == 0:
        #     print(f"Throughput: {1000 / (perf_counter() - start):.2f}samples/s")
        #     start = perf_counter()
