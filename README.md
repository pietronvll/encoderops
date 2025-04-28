1. Load data using their dataloader and lagged dataloader 
2. Model can be trained. Create proper training script


## Notes
### Installation
```bash
uv pip install torch setuptools
uv pip install torch-scatter --no-build-isolation
uv sync
```

## Dataset preprocessing
To be used, the raw trajectories downloaded from De Shaw Research must be preprocessed. 

1. First we need to generate a `.pdb` file for the trajectory. So far I've done it manually using [VMD](https://www.ks.uiuc.edu/Research/vmd/).
2. Then we convert the dataset into a `.lmdb` for multithread loading. This can be done via the `uv run --env-file=.env -- python -m scripts.to_lmdb --traj_id=<CLN025-0>`. Pass the `--help` keyword to the above command for the complete list of available arguments.