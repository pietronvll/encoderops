### Installation
[Install uv](https://docs.astral.sh/uv/), and from the repo root just type:
```bash
uv sync --no-install-package torch-scatter
uv sync --no-build-isolation
```
## Training commands:
Create a `.venv` file in the root of the repo to define the `DATA_PATH` environment variable. The datasets will be downloaded / processed in this path. 
```
DATA_PATH = your/dataset/path
```
From the root of the repo run:
### Lorenz63 
```
uv run --env-file=.venv -- python -m exps.lorenz63.trainer l63 --help
```
### TRP-CAGE (protein-folding)
Before running this experiment, you need to obtain a copy of the data by requesting it at [this webpage](https://www.deshawresearch.com/downloads/download_trajectory_science2011.cgi/). Then, follow these steps to preprocess the data:
- Once downloaded, extract the `DESRES-Trajectory_2JOF-0-protein.tar.xz` file inside the dataset folder as defined in `DATA_PATH`.
- Copy the topology file `exps/trpcage/2JOF-0-protein.pdb`at the location `DATA_PATH/DESRES-Trajectory_2JOF-0-protein/2JOF-0-protein/2JOF-0-protein.pdb`.
- From the root of the repository run the following command:
```
uv run --env-file=.env -- python -m scripts.to_lmdb --protein-id 2JOF
```
It is _strongly advised_ to have the `DATA_PATH` on an SSD. Once the data has been preprocessed, just run
```
uv run --env-file=.venv -- python -m exps.trpcage.trainer trp-cage --help
```
### Calixarene (ligand-binding) 
```
uv run --env-file=.venv -- python -m exps.calixarene.trainer G2 --help # G2 ligand
uv run --env-file=.venv -- python -m exps.calixarene.trainer G13 --help # G1 + G3 ligands
```
#### ENSO 
```
uv run --env-file=.venv -- python -m exps.ENSO.trainer ENSO --help
```