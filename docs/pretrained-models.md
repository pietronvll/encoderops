# Pretrained Demos

These notebooks default to the published Hugging Face checkpoints. Each notebook also exposes a small checkpoint-selection cell so you can switch to locally trained weights instead.

## Hugging Face resources

- Models:
  - <https://huggingface.co/CSML-IIT/encoderops>
  - <https://huggingface.co/pnovelli/encoderops>
- Datasets:
  - <https://huggingface.co/datasets/pnovelli/encoderops>
  - ENSO SST files are also available from `CSML-IIT/encoderops`.

## Lorenz63

- Notebook: `exps/lorenz63/analysis.ipynb`
- The surviving public notebook currently uses the model repo `CSML-IIT/encoderops`.
- Lorenz63 checkpoint subdirectories referenced in the notebook include:
  - `lorenz63/EvolutionOperator_dataopt`
  - `lorenz63/VAMPNets`
  - `lorenz63/DPNets`
  - `lorenz63/DAE`
  - `lorenz63/CAE`
- Recommended first action: inspect the training guide, then open the analysis notebook.

## ENSO / SST

- Notebook: `exps/ENSO/analysis.ipynb`
- The current notebook loads the CESM checkpoint from the model repo `CSML-IIT/encoderops` with file `ENSO/EvOp_CESM.pt`.
- Datasets expected by the code:
  - `SST/sst_monthly.nc`
  - `SST/cesm_sst_regridded_1.5deg_850-2005.nc`
- Recommended first action: inspect the notebook and reuse the pretrained checkpoint path instead of retraining.

## Calixarene

- Notebook: `exps/calixarene/analysis.ipynb`
- Checkpoints used in the notebook:
  - `calixarene-G2/checkpoints/last.ckpt`
  - `calixarene-G1+3/checkpoints/last.ckpt`
- Dataset is mirrored on Hugging Face under the `calixarene/` directory.

## TRP-Cage

- Notebook: `exps/trpcage/analysis.ipynb`
- Checkpoint used in the notebook: `encoderops-2J0F/checkpoints/last.ckpt`
- Raw trajectory data is not bundled with the public repo. You still need to request the original DESRES trajectory data separately for full preprocessing or retraining.

## Launching notebooks

After installing the full environment from the root README:

```bash
uv run -- jupyter lab
```

The notebooks have been cleaned so they no longer ship local machine paths or stale execution output. They are intended for exploration, not as the primary onboarding surface.
