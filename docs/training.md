# Training Models

This repo exposes the same training code used for the paper across climate, molecular-dynamics, and toy-system experiments.

The trainers are the public entry points. By default they now run with:
- `offline=True`
- `accelerator="auto"`
- `num_devices=1`

For larger runs you will typically override those defaults for your local cluster or workstation setup.

## Environment

Use the full environment from the root README:

```bash
uv sync --no-install-package torch-scatter
uv sync --no-build-isolation
```

Create a dataset root through `.env` or pass a path explicitly in code:

```bash
cp .env.example .env
```

## CLI entry points

All trainer entry points expose their config surface through Tyro:

```bash
uv run --env-file=.env -- python -m exps.ENSO.trainer ENSO_ORAS5 --help
uv run --env-file=.env -- python -m exps.ENSO.trainer ENSO_CESM --help
uv run --env-file=.env -- python -m exps.calixarene.trainer G2 --help
uv run --env-file=.env -- python -m exps.calixarene.trainer G13 --help
uv run --env-file=.env -- python -m exps.trpcage.trainer trp-cage --help
uv run --env-file=.env -- python -m exps.lorenz63.trainer l63 --help
```

## Launching training

Start from the same commands without `--help`:

```bash
uv run --env-file=.env -- python -m exps.ENSO.trainer ENSO_ORAS5
uv run --env-file=.env -- python -m exps.ENSO.trainer ENSO_CESM
uv run --env-file=.env -- python -m exps.calixarene.trainer G2
uv run --env-file=.env -- python -m exps.calixarene.trainer G13
uv run --env-file=.env -- python -m exps.trpcage.trainer trp-cage
uv run --env-file=.env -- python -m exps.lorenz63.trainer l63
```

## Programmatic configuration

If you prefer Python over CLI overrides, call the trainer directly:

```python
from exps.ENSO.trainer import main
from src.configs import defaults

cfg = defaults["ENSO_ORAS5"][1]
cfg.data_args.data_path = "/path/to/datasets"
cfg.offline = False
cfg.accelerator = "cuda"
cfg.num_devices = 1
main(cfg)
```

The same pattern works for the other trainers by importing their `main()` function and selecting the corresponding preset from `src.configs.defaults`.
