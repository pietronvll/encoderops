#!/bin/bash

echo "Executing: uv run train.py $@"
uv run --env-file=.env -- python -m trainers.single_task $@ --num_devices=2 --data_args.batch_size=128