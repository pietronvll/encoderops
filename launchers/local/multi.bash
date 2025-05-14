#!/bin/bash

echo "Executing multitask trainers with preset $@"
uv run --env-file=.env -- python -m trainers.multi_task $@ --num_devices=2 --batch_size=32
uv run --env-file=.env -- python -m trainers.multi_task $@ --num_devices=2 --batch_size=32