#!/bin/bash
#SBATCH --account=IscrC_LR4LSDS        # project name
#SBATCH --partition=boost_usr_prod  # partition to be used
#SBATCH --time 4:00:00             # format: HH:MM:SS
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=1         # tasks out of 32
#SBATCH --gres=gpu:1                # gpus per node out of 4
#SBATCH --cpus-per-task=8
############################

export OMP_NUM_THREADS=1

echo "Executing: uv run train.py $@"
uv run --env-file=.env -- python -m trainers.single_task $@ --num_devices=1 --data_args.batch_size=1024  