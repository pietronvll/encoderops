#!/bin/bash
#SBATCH --account=IscrC_LR4LSDS     # project name
#SBATCH --partition=boost_usr_prod  # partition to be used
#SBATCH --time 4:00:00              # format: HH:MM:SS
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=4         # tasks out of 32
#SBATCH --gres=gpu:4                # gpus per node out of 4
#SBATCH --cpus-per-task=8
############################

export OMP_NUM_THREADS=1

echo "========== calixarene training script $@=========="
uv run --env-file=.env -- srun python -m exps.calixarene.trainer $@ --trainer_args.batch_size=256  --offline