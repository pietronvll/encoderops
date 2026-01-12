#!/bin/bash
# SLURM launcher for MDCATH benchmark on Leonardo/CINECA
# Usage: sbatch benchmark_launcher.bash --gpus 4 --nodes 2 [other params]
# Or: bash benchmark_launcher.bash (for local testing)

# SLURM Configuration
#SBATCH --account=IscrB_ProAmmo     # project account
#SBATCH --partition=boost_usr_prod  # partition to use
#SBATCH --time=4:00:00              # max time HH:MM:SS
#SBATCH --nodes=2                   # number of nodes (override with --nodes)
#SBATCH --ntasks-per-node=4         # 4 tasks per node (one per GPU)
#SBATCH --gres=gpu:4                # GPUs per node (override with --gpus)
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mdcath-benchmark
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

############################
# Parse command line arguments
############################

NUM_GPUS=4
NUM_NODES=1
EPOCHS=1
BATCH_SIZE=128
DATALOADER_WORKERS=8
TEMPERATURE="348"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --workers)
            DATALOADER_WORKERS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

############################
# Environment setup
############################

export OMP_NUM_THREADS=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "========== MDCATH Benchmark =========="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPUs per node: $NUM_GPUS"
echo "Number of nodes: $NUM_NODES"
echo "Total GPUs: $((NUM_GPUS * NUM_NODES))"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Dataloader workers: $DATALOADER_WORKERS"
echo "Temperature: $TEMPERATURE"
echo "======================================"

############################
# Run benchmark
############################

# Build command
CMD="python -m exps.mdcath.benchmark"
CMD="$CMD --num-gpus=$NUM_GPUS"
CMD="$CMD --num-nodes=$NUM_NODES"
CMD="$CMD --epochs=$EPOCHS"
CMD="$CMD --batch-size=$BATCH_SIZE"
CMD="$CMD --dataloader-workers=$DATALOADER_WORKERS"
CMD="$CMD --temperature=$TEMPERATURE"

echo "Command: $CMD"
echo "======================================"

# Use srun for distributed training on SLURM
if command -v srun &> /dev/null && [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Running with srun for distributed training..."
    uv run --env-file=.env -- srun $CMD
else
    echo "Running locally (not in SLURM job)..."
    uv run --env-file=.env -- $CMD
fi
