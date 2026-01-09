#!/bin/bash
# SLURM launcher for MDCATH benchmark on Leonardo/CINECA
# Usage: sbatch benchmark_launcher.bash --gpus 4 --nodes 2 [other params]
# Or: bash benchmark_launcher.bash (for local testing)

# SLURM Configuration
#SBATCH --account=IscrB_ProAmmo     # project account
#SBATCH --partition=boost_usr_prod  # partition to use
#SBATCH --time=4:00:00              # max time HH:MM:SS
#SBATCH --nodes=2                   # number of nodes (override with --nodes)
#SBATCH --ntasks-per-node=4         # one task per node for distributed training
#SBATCH --gres=gpu:4                # GPUs per node (override with --gpus)
#SBATCH --cpus-per-task=8
#SBATCH --job-name=mdcath-benchmark

############################
# Parse command line arguments
############################

NUM_GPUS=1
NUM_NODES=1
EPOCHS=1
BATCH_SIZE=128
BENCHMARK_NAME="benchmark"
OFFLINE=false
DATALOADER_WORKERS=8

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
        --name)
            BENCHMARK_NAME="$2"
            shift 2
            ;;
        --offline)
            OFFLINE=true
            shift
            ;;
        --workers)
            DATALOADER_WORKERS="$2"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "========== MDCATH Benchmark =========="
echo "GPUs per node: $NUM_GPUS"
echo "Number of nodes: $NUM_NODES"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Benchmark name: $BENCHMARK_NAME"
echo "Dataloader workers: $DATALOADER_WORKERS"
echo "Offline mode: $OFFLINE"
echo "======================================"

############################
# Run benchmark
############################

# Build command
CMD="uv run --env-file=.env -- python -m exps.mdcath.benchmark"
CMD="$CMD --num_gpus=$NUM_GPUS"
CMD="$CMD --num_nodes=$NUM_NODES"
CMD="$CMD --epochs=$EPOCHS"
CMD="$CMD --batch_size=$BATCH_SIZE"
CMD="$CMD --benchmark_name=$BENCHMARK_NAME"
CMD="$CMD --dataloader_workers=$DATALOADER_WORKERS"

if [ "$OFFLINE" = true ]; then
    CMD="$CMD --offline"
fi

# Use srun for distributed training
if command -v srun &> /dev/null && [ ! -z "$SLURM_NODEID" ]; then
    echo "Running with srun for distributed training..."
    srun $CMD
else
    echo "Running locally (srun not available or not in SLURM job)..."
    $CMD
fi
