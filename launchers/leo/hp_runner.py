#!/usr/bin/env python3
"""
Simple script to submit multiple SLURM jobs for hyperparameter search.
"""

import subprocess
import os
import itertools
from pathlib import Path

# Define hyperparameter options
hyperparams = {
    'max_grad_norm': ['0.2', 'none'],
    'normalize_lin': ['true', 'false'],
    'min_encoder_lr': ['none', '0.0001']
}

# Generate all combinations of hyperparameters
keys = hyperparams.keys()
combinations = list(itertools.product(*(hyperparams[key] for key in keys)))
total_jobs = len(combinations)

print(f"Launching hyperparameter search with {total_jobs} total jobs")

# Submit a job for each combination
for job_idx, combo in enumerate(combinations, 1):
    # Create a dictionary of the current hyperparameter values
    current_params = dict(zip(keys, combo))
    
    # Prepare arguments
    cmd_args = ["chignolin-prod"]
    
    # Add overrides based on hyperparameter values
    if current_params['max_grad_norm'] == 'none':
        cmd_args.append('--model_args.max_grad_norm=None')
    else:
        cmd_args.append(f'--model_args.max_grad_norm={current_params["max_grad_norm"]}')
    
    # Handle boolean flag properly
    if current_params['normalize_lin'] == 'true':
        cmd_args.append('--model_args.normalize-lin')
    else:
        cmd_args.append('--model_args.no-normalize-lin')
    
    if current_params['min_encoder_lr'] == 'none':
        cmd_args.append('--model_args.min-encoder-lr=None')
    else:
        cmd_args.append(f'--model_args.min-encoder-lr={current_params["min_encoder_lr"]}')
    
    # Print job information
    param_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
    print(f"[{job_idx}/{total_jobs}] Submitting job with: {param_str}")
    
    # Construct sbatch command with the launcher script and arguments
    sbatch_command = ["sbatch", "launchers/leo/launcher.sbatch"] + cmd_args
    
    # Submit the job
    subprocess.run(sbatch_command)

print(f"All {total_jobs} jobs submitted to SLURM queue.")
print("Use 'squeue -u $USER' to monitor your jobs.")