#!/usr/bin/env python3
"""
Scaling Study Job Submitter for MDCATH Benchmarking

Submits a series of benchmark jobs with increasing GPU/node configurations
to perform a comprehensive scaling study.

Usage:
    python exps/mdcath/submit_scaling_study.py [OPTIONS]

Examples:
    # Dry-run to see what would be submitted
    python exps/mdcath/submit_scaling_study.py --dry-run

    # Submit all jobs with custom batch size
    python exps/mdcath/submit_scaling_study.py --batch-size 256

    # Submit with offline W&B mode
    python exps/mdcath/submit_scaling_study.py --offline
"""

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple


class ScalingStudySubmitter:
    """Submit scaling study benchmarks to SLURM."""
    
    def __init__(
        self,
        epochs: int = 1,
        batch_size: int = 128,
        workers: int = 8,
        dry_run: bool = False,
        offline: bool = False,
        job_prefix: str = "scaling_study",
        account: str = "IscrB_ProAmmo",
        partition: str = "boost_usr_prod",
        max_time: str = "2:00:00",
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.workers = workers
        self.dry_run = dry_run
        self.offline = offline
        self.job_prefix = job_prefix
        self.account = account
        self.partition = partition
        self.max_time = max_time
        self.submitted_jobs: List[Tuple[str, str]] = []
        
        # Get repository root
        self.repo_root = Path(__file__).parent.parent.parent
    
    def submit_job(
        self,
        nodes: int,
        gpus_per_node: int,
        job_suffix: str,
        description: str,
    ) -> None:
        """Submit a single benchmark job."""
        
        job_name = f"{self.job_prefix}_{job_suffix}"
        total_gpus = nodes * gpus_per_node
        
        # Build sbatch command
        sbatch_cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--nodes={nodes}",
            f"--gres=gpu:{gpus_per_node}",
            "--ntasks-per-node=1",
            "--cpus-per-task=8",
            f"--account={self.account}",
            f"--partition={self.partition}",
            f"--time={self.max_time}",
            "exps/mdcath/benchmark_launcher.bash",
            f"--gpus={gpus_per_node}",
            f"--nodes={nodes}",
            f"--epochs={self.epochs}",
            f"--batch-size={self.batch_size}",
            f"--workers={self.workers}",
            f"--name={job_suffix}",
        ]
        
        if self.offline:
            sbatch_cmd.append("--offline")
        
        cmd_str = " ".join(sbatch_cmd)
        
        print(f"Submitting: {description}")
        print(f"  Total GPUs: {total_gpus} ({nodes} node(s) × {gpus_per_node} GPU(s))")
        print(f"  Command: {cmd_str}")
        
        if self.dry_run:
            print(f"  Status:  [DRY-RUN - Not submitted]")
        else:
            try:
                # Change to repo directory
                import os
                original_cwd = os.getcwd()
                os.chdir(self.repo_root)
                
                result = subprocess.run(
                    cmd_str,
                    shell=True,
                    capture_output=True,
                    text=True,
                )
                
                os.chdir(original_cwd)
                
                if result.returncode == 0:
                    # Extract job ID from output
                    output = result.stdout + result.stderr
                    for line in output.split("\n"):
                        if "Submitted batch job" in line:
                            job_id = line.split()[-1]
                            print(f"  Status:  ✓ Submitted (Job ID: {job_id})")
                            self.submitted_jobs.append((job_suffix, job_id))
                            break
                else:
                    print(f"  Status:  ✗ Failed")
                    print(f"  Error: {result.stderr}")
            except Exception as e:
                print(f"  Status:  ✗ Error: {e}")
        
        print()
    
    def run_scaling_study(self) -> None:
        """Run the complete scaling study."""
        
        print("=" * 50)
        print("MDCATH Scaling Study Job Submitter")
        print("=" * 50)
        print(f"Epochs:        {self.epochs}")
        print(f"Batch size:    {self.batch_size}")
        print(f"Workers:       {self.workers}")
        print(f"Account:       {self.account}")
        print(f"Partition:     {self.partition}")
        print(f"Max time:      {self.max_time}")
        print(f"Job prefix:    {self.job_prefix}")
        print(f"Dry run:       {self.dry_run}")
        print(f"Offline mode:  {self.offline}")
        print("=" * 50)
        print()
        
        print("Submitting benchmark jobs...")
        print()
        
        # Submit jobs: (nodes, gpus_per_node, suffix, description)
        self.submit_job(1, 1,  "1gpu_1node",   "1 GPU on 1 node")
        self.submit_job(1, 2,  "2gpu_1node",   "2 GPUs on 1 node")
        self.submit_job(1, 3,  "3gpu_1node",   "3 GPUs on 1 node")
        self.submit_job(1, 4,  "4gpu_1node",   "4 GPUs on 1 node")
        self.submit_job(2, 4,  "8gpu_2nodes",  "8 GPUs on 2 nodes (4 each)")
        self.submit_job(3, 4,  "12gpu_3nodes", "12 GPUs on 3 nodes (4 each)")
        self.submit_job(4, 4,  "16gpu_4nodes", "16 GPUs on 4 nodes (4 each)")
        
        # Print summary
        print("=" * 50)
        print("Summary")
        print("=" * 50)
        
        if self.dry_run:
            print("DRY-RUN mode: No jobs were submitted")
            print("Remove --dry-run flag to submit actual jobs")
        else:
            print(f"Total jobs submitted: {len(self.submitted_jobs)}")
            
            if self.submitted_jobs:
                print()
                print("Job IDs:")
                for i, (suffix, job_id) in enumerate(self.submitted_jobs, 1):
                    print(f"  [{i}] {suffix:20s} → {job_id}")
                
                print()
                print("Monitoring commands:")
                print("  Check status:  squeue -u $USER")
                print("  Check all:     sinfo -N -l")
                print("  Job details:   scontrol show job <job_id>")
                print("  Job output:    tail -f logs/slurm-<job_id>.out")
        
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Submit scaling study benchmarks to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without submitting",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline W&B mode",
    )
    parser.add_argument(
        "--name",
        default="scaling_study",
        help="Prefix for job names (default: scaling_study)",
    )
    parser.add_argument(
        "--account",
        default="IscrB_ProAmmo",
        help="SLURM account (default: IscrB_ProAmmo)",
    )
    parser.add_argument(
        "--partition",
        default="boost_usr_prod",
        help="SLURM partition (default: boost_usr_prod)",
    )
    parser.add_argument(
        "--time",
        default="2:00:00",
        help="Max time per job (default: 2:00:00)",
    )
    
    args = parser.parse_args()
    
    submitter = ScalingStudySubmitter(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
        offline=args.offline,
        job_prefix=args.name,
        account=args.account,
        partition=args.partition,
        max_time=args.time,
    )
    
    submitter.run_scaling_study()


if __name__ == "__main__":
    main()
