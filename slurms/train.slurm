#!/bin/bash

#SBATCH --job-name=clapt_llama                                          # Job name
#SBATCH --nodes=4                                                       # Request k node(s)
#SBATCH --gres=gpu:8                                                    # Request 8 GPUs per node
#SBATCH --ntasks-per-node=8                                             # Launch 8 tasks on the node, one per GPU
#SBATCH --partition=training-compute                                    # Specify the GPU partition
#SBATCH --time=00:00:00                                                 # Time limit hrs:min:sec
#SBATCH --output=slurm-logs/embed/clapt_llama_%j.log                    # Standard output and error log

# Activate Poetry environment (ensure your poetry is accessible, or use absolute path to poetry executable)
export PATH="$HOME/.local/bin:$PATH"

# Navigate to your project directory
cd $HOME/clapt

export MODEL_NAME=clapt
export HF_HOME=/app/.cache/huggingface

srun poetry run python clapt/training.py
