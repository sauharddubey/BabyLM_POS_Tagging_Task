#!/bin/bash
#SBATCH --job-name=babylm-preprocess
#SBATCH --time=00:30:00
#SBATCH --output=logs/preprocess-%j.out

echo "=== LRZ Job Preprocessing Start ==="
date

# Activate virtual environment
source venv/bin/activate

# Run preprocessing and download/cache datasets
python preprocess.py

echo "=== LRZ Job Preprocessing Complete ==="
date
