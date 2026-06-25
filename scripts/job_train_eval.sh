#!/bin/bash
#SBATCH --job-name=babylm-train-eval
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-eval-%j.out

echo "=== LRZ Job Pretraining and Evaluation Start ==="
date

# Print GPU information
nvidia-smi

# Activate virtual environment
source venv/bin/activate

# Disable tqdm progress bars to keep Slurm logs clean
export TQDM_DISABLE=1

# Run orchestrator - stage 'all' runs training, evaluations, and compiles the report
python -u main.py --stage all --epochs 5 --batch_size 64 --lr 5e-5 --alpha 1.0

echo ">>> Converting checkpoints to Hugging Face format..."
python scripts/convert_checkpoints.py

echo ">>> Submitting official BabyLM evaluation pipeline job..."
sbatch babylm-eval/strict/job_eval_all.sh

echo "=== LRZ Job Pretraining and Evaluation Complete ==="
date



