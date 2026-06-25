# BabyLM Multi-Task BERT Pretraining & Evaluation Suite

This repository implements a modular, multi-task pretraining and evaluation pipeline for BERT models tailored to the **BabyLM 2026 Challenge (Strict-Small track)**.

The suite supports training BERT-Mini models with combinations of three pretraining tasks:
1. **Masked Language Modeling (MLM)** (Standard token prediction)
2. **Next Sentence Prediction (NSP)** (Sentence pair relationship prediction)
3. **Part-of-Speech (POS) Tagging** (Word-level syntactic prediction utilizing subword-level average pooling)

---

## 📂 Repository Structure

The codebase is organized as follows:

```
├── main.py                    # Main pipeline orchestrator (Stage runner)
├── train.py                   # Custom pretraining script
├── eval.py                    # Evaluation script (PPL, BLiMP, and GLUE)
├── preprocess.py              # Download, tokenizer, POS tagger & caching script
│
├── src/                       # Core package modules
│   ├── dataset.py             # Pretrain/downstream datasets & collators
│   └── models.py              # MultiTaskBERT model architecture & heads
│
├── scripts/                   # Script entries and jobs
│   ├── convert_checkpoints.py # Export best checkpoints to HuggingFace format
│   ├── verify_pipeline.py     # Verify pipeline modules and layers
│   ├── summarize_results.py   # Aggregate official BabyLM eval results
│   ├── job_preprocess.sh      # Slurm pre-processing script
│   └── job_train_eval.sh      # Slurm pretraining orchestrator script
│
├── data/                      # Preprocessed corpora, dictionaries (gitignored)
│   ├── preprocessed_train/    # Word-level tokenized train corpus
│   ├── preprocessed_val/      # Word-level tokenized validation corpus
│   ├── eval_data/             # Cached GLUE, WikiText-2, and BLiMP datasets
│   ├── nltk_data/             # Local NLTK tokenizer/tagger resources
│   └── pos_to_id.json         # Vocabulary map for POS labels
│
├── checkpoints/               # Local training checkpoints (gitignored)
├── logs/                      # Slurm output log files (gitignored)
├── babylm-eval/               # Official BabyLM eval pipeline (gitignored, clone separately — see Setup)
├── reports/                   # Compiled evaluation markdown files
│   ├── results_summary.md     # Aggregated downstream official results
│   ├── evaluation_report.md   # Comparative analysis of model variants
│   └── PipelineExplanation.md # Explanation of the multi-task setup
└── hf_models/                 # HuggingFace-compatible export models (gitignored)
```

---

## ⚙️ Setup Instructions

### 1. Environment Setup
Create and activate a virtual environment, then install the dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Clone the Official BabyLM Evaluation Pipeline
The official BabyLM evaluation suite must be cloned separately into the project root. It is **gitignored** because it is an independent subproject:
```bash
git clone https://github.com/babylm/evaluation-pipeline-2024.git babylm-eval
# Then follow its own README for dependency setup
pip install -r babylm-eval/requirements.txt
```

### 3. Preprocess Data and Cache Evaluators
Download the BabyLM corpus, run NLTK tokenization and POS tagging, and download/cache GLUE tasks, BLiMP, and WikiText-2:
```bash
# Direct execution
python preprocess.py

# Via Slurm
sbatch scripts/job_preprocess.sh
```

---

## 🚀 Running the Pipeline

### 1. Orchestrated Training and Evaluation
To run the full suite (runs all 8 model variants sequentially through training, offline evaluation, and report compilation):
```bash
# Direct execution
python main.py --stage all --epochs 5 --batch_size 64

# Via Slurm
sbatch scripts/job_train_eval.sh
```

### 2. Run Individual Model Pretraining
You can pretrain a model with specific objectives by running `train.py`:
```bash
python train.py \
    --tasks mlm,pos \
    --model_dir ./checkpoints/mlmposbert \
    --epochs 5 \
    --batch_size 64 \
    --lr 5e-5 \
    --alpha 1.0
```

### 3. Run Individual Offline Evaluation
To run the validation evaluation (Perplexity, BLiMP, and GLUE) on a specific checkpoint:
```bash
python eval.py \
    --checkpoint_path ./checkpoints/mlmposbert/checkpoint_best.pt \
    --tasks perplexity,blimp,glue
```

---

## 📊 Evaluation & Results Compilation

Once pretraining is complete:
1. **Convert checkpoints** to HuggingFace format for submission/evaluation:
   ```bash
   python scripts/convert_checkpoints.py
   ```
2. Run the **official evaluation pipeline** (located in `babylm-eval/`).
3. **Compile and summarize results** across all trained models:
   ```bash
   python scripts/summarize_results.py
   ```
   The results will be written to `reports/results_summary.md`.

---

## 🛠️ Verification and Testing
To run the automated test suite that validates dataset collation, word-level average pooling dimensions, and backward compatibility:
```bash
python scripts/verify_pipeline.py
```
