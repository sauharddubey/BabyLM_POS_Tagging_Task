# Codebase Context & Developer's Manual: BabyLM Challenge

This document stores the complete developer context of the structured BabyLM Challenge codebase. It serves as a guide for future agents or pair-programmers to understand the directory layout, optimizations, configurations, and pipeline orchestrations.

---

## 📂 Project Directory Structure

```text
BabyLM_Challenge/
├── src/
│   ├── models.py       # MultiTaskBERT and prediction heads
│   └── dataset.py      # All dataset wrappers, collators, and helpers
├── preprocess.py       # Download datasets, POS tagger (NLTK), and offline caching
├── train.py            # Pretraining script entry point
├── eval.py             # Evaluation script entry point (GLUE, BLiMP, BLEU, Perplexity)
├── main.py             # Pipeline orchestrator / Driver script
├── requirements.txt    # Project dependencies
├── job_train_eval.sh   # Slurm pretraining script
├── PipelineExplaination.md # Detailed step-by-step math and example walkthrough
└── Agent_Files/
    └── Agent.md        # [This File] Codebase developer context
```

---

## 🧩 Key Codebase Components

### 1. `src/models.py`
Contains the [MultiTaskBERT](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L20) class.
* **Base Encoder:** A customized `BertModel` scaled to `BERT-Mini` dimensions (6 hidden layers, 8 attention heads, 256 hidden dimension).
* **Heads:**
  - MLM Head (tied weight projection)
  - NSP Head (2-class classification)
  - POS Classifier (45-class classification)
* **Optimization Highlights:**
  - **Vectorized subword pooling:** Uses batched matrix multiplication (`torch.bmm`) with normalized masks to mean-pool subword representations into word-level vectors on GPU.
  - **Vectorized tag extraction:** Gathers POS labels from token-level representations using monotonic `argmax` index extraction.
  - **Weight Tying:** Shares the embedding parameters with the final MLM head output linear layer.

### 2. `src/dataset.py`
Encapsulates all datasets and collators.
* **PretrainDataset:** Pairs adjacent sentences (IsNext/NotNext) for the Next Sentence Prediction task.
* **MultiTaskCollator:** Performs dynamic masking (80/10/10 rule) on the fly, tokenizes pre-split word lists, and aligns word-level POS tags to WordPiece tokens.
* **Downstream Helpers:** Contains `GLUEDataset`, `TranslationDataset` (Multi30k), and respective collators.

### 3. `train.py`
Runs pretraining.
* Takes command line flags (`--tasks`, `--model_dir`, `--epochs`, `--batch_size`, `--lr`, `--alpha`, `--resume`).
* **Optimization Highlight:** Configures grouped parameters for the `AdamW` optimizer, excluding LayerNorm weights and biases from weight decay.
* Handles checkpointing: Saves latest (`checkpoint_latest.pt`) and best (`checkpoint_best.pt`) model configurations.

### 4. `eval.py`
Evaluates checkpoints.
* **Pseudo-Perplexity (PPL):** Measures MLM sequence likelihood on WikiText-2 and BabyLM validation splits. Batched in parallel on GPU for 30x speedups.
* **BLiMP:** Zero-shot grammatical accuracy on minimal sentence pairs (e.g., SVA, anaphor gender agreement). Batched in parallel.
* **GLUE:** Fine-tunes a sequence classification classifier on MRPC, CoLA, and SST-2.
* **BLEU:** Fine-tunes an `EncoderDecoderModel` on Multi30k. Uses English vocabulary configuration on target decoder to reduce parameters to 15M (preventing low-resource generative overfitting).

### 5. `main.py`
Serves as the high-level orchestrator.
* Configures 5 model variants (e.g. `BASEBERT`, `POSBERT`, `MLM_POS_NSPBERT`).
* Automatically runs preprocessing, pretraining, offline evaluations, and report compiling sequentially.
* Compiles results into a Markdown report `evaluation_report.md`.

---

## 🏃 How to Run the Pipeline

### Virtual Environment Activation:
```bash
source venv/bin/activate
```

### Run the Orchestrator Pipeline (Sequential Stages):
```bash
python main.py --stage all --epochs 5 --batch_size 64 --lr 5e-5 --alpha 1.0
```
*Choose individual stages:* `preprocess`, `train`, `eval`, `report`, or `all`.

### Run Standalone Pretraining (Example: MLM + POS tasks):
```bash
python train.py --tasks mlm,pos --model_dir ./checkpoints_mlmposbert --epochs 5 --batch_size 64 --resume
```

### Run Standalone Evaluation on a Checkpoint:
```bash
python eval.py --checkpoint_path ./checkpoints_mlmposbert/checkpoint_best.pt --tasks perplexity,blimp,glue,bleu
```

---

## 📋 SLURM Integration

The pretraining is designed to run on high-performance compute clusters using the Slurm script [job_train_eval.sh](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/job_train_eval.sh).

Submit the pretraining job:
```bash
sbatch job_train_eval.sh
```

Cancel a job:
```bash
scancel <job_id>
```

Monitor logs:
```bash
tail -f train-eval-<job_id>.out
```
