# scripts Directory

This directory contains shell batch jobs and Python utility scripts to support automated training, conversion, and analysis.

## 📂 Directory Contents

- **[`convert_checkpoints.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/scripts/convert_checkpoints.py)**: Formats checkpoint weights into standard HuggingFace `PreTrainedModel` format for external testing/uploads.
- **[`summarize_results.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/scripts/summarize_results.py)**: Compiles multi-model evaluation statistics into summary reports.
- **[`build_pos_vocab_map.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/scripts/build_pos_vocab_map.py)**: Compiles POS tag-to-vocabulary token ID relationships from the preprocessed training dataset corpus.
- **[`job_preprocess.sh`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/scripts/job_preprocess.sh)**: Slurm batch script to preprocess dataset splits.
- **[`job_train_eval.sh`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/scripts/job_train_eval.sh)**: Slurm orchestrator script running training and evaluations.

---

## 🛠️ Implementation Details

### 1. Model Conversion (`convert_checkpoints.py`)
- Standard checkpoints from pretraining contain specialized MultiTask heads. To submit checkpoints or use them on HuggingFace, they must be stripped back to standard `BertModel` or `BertForMaskedLM` wrappers.
- The conversion script instantiates a HuggingFace `BertForMaskedLM` mapping configuration, populates it from the checkpoint weight tensors, and saves it along with standard HuggingFace tokenizer config files.

### 2. Result Summarizer (`summarize_results.py`)
- Evaluates evaluation JSON dumps across runs and compiles them into a markdown summary report (`docs/results_summary.md`).
- Aggregates metrics including WikiText-2 validation perplexity, zero-shot BLiMP accuracy, and downstream fine-tuned GLUE scores.

### 3. POS Vocab Mapping Builder (`build_pos_vocab_map.py`)
- Loops through tokenized training files and computes token-POS occurrences.
- Always allows all BERT special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`) for all POS tags to ensure pretraining syntactic flow.
- Adds tokens that were never observed in the corpus to all POS tags as a fallback.
- Stores the output dictionary as a mapping of POS tag ID to a sorted list of integer token IDs in `data/pos_to_vocab.json`.

### 4. Slurm Cluster Integrations
- **`job_preprocess.sh`**:
  - Sets CPU/memory constraints, logs output/error files, and triggers `preprocess.py`.
- **`job_train_eval.sh`**:
  - Requests hardware resources (GPU), schedules sequential training loops, converts generated model checkpoints, and triggers summaries.
