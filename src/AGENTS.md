# src Directory

This directory contains the core PyTorch dataset loaders, collators, and transformer architectures.

## 📂 Directory Contents

- **[`dataset.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py)**: Defines dataset structures and customized data collators for pretraining and fine-tuning.
- **[`models.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py)**: Defines model architectures, multi-task prediction heads, and the layered POS vocabulary reduction layer.

---

## 🛠️ Code Implementations & Architecture

### 1. Dataset & Collators (`dataset.py`)
- **`PretrainDataset`**:
  - Handles dynamic Next Sentence Prediction (NSP) sentence pairing (50% positive matches, 50% randomly sampled negative matches).
- **`MultiTaskCollator`**:
  - Performs dynamic masking (80% mask tokens, 10% random, 10% original) on the fly for Masked Language Modeling (MLM).
  - Aligns WordPiece subword tokens back to their parent word-level POS tags.
- **`GLUEDataset`**:
  - Custom dataset class for fine-tuning evaluations on downstream GLUE tasks (CoLA, SST-2, MRPC).
- **`GLUECollator`**:
  - Formats sequences for downstream task evaluations.
- **`TranslationDataset` & `TranslationCollator`**:
  - Support sequence-to-sequence translation tasks (Multi30k) for evaluation.

### 2. Multi-Task Model Architectures (`models.py`)
- **`MultiTaskBERT`**:
  - The standard model architecture utilizing a custom scaled `BertModel` (6 layers, 8 heads, 256 embedding dimension) with standard BERT-Mini configuration.
  - Features three parallel forward task heads: MLM Head (weight-tied with input embeddings), NSP Head (binary classifier), and POS Tagging Head (45 Penn Treebank classes).
  - Implements **Vectorized Subword-level Average Pooling**: average pools subword representations back into word-level tensors to align with word-level POS labels.
- **`LayeredPOSMLMBert`**:
  - The experimental layered model architecture.
  - Implements soft, hard, and gold POS-to-Vocabulary constraints:
    - **Gold**: Restricts vocabulary predictions based on actual ground-truth POS tags.
    - **Hard**: Uses predicted POS argmax indices to mask forbidden vocab tokens.
    - **Soft**: Filters vocabulary predictions using the probability distribution output of the POS head.
