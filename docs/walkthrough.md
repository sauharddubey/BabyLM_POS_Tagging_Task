# BabyLM Multi-Task BERT — Comprehensive Walkthrough

This document provides an end-to-end, step-by-step technical description of everything that happens inside the BabyLM Multi-Task pretraining and evaluation suite — from raw text all the way to final evaluation scores — with concrete examples for each stage and every modeling objective.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 1 — Data Preprocessing (`preprocess.py`)](#2-stage-1--data-preprocessing-preprocesspy)
3. [Stage 2 — Dataset & Collation (`src/dataset.py`)](#3-stage-2--dataset--collation-srcdatasetpy)
   - [3.1 NSP Pair Construction](#31-nsp-pair-construction)
   - [3.2 Tokenization & Encoding](#32-tokenization--encoding)
   - [3.3 MLM Masking Strategy](#33-mlm-masking-strategy)
   - [3.4 Word-ID Alignment for POS](#34-word-id-alignment-for-pos)
4. [Stage 3 — Model Architecture (`src/models.py`)](#4-stage-3--model-architecture-srcmodelspy)
   - [4.1 The Base BERT Encoder](#41-the-base-bert-encoder)
   - [4.2 MLM Head](#42-mlm-head)
   - [4.3 NSP Head](#43-nsp-head)
   - [4.4 POS Head with Subword Pooling](#44-pos-head-with-subword-pooling)
5. [Stage 4 — Loss Calculation per Objective](#5-stage-4--loss-calculation-per-objective)
   - [5.1 MLM Loss](#51-mlm-loss)
   - [5.2 NSP Loss](#52-nsp-loss)
   - [5.3 POS Loss](#53-pos-loss)
   - [5.4 Combined Loss Aggregation](#54-combined-loss-aggregation)
6. [Stage 5 — Training Loop (`train.py`)](#6-stage-5--training-loop-trainpy)
7. [Stage 6 — Evaluation (`eval.py`)](#7-stage-6--evaluation-evalpy)
   - [6.1 Pseudo-Perplexity (PPL)](#61-pseudo-perplexity-ppl)
   - [6.2 BLiMP Zero-Shot Evaluation](#62-blimp-zero-shot-evaluation)
   - [6.3 GLUE Downstream Fine-Tuning](#63-glue-downstream-fine-tuning)
   - [6.4 BLEU Machine Translation](#64-bleu-machine-translation)
8. [Model Variants & Their Differences](#8-model-variants--their-differences)
9. [End-to-End Example: One Training Step](#9-end-to-end-example-one-training-step)

---

## 1. Pipeline Overview

The pipeline is orchestrated by [`main.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/main.py) and runs four sequential stages:

```
Raw BabyLM Text
      │
      ▼
[preprocess.py]  ── NLTK tokenization + POS tagging + HuggingFace caching
      │
      ▼
[train.py]       ── Multi-task BERT pretraining (MLM / NSP / POS)
      │
      ▼
[eval.py]        ── Perplexity, BLiMP, GLUE, BLEU evaluation
      │
      ▼
[main.py]        ── Compile markdown comparison table
```

Eight model variants are trained in sequence, differing only in which subset of `{mlm, nsp, pos}` objectives are active.

---

## 2. Stage 1 — Data Preprocessing (`preprocess.py`)

**File:** [`preprocess.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/preprocess.py)

### What it does

The BabyLM 2026 Strict-Small corpus (`BabyLM-community/BabyLM-2026-Strict-Small`) is a ~10 million word dataset of child-directed speech and related text. Each example is a raw text string.

**Step-by-step:**

1. **NLTK Resource Setup**: NLTK's `punkt` sentence tokenizer and `averaged_perceptron_tagger` POS tagger are downloaded into `data/nltk_data/` so they are available offline on compute nodes.

2. **Word Tokenization + POS Tagging** via `pos_tag_fn`:
   ```python
   text = "The cat sat on the mat."
   words = nltk.word_tokenize(text)
   # → ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
   pos_tuples = nltk.pos_tag(words)
   # → [('The','DT'), ('cat','NN'), ('sat','VBD'), ('on','IN'), ('the','DT'), ('mat','NN'), ('.', '.')]
   pos_tags = ['DT', 'NN', 'VBD', 'IN', 'DT', 'NN', '.']
   ```
   This runs in parallel with `num_proc=4`.

3. **POS Vocabulary Building**: All unique Penn Treebank POS tags found in the training set are collected, sorted alphabetically, and mapped to integer IDs:
   ```json
   // data/pos_to_id.json (example subset)
   { "''": 0, ",": 1, ".": 2, "CC": 3, "CD": 4, "DT": 5, "EX": 6,
     "FW": 7, "IN": 8, "JJ": 9, "NN": 10, "NNS": 11, "NNP": 12, ... }
   ```

4. **Label Mapping**: Each POS tag string is replaced by its integer ID:
   ```python
   pos_labels = [5, 10, 30, 8, 5, 10, 2]  # for the example above
   ```

5. **Train/Val Split**: A sequential 95%/5% split is made (first 95% for train, last 5% for validation). This preserves discourse continuity — consecutive sentences in train are actually consecutive in the corpus, which is important for NSP.

6. **Downstream Dataset Caching**: WikiText-2, GLUE (MRPC, CoLA, SST-2), BLiMP subsets, and Multi30k are downloaded and cached to `data/eval_data/` for offline use.

> [!NOTE]
> The preprocessed datasets store lists of **word-level** tokens and their POS label IDs, not subword tokens. The subword tokenization happens on-the-fly in the collator.

---

## 3. Stage 2 — Dataset & Collation (`src/dataset.py`)

**File:** [`src/dataset.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py)

The [`PretrainDataset`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L5-L57) and [`MultiTaskCollator`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L59-L182) together handle the full transformation from raw word lists to batched tensors.

### 3.1 NSP Pair Construction

**Where:** [`PretrainDataset.__getitem__`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L17-L57)

When `nsp_active=True` (i.e. when training with any NSP-containing task set), each call to `__getitem__` dynamically creates a sentence pair:

- **50% of the time** (`is_next=True`): sentence B is the **immediately following** sentence in the corpus (index `idx + 1`). This represents a genuine consecutive pair. NSP label = **0** (IsNext).
- **50% of the time** (`is_next=False`): sentence B is a **random sentence** from anywhere else in the corpus. NSP label = **1** (NotNext).

> [!IMPORTANT]
> The NSP label convention follows the original BERT paper: **0 = IsNext, 1 = NotNext**. This is the opposite of what you might expect intuitively. The loss is still cross-entropy over 2 classes.

**Example:**
```
idx=42:  words_a = ['The', 'dog', 'barked']     pos_a = [5, 10, 30]
idx=43:  words_b = ['It', 'was', 'very', 'loud'] pos_b = [15, 27, 18, 20]
next_label = 0   ← IsNext (true consecutive pair)

idx=42:  words_a = ['The', 'dog', 'barked']
idx=7800 words_b = ['Scientists', 'discovered']  ← random sentence
next_label = 1   ← NotNext
```

When `nsp_active=False`, `words_b = None`, `pos_b = None`, and a dummy label `next_label = 0` is returned (it will be ignored at training time since NSP loss won't be computed).

---

### 3.2 Tokenization & Encoding

**Where:** [`MultiTaskCollator.__call__`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L69-L182)

The `BertTokenizerFast` from `bert-base-uncased` is applied on the whole batch at once using `is_split_into_words=True`. This means words are already pre-tokenized at the word level, and the tokenizer handles subword splitting:

```
Input words:  ['The', 'dog', 'barked', 'at', 'strangers']
Subword tokens: [CLS] the dog bark ##ed at strangers [SEP]
Token IDs:     [ 101, 1996, 3899, 6260, 2098, 2012, 8768, 102 ]
```

For NSP pairs (sentence A + sentence B), the tokenizer merges them with two `[SEP]` tokens and sets `token_type_ids`:

```
[CLS] the  dog  bark ##ed  [SEP] it  was  loud [SEP] [PAD] [PAD]
  0    0    0    0    0     0    1   1    1    1    0    0
  ↑ segment 0 (sentence A)       ↑ segment 1 (sentence B)
```

Key encoding settings:
- `max_length=128` — sequences are truncated to 128 subword tokens
- `padding=True` — all examples in a batch padded to the same length
- `return_tensors="pt"` — returns PyTorch tensors

---

### 3.3 MLM Masking Strategy

**Where:** [`MultiTaskCollator.__call__`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L101-L128), lines 102–128

This is the core BERT masking strategy. It proceeds in four steps:

#### Step 1 — Clone labels
```python
labels = input_ids.clone()   # shape: [batch_size, seq_len]
```
`labels` starts as a copy of the token IDs. After masking, only the masked positions retain a valid label; all others are set to `-100` (PyTorch's "ignore" sentinel for cross-entropy).

#### Step 2 — Exclude special tokens from masking candidates
```python
probability_matrix = torch.full(labels.shape, 0.15)
special_tokens_mask = tokenizer.get_special_tokens_mask(...)  # marks [CLS],[SEP],[PAD]
probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
```
Positions corresponding to `[CLS]` (101), `[SEP]` (102), `[PAD]` (0), and `[MASK]` (103) tokens have their probability set to 0 — they will **never** be selected for masking.

#### Step 3 — Sample masked positions
```python
masked_indices = torch.bernoulli(probability_matrix).bool()
labels[~masked_indices] = -100   # Only masked positions contribute to loss
```
~15% of the **non-special** tokens are selected. All unselected positions in `labels` are set to `-100`.

#### Step 4 — Apply the 80/10/10 replacement rule
Of the ~15% selected positions:
- **80%** → replaced with `[MASK]` token (ID 103)
- **10%** → replaced with a **random** vocabulary token
- **10%** → kept **unchanged** (original token is the input; model must still predict it)

```python
# 80%: replace with [MASK]
indices_replaced = torch.bernoulli(torch.full(shape, 0.8)).bool() & masked_indices
input_ids[indices_replaced] = tokenizer.mask_token_id   # 103

# 10%: replace with random token  (note: 0.5 of remaining 20% = 10% overall)
indices_random = torch.bernoulli(torch.full(shape, 0.5)).bool() & masked_indices & ~indices_replaced
random_words = torch.randint(len(tokenizer), shape)
input_ids[indices_random] = random_words[indices_random]

# Remaining 10%: no change to input_ids, label still set
```

**Concrete example (single sequence):**

```
Position:   0    1    2    3    4    5    6    7
Token:     [CLS] the  dog  bark ##ed  at   str [SEP]
Selected:   NO   NO   YES  NO   YES  YES  NO   NO

Applied:
  pos 2 (dog)   → 80% → [MASK]       input=103, label=3899
  pos 4 (##ed)  → 10% → random tok   input=8421, label=2098
  pos 5 (at)    → 10% → unchanged    input=2012, label=2012

Final input_ids: [101, 1996, 103,  6260, 8421, 2012, 8768, 102]
Final labels:    [-100,-100, 3899, -100, 2098, 2012, -100, -100]
```

> [!NOTE]
> The 80/10/10 ratio exists to prevent the model from learning to only predict when it sees `[MASK]`. During downstream tasks (fine-tuning), `[MASK]` tokens are absent, so the model must generalize its internal representations beyond the mask signal.

---

### 3.4 Word-ID Alignment for POS

**Where:** [`MultiTaskCollator.__call__`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/dataset.py#L130-L170), lines 130–170

This is the most intricate part of the collator. POS labels are at the **word level**, but BERT operates on **subword tokens**. The mapping must be reconstructed explicitly for each example in the batch.

The HuggingFace tokenizer provides `encoding.word_ids(b)` which returns, for each subword token position, the index of the original word it came from (or `None` for special tokens).

**Example (single sentence, no NSP):**
```
Words:   ['barked', 'loudly']
Subword: [CLS] bark  ##ed  loud  ##ly  [SEP]
word_ids: None   0     0     1     1    None

→ aligned_word_ids:   [-1, 0, 0, 1, 1, -1]
→ aligned_pos_labels: [-100, POS[0], POS[0], POS[1], POS[1], -100]
```

**Example (NSP pair: sentence A + sentence B):**
```
words_a = ['The', 'dog']      len_a = 2
words_b = ['It', 'ran', 'fast']

Subword tokens: [CLS] the dog [SEP] it ran fast [SEP]
token_type_ids:   0    0   0    0    1   1    1    1
word_ids:        None  0   1   None  0   1    2   None

Processing:
  token 0: None          → word_id=-1,      pos_label=-100
  token 1: type=0, id=0  → word_id=0,       pos_label=pos_a[0]
  token 2: type=0, id=1  → word_id=1,       pos_label=pos_a[1]
  token 3: None          → word_id=-1,      pos_label=-100  ([SEP])
  token 4: type=1, id=0  → word_id=0+2=2,   pos_label=pos_b[0]  ← OFFSET by len_a=2
  token 5: type=1, id=1  → word_id=1+2=3,   pos_label=pos_b[1]
  token 6: type=1, id=2  → word_id=2+2=4,   pos_label=pos_b[2]
  token 7: None          → word_id=-1,      pos_label=-100  ([SEP])

→ aligned_word_ids:   [-1, 0, 1, -1, 2, 3, 4, -1]
→ aligned_pos_labels: [-100, PA0, PA1, -100, PB0, PB1, PB2, -100]
```

> [!IMPORTANT]
> The word ID offset (`w_id + len_a` for segment B tokens) is critical. Without it, word 0 of sentence B and word 0 of sentence A would both map to global word index 0, causing incorrect pooling in the POS head. The offset makes each word unique across the full concatenated sequence.

---

## 4. Stage 3 — Model Architecture (`src/models.py`)

**File:** [`src/models.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py)

### 4.1 The Base BERT Encoder

[`MultiTaskBERT`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L22-L171) uses a **BERT-Mini scale** configuration:

| Parameter | Value |
|---|---|
| `hidden_size` | 256 |
| `num_hidden_layers` | 6 |
| `num_attention_heads` | 8 |
| `intermediate_size` | 1024 |
| `max_position_embeddings` | 512 |
| `vocab_size` | 30,522 (bert-base-uncased) |

The BERT encoder takes `(input_ids, attention_mask, token_type_ids)` and produces:
- `sequence_output`: shape `[B, L, 256]` — hidden state for every token
- `pooler_output`: shape `[B, 256]` — `[CLS]` hidden state passed through a tanh-activated dense layer, used for classification

### 4.2 MLM Head

**Class:** [`BERTMLMHead`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L6-L20)

```python
nn.Sequential(
    nn.Linear(256, 256),   # dense transform
    nn.GELU(),             # smooth activation
    nn.LayerNorm(256),     # stabilize activations
    nn.Linear(256, 30522) # project to vocab size
)
```

Input: `sequence_output` → shape `[B, L, 256]`  
Output: `mlm_logits` → shape `[B, L, 30522]`

**Weight tying:** The final projection matrix (`nn.Linear(256, 30522)`) is **tied** to the word embedding matrix:
```python
self.mlm_head.predictions[3].weight = self.bert.embeddings.word_embeddings.weight
```
This means the same 30522×256 matrix is used both to embed input tokens and to project hidden states into vocabulary logits. Weight tying reduces parameters and improves training stability because the embedding space and output space share the same geometry.

### 4.3 NSP Head

```python
self.nsp_head = nn.Linear(256, 2)
```

Input: `pooler_output` → shape `[B, 256]`  
Output: `nsp_logits` → shape `[B, 2]` (logits for classes 0=IsNext, 1=NotNext)

The `[CLS]` token's pooled representation summarizes the entire sequence pair, making it the natural choice for the sentence-level relationship prediction task.

### 4.4 POS Head with Subword Pooling

**Lines:** [94–139](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L94-L139)

This is the most mathematically interesting part. POS tags are **word-level**, but BERT gives **subword-level** hidden states. The solution: **vectorized average pooling** using a matrix multiplication.

#### Step 1 — Build the Pooling Matrix P

Given `word_ids` tensor of shape `[B, L]` (with -1 for special tokens), we construct the boolean membership matrix:

```python
word_indices = torch.arange(max_words).view(1, max_words, 1)
mask = (word_ids.unsqueeze(1) == word_indices).float()
# shape: [B, max_words, L]
# mask[b, w, t] = 1 if token t belongs to word w in example b
```

**Example (batch size = 1, 3 words, sequence length 6):**
```
word_ids:  [-1,  0,  0,  1,  2,  -1]

mask (word 0): [0, 1, 1, 0, 0, 0]   ← bark, ##ed
mask (word 1): [0, 0, 0, 1, 0, 0]   ← at
mask (word 2): [0, 0, 0, 0, 1, 0]   ← strangers
```

#### Step 2 — Normalize to Averaging Weights

```python
subword_counts = mask.sum(dim=-1, keepdim=True)  # [B, max_words, 1]
subword_counts = torch.clamp(subword_counts, min=1.0)
P = mask / subword_counts   # [B, max_words, L]
```

`P[b, w, :]` is now a distribution over tokens: positions belonging to word `w` have equal weight summing to 1, all others are 0.

```
P (row 0, word "barked" = bark+##ed):  [0, 0.5, 0.5, 0, 0, 0]
P (row 1, word "at"):                  [0, 0,   0,   1, 0, 0]
P (row 2, word "strangers"):           [0, 0,   0,   0, 1, 0]
```

#### Step 3 — Pool Subword States into Word Representations

```python
pooled_word_reprs = torch.bmm(P, sequence_output)
# [B, max_words, L] × [B, L, 256] → [B, max_words, 256]
```

Each row of `pooled_word_reprs` is now the **arithmetic mean** of the hidden states of all subwords that make up that word. This is fully vectorized and runs in a single GPU kernel.

#### Step 4 — Align POS Labels to Words

Since `pos_labels` is aligned at the **token level** (one label per subword position), we need to pick the label from the **first subword** of each word:

```python
index_tensor = (word_ids.unsqueeze(1) == word_indices)       # [B, max_words, L]
first_indices = index_tensor.double().argmax(dim=-1)          # [B, max_words]
aligned_pos_labels = torch.gather(pos_labels, dim=1, index=first_indices)
```

`argmax` over the boolean tensor finds the **smallest token index** where the word membership is True (since `argmax` returns the first True position when there are ties). Words with no subword representation (padding words) get `aligned_pos_labels` filled with -100:
```python
word_present = index_tensor.any(dim=-1)
aligned_pos_labels = aligned_pos_labels.masked_fill(~word_present, -100)
```

#### Step 5 — Classify

```python
pos_logits = self.pos_classifier(pooled_word_reprs)  # [B, max_words, num_pos_tags]
```

This linear layer maps each word's 256-dimensional representation to a distribution over all POS tag classes.

---

## 5. Stage 4 — Loss Calculation per Objective

### 5.1 MLM Loss

**Where:** [`models.py` lines 82–83](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L82-L83)

```python
mlm_loss = F.cross_entropy(
    mlm_logits.view(-1, vocab_size),  # [B*L, 30522]
    labels.view(-1)                    # [B*L]
)
```

PyTorch's `cross_entropy` automatically **ignores** positions where `labels == -100`. So only the ~15% of masked tokens contribute to the gradient.

**What is predicted:** The original token ID at each masked position.

**Formula:**
$$L_{\text{MLM}} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log P_\theta(x_t \mid x_{\setminus t}^{\text{masked}})$$

where $\mathcal{M}$ is the set of masked token positions.

**Concrete example:**

```
labels:    [-100, -100, 3899, -100, 2098, 2012, -100, -100]
logits at position 2: [... 0.12 (for token 3899="dog"), ...]
logits at position 4: [... 0.08 (for token 2098="##ed"), ...]
logits at position 5: [... 0.31 (for token 2012="at"), ...]

loss = -(log(softmax(logits[2])[3899]) + log(softmax(logits[4])[2098]) + log(softmax(logits[5])[2012])) / 3
```

### 5.2 NSP Loss

**Where:** [`models.py` lines 90–91](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L90-L91)

```python
nsp_loss = F.cross_entropy(
    nsp_logits.view(-1, 2),          # [B, 2]
    next_sentence_label.view(-1)      # [B]
)
```

**What is predicted:** Whether sentence B follows sentence A (binary classification).

**Formula:**
$$L_{\text{NSP}} = -\frac{1}{B} \sum_{b=1}^{B} \log P_\theta(y_b \mid \text{[CLS]}_b)$$

where $y_b \in \{0=\text{IsNext}, 1=\text{NotNext}\}$ is the binary label.

**Example:**
```
Batch size = 2:
  Sample 0: IsNext pair   → label = 0
  Sample 1: NotNext pair  → label = 1

nsp_logits:
  sample 0: [2.1, 0.3]   → softmax=[0.87, 0.13] → loss = -log(0.87)
  sample 1: [-0.5, 1.8]  → softmax=[0.08, 0.92] → loss = -log(0.92)

nsp_loss = (-log(0.87) + -log(0.92)) / 2 ≈ 0.21
```

### 5.3 POS Loss

**Where:** [`models.py` lines 128–132](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L128-L132)

```python
pos_loss = F.cross_entropy(
    pos_logits.view(-1, num_pos_tags),   # [B*max_words, num_tags]
    aligned_pos_labels.reshape(-1),       # [B*max_words]
    ignore_index=-100
)
```

**What is predicted:** The Penn Treebank POS tag (e.g. `NN`, `VBD`, `DT`) for each **word** in the input, based on the averaged subword representations.

The `ignore_index=-100` causes PyTorch to skip:
- Special token positions (`[CLS]`, `[SEP]`, `[PAD]`)
- Padding words (words that don't exist in a shorter sequence)

**Formula:**
$$L_{\text{POS}} = -\frac{1}{|W|} \sum_{w \in W} \log P_\theta(\text{tag}_w \mid \bar{h}_w)$$

where $\bar{h}_w = \frac{1}{|S_w|} \sum_{t \in S_w} h_t$ is the mean hidden state of the subwords in word $w$.

**Concrete example:**
```
Sentence: "The quick brown fox"
Words: ['The', 'quick', 'brown', 'fox']
POS:   ['DT',  'JJ',    'JJ',    'NN']   → IDs: [5, 9, 9, 10]

Subwords: [CLS] the quick brown fox [SEP]
word_ids: [-1, 0, 1, 2, 3, -1]

pooled_word_reprs: [h_The, h_quick, h_brown, h_fox]  (each 256-dim)

pos_logits: [[...for The...], [...for quick...], [...for brown...], [...for fox...]]
            shape: [4, num_tags]

aligned_pos_labels: [5, 9, 9, 10]  (for the 4 words)

pos_loss = cross_entropy([h_The→logits, ...], [5, 9, 9, 10])
```

### 5.4 Combined Loss Aggregation

**Where:** [`models.py` lines 141–165](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L141-L165)

The loss aggregation logic depends on which task combination is active:

| Model Variant | Tasks | Loss Formula |
|---|---|---|
| `BASEBERT` | `mlm, nsp` | $L_{MLM} + L_{NSP}$ |
| `MLM_ONLY` | `mlm` | $L_{MLM}$ |
| `POSBERT` | `pos` | $\alpha \cdot L_{POS}$ |
| `MLMPOSBert` | `mlm, pos` | $L_{MLM} + \alpha \cdot L_{POS}$ |
| `NSPPOSBERT` | `nsp, pos` | $L_{NSP} + \alpha \cdot L_{POS}$ |
| `MLM_POS_NSPBERT` | `mlm, nsp, pos` | $L_{MLM} + L_{NSP} + \alpha \cdot L_{POS}$ |
| `MLMPOS_alpha02` | `mlm, pos` | $L_{MLM} + 0.2 \cdot L_{POS}$ |
| `MLMPOSNSP_alpha02` | `mlm, nsp, pos` | $L_{MLM} + L_{NSP} + 0.2 \cdot L_{POS}$ |

The `alpha` parameter (default 1.0, or 0.2 for the `_alpha02` variants) **scales the POS loss contribution**. A lower alpha prevents POS from dominating MLM, which is the primary signal for language understanding.

The code uses explicit `set(tasks) ==` comparisons to distinguish cases:
```python
if set(tasks) == {'mlm', 'nsp', 'pos'}:
    total_loss = mlm_val + nsp_val + alpha * pos_val
elif set(tasks) == {'mlm', 'pos'}:
    total_loss = mlm_val + alpha * pos_val
elif 'pos' in tasks:
    # Catches nsp+pos and pos-only
    total_loss = sum(non_pos_losses) + alpha * pos_val
else:
    total_loss = sum(active_losses)  # Pure MLM or pure NSP
```

---

## 6. Stage 5 — Training Loop (`train.py`)

**File:** [`train.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/train.py)

### Optimizer & Scheduler

**AdamW** with weight decay is used with differential parameter grouping:
- Parameters with names containing `"bias"` or `"LayerNorm.weight"`: `weight_decay=0.0`
- All other parameters: `weight_decay=0.01`

This follows BERT's original training recipe — LayerNorm parameters and biases should not be penalized.

**Linear LR schedule** with warmup:
```python
num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
```
Learning rate linearly ramps up from 0 to `lr` during warmup, then linearly decays back to 0.

### Training Step

For each batch:
1. `batch_inputs` are moved to GPU
2. Forward pass: `model(**batch_inputs, tasks=tasks, alpha=alpha)` → `outputs`
3. `loss = outputs['loss']` (the combined loss)
4. `optimizer.zero_grad()` → `loss.backward()` → gradient clip at norm 1.0 → `optimizer.step()` → `lr_scheduler.step()`

The progress bar shows the total loss plus individual component losses (MLM, NSP, POS) in real time.

### Checkpointing

Two checkpoints are saved:
- `checkpoint_latest.pt`: saved every epoch (for resume capability)
- `checkpoint_best.pt`: saved only when validation loss improves

Each checkpoint stores: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_loss`, `train_loss`, `val_loss`.

---

## 7. Stage 6 — Evaluation (`eval.py`)

**File:** [`eval.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/eval.py)

### 6.1 Pseudo-Perplexity (PPL)

**Function:** [`evaluate_perplexity`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/eval.py#L24-L83)

Standard perplexity requires an autoregressive (left-to-right) model. BERT is bidirectional, so a **Pseudo-Log-Likelihood (PLL)** approximation is used instead.

**Method:** For each sentence of length $n$ non-special tokens, we create $n$ masked copies of the sentence — one per token — and compute the model's probability of predicting the correct token at that masked position:

$$\text{PLL}(S) = \sum_{t=1}^{n} \log P_\theta(x_t \mid x_{\setminus t})$$

$$\text{PPL}(S) = \exp\left(-\frac{1}{n} \text{PLL}(S)\right) = \exp\left(-\frac{1}{n}\sum_{t=1}^{n} \log P_\theta(x_t \mid x_{\setminus t})\right)$$

**Step-by-step for a 4-token sentence `"the dog barked"`:**

```
Original: [CLS] the(1) dog(2) bark(3) ##ed(4) [SEP]
Non-special tokens: positions 1, 2, 3, 4

Create 4 masked copies:
  Copy 0: [CLS] [MASK] dog  bark ##ed [SEP]  → predict token at pos 1
  Copy 1: [CLS] the  [MASK] bark ##ed [SEP]  → predict token at pos 2
  Copy 2: [CLS] the  dog  [MASK] ##ed [SEP]  → predict token at pos 3
  Copy 3: [CLS] the  dog  bark  [MASK][SEP]  → predict token at pos 4

All 4 copies run through the model in a batch.
From each copy, read the logit at the masked position:
  loss_0 = -log P("the"  | "_ dog barked")
  loss_1 = -log P("dog"  | "the _ barked")
  loss_2 = -log P("bark" | "the dog _ ##ed")
  loss_3 = -log P("##ed" | "the dog bark _")

avg_loss = (loss_0 + loss_1 + loss_2 + loss_3) / 4
PPL = exp(avg_loss)
```

This is aggregated over up to 500 sentences from WikiText-2 (held-out) and 500 from the BabyLM validation split.

> [!NOTE]
> PLL is quadratic in sequence length (each token requires one forward pass), so computation is expensive. A subset of 500 examples is used for practical speed.

### 6.2 BLiMP Zero-Shot Evaluation

**Function:** [`evaluate_blimp`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/eval.py#L128-L153)

BLiMP (Benchmark of Linguistic Minimal Pairs) consists of pairs of grammatically correct vs. incorrect sentences that differ in exactly one word. A language model should assign higher probability to the grammatical sentence.

**Method:** For each pair `(sentence_good, sentence_bad)`, compute PLL for each, then compare:

```
sentence_good: "The dogs bark at strangers"
sentence_bad:  "The dogs barks at strangers"   ← wrong subject-verb agreement

PLL(good) = log P("The") + log P("dogs") + log P("bark") + ...
PLL(bad)  = log P("The") + log P("dogs") + log P("barks") + ...

If PLL(good) > PLL(bad): correct → accuracy += 1
```

The final score is `(correct / total) * 100` as a percentage. Two BLiMP subsets are evaluated:
- `regular_plural_subject_verb_agreement_1` — subject-verb agreement
- `anaphor_gender_agreement` — pronoun gender agreement

### 6.3 GLUE Downstream Fine-Tuning

**Function:** [`fine_tune_glue`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/eval.py#L158-L230)

The pretrained BERT encoder is loaded into a `BertForSequenceClassification` head, which adds a dropout + linear layer on top of the `[CLS]` pooler output.

**Weight loading:** Only the BERT encoder weights are transferred:
```python
cleaned_state_dict = {k[5:]: v for k, v in encoder_state_dict.items() if k.startswith("bert.")}
classifier_model.bert.load_state_dict(cleaned_state_dict, strict=False)
```
The `k[5:]` strips the leading `"bert."` prefix from our `MultiTaskBERT`'s state dict keys to match HuggingFace's `BertForSequenceClassification.bert` namespace.

**Three GLUE tasks:**
| Task | Input | Label | Metric |
|---|---|---|---|
| **MRPC** | Two sentences (paraphrase pairs) | 0=not paraphrase, 1=paraphrase | Accuracy + F1 |
| **CoLA** | Single sentence | 0=ungrammatical, 1=grammatical | Matthews Correlation Coefficient (MCC) |
| **SST-2** | Single sentence (movie review) | 0=negative, 1=positive | Accuracy |

Each task is fine-tuned for 3 epochs with AdamW at lr=2e-5. SST-2 training is subsampled to 10,000 examples for speed.

### 6.4 BLEU Machine Translation

**Function:** [`fine_tune_bleu`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/eval.py#L235-L336)

The BERT encoder is used as the encoder half of a `BertEncoderDecoder` seq2seq model. A fresh BERT decoder (with cross-attention enabled: `add_cross_attention=True`, `is_decoder=True`) is initialized from scratch and trained jointly.

```
[English sentence] → [Pretrained BERT Encoder] → encoder hidden states
                                                       ↓ cross-attention
                                              [Fresh BERT Decoder] → [German tokens]
```

Training follows a teacher-forcing regime (standard seq2seq), and evaluation uses beam search (num_beams=2) to generate German translations on the Multi30k test set. BLEU is computed using NLTK's `corpus_bleu`.

---

## 8. Model Variants & Their Differences

The repository includes both standard **MultiTaskBERT** and **LayeredPOSMLMBert** configurations:

| Variant | Architecture | MLM | NSP | POS | Layering Mode | Notes |
|---|---|:---:|:---:|:---:|---|---|
| `BASEBERT` | MultiTaskBERT | ✓ | ✓ | — | — | Standard BERT pretraining |
| `MLM_ONLY` | MultiTaskBERT | ✓ | — | — | — | MLM without sentence-level objective |
| `POSBERT` | MultiTaskBERT | — | — | ✓ | — | POS-only, a syntax-focused baseline |
| `MLMPOSBert` | MultiTaskBERT | ✓ | — | ✓ | — | MLM + syntactic supervision |
| `NSPPOSBERT` | MultiTaskBERT | — | ✓ | ✓ | — | Sentence + syntactic objective |
| `MLM_POS_NSPBERT` | MultiTaskBERT | ✓ | ✓ | ✓ | — | All three objectives |
| `MLMPOS_alpha02` | MultiTaskBERT | ✓ | — | ✓ | — | MLM + POS with α=0.2 (reduced POS) |
| `MLMPOSNSP_alpha02` | MultiTaskBERT | ✓ | ✓ | ✓ | — | All three with α=0.2 (reduced POS) |
| `LAYERED_POS_MLM_SOFT` | LayeredPOSMLMBert | ✓ | — | ✓ | **Soft** | Differentiable probability mask overlay |
| `LAYERED_POS_MLM_HARD` | LayeredPOSMLMBert | ✓ | — | ✓ | **Hard** | Argmax prediction mask overlay |
| `LAYERED_POS_MLM_GOLD` | LayeredPOSMLMBert | ✓ | — | ✓ | **Gold** | Ground-truth tag mask overlay (train only) |

> [!TIP]
> The `alpha=0.2` variants test whether a **weaker** syntactic signal helps more than a full-strength one. The `LAYERED` models go a step further, restricting the search space of the vocabulary based on syntactic predictions.

---

## 9. End-to-End Example: One Training Step

This traces the full pipeline for a batch of 2 examples with tasks `['mlm', 'pos']` (MLMPOSBert):

**Input (from dataset):**
```
Example 0:  words_a = ['A', 'large', 'dog', 'barked']
            pos_a   = [DT=5, JJ=9, NN=10, VBD=30]
            words_b = None  (no NSP)
            next_label = 0

Example 1:  words_a = ['She', 'quickly', 'ran']
            pos_a   = [PRP=22, RB=23, VBD=30]
            words_b = None
            next_label = 0
```

**Collator output:**

Tokenization:
```
Ex 0: [CLS] a large dog bark ##ed [SEP] [PAD]   (len=8)
Ex 1: [CLS] she quickly ran        [SEP] [PAD] [PAD] (padded to len=8)

input_ids (before masking):
  [[101, 1037, 2312, 3899, 6260, 2098, 102, 0],
   [101, 2016, 4415, 2743,  102,    0,   0, 0]]
```

MLM masking (hypothetical):
```
Ex 0: pos 2 (large) masked → [MASK]
      labels: [-100, -100, 2312, -100, -100, -100, -100, -100]

Ex 1: pos 3 (ran) masked → random token (e.g. 8000)
      labels: [-100, -100, -100, 2743, -100, -100, -100, -100]
```

Word ID alignment:
```
Ex 0: word_ids = [-1, 0, 1, 2, 2, 3, -1, -1]
      (note: "bark" and "##ed" both map to word 2)
      pos_labels = [-100, 5, 9, 10, 10, 30, -100, -100]

Ex 1: word_ids = [-1, 0, 1, 2, -1, -1, -1, -1]
      pos_labels = [-100, 22, 23, 30, -100, -100, -100, -100]
```

**Forward pass through `MultiTaskBERT`:**

1. BERT encoder processes masked `input_ids`:
   ```
   sequence_output: shape [2, 8, 256]
   pooler_output:   shape [2, 256]   (unused — no NSP)
   ```

2. **MLM head:**
   ```
   mlm_logits = mlm_head(sequence_output)  → [2, 8, 30522]
   
   # Ex 0: logits at pos 2 → should predict "large" (2312)
   # Ex 1: logits at pos 3 → should predict "ran" (2743)
   
   mlm_loss = cross_entropy(
       mlm_logits.view(-1, 30522),
       labels.view(-1)           # all -100 except pos 2 (Ex0) and pos 3 (Ex1)
   )
   ```

3. **POS head (subword pooling):**
   ```
   max_words = 4  (largest word index in batch + 1)
   
   Build P:
     Ex 0:
       P[0, word0, :] = [0, 1, 0, 0, 0, 0, 0, 0]         # "a"
       P[0, word1, :] = [0, 0, 1, 0, 0, 0, 0, 0]         # "large"
       P[0, word2, :] = [0, 0, 0, 0.5, 0.5, 0, 0, 0]     # "dog" = bark+##ed (averaged)
       P[0, word3, :] = [0, 0, 0, 0, 0, 1, 0, 0]         # "barked"
     
   pooled_word_reprs = P @ sequence_output → [2, 4, 256]
   
   pos_logits = pos_classifier(pooled_word_reprs) → [2, 4, num_tags]
   
   aligned_pos_labels:
     Ex 0: [5, 9, 10, 30]  → DT, JJ, NN, VBD
     Ex 1: [22, 23, 30, -100]  → PRP, RB, VBD, (no word 3 → ignore)
   
   pos_loss = cross_entropy(
       pos_logits.view(-1, num_tags),
       aligned_pos_labels.reshape(-1)   # -100 positions ignored
   )
   ```

4. **Total loss (α=1.0):**
   ```
   total_loss = mlm_loss + 1.0 * pos_loss
   ```

5. **Backward pass:**
   ```
   total_loss.backward()
   # Gradients flow through pos_classifier, bert encoder,
   # and mlm_head (including shared word embeddings)
   
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   optimizer.step()
   lr_scheduler.step()
   ```

---

> [!NOTE]
> The `POSBERT` model (POS-only) still goes through tokenization and collation identically — it just has `mlm_active=False` in the collator (so `labels.fill_(-100)` and no masking is applied) and the NSP is not used. The POS head still receives the full unmasked sequence through BERT and learns syntactic structure without any MLM signal.

---

## 10. Layered POS-MLM Vocabulary Reduction Models

**File:** [`src/models.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py) (class [`LayeredPOSMLMBert`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/src/models.py#L173-L323))

Standard multi-task models predict semantic tokens and syntactic parts-of-speech via independent output heads. In contrast, the **Layered** architecture models the conditional dependency of semantics on syntax:

```
[Input IDs] → [Base BERT Encoder] → [Token Representation H]
                                           ↓
                                  [POS Classifier]
                                           ↓ (POS Probabilities)
                                   [Vocabulary Mask]
                                           ↓
[Input IDs] → [Base BERT Encoder] → [MLM Classifier] → [Logits Combination]
```

### 10.1 Mathematical Masking Modes
The model restricts the MLM prediction head's search space at the final softmax projection layer. This is achieved by computing a vocabulary mask $M_{mask} \in \mathbb{R}^{B \times L \times V}$ and adding its logarithm to the raw MLM logits:

$$\text{MaskedLogits} = \text{RawLogits} + \log(M_{mask} + \epsilon)$$

The vocabulary mask is computed in three different modes:

1. **Gold Masking (`gold`)**:
   - Uses the ground-truth part-of-speech labels (during training only):
     $$M_{mask} = \mathbf{pos\_to\_vocab\_mask}[y_{gold}]$$
   - Any vocabulary token that does not share the correct gold POS tag is zeroed out in the mask, reducing its logit to $-\infty$.

2. **Hard Masking (`hard`)**:
   - Uses the `argmax` predicted part-of-speech tag:
     $$\hat{y}_{pos} = \arg\max(\text{pos\_logits})$$
     $$M_{mask} = \mathbf{pos\_to\_vocab\_mask}[\hat{y}_{pos}]$$
   - This creates a hard constraint based on model confidence, but the argmax operation is non-differentiable.

3. **Soft Masking (`soft`)**:
   - Computes POS tag probabilities using softmax:
     $$P_{pos} = \text{softmax}(\text{pos\_logits}) \in \mathbb{R}^{B \times L \times C}$$
   - Performs a batched matrix multiplication with the static mapping buffer $\mathbf{M}_{map} \in \mathbb{R}^{C \times V}$ (where $C$ is the number of POS categories, and $V$ is the vocab size):
     $$M_{mask} = P_{pos} \times \mathbf{M}_{map}$$
   - Because the matrix multiplication and softmax operations are fully differentiable, gradients from the semantic MLM loss flow backward *through* the mask directly into the POS Classifier parameters, co-tuning both heads.

### 10.2 Example-Driven Backpropagation Flow (Soft Mode)
Consider the sentence: *"the [MASK] sat on the mat."* where the masked word at index 1 is `"cat"`.

1. **Forward Pass**:
   - POS classifier predicts probabilities over index 1:
     $$P(\text{Noun}) = 0.85, \quad P(\text{Verb}) = 0.10, \quad P(\text{Adjective}) = 0.05$$
   - The vocabulary mask at index 1 is a mixture:
     $$M_{mask} = 0.85 \cdot \mathbf{M}_{Noun} + 0.10 \cdot \mathbf{M}_{Verb} + 0.05 \cdot \mathbf{M}_{Adj}$$
   - For vocabulary token `"cat"` (which is mapped under Noun):
     $$M_{mask}[\text{"cat"}] \approx 0.85$$
   - For vocabulary token `"sat"` (which is mapped under Verb):
     $$M_{mask}[\text{"sat"}] \approx 0.10$$
   - Masked logits add $\log(M_{mask} + \epsilon)$ to raw logits, suppressing `"sat"` while preserving `"cat"`.

2. **Backward Pass**:
   - The ground-truth token is `"cat"`. The cross-entropy loss $\mathcal{L}_{MLM}$ decreases if the probability of predicting `"cat"` increases.
   - The probability of predicting `"cat"` depends directly on the mask value $M_{mask}[\text{"cat"}]$, which is scaled by $P(\text{Noun})$.
   - By the chain rule:
     $$\frac{\partial \mathcal{L}_{MLM}}{\partial y_{pos}(\text{Noun})} = \frac{\partial \mathcal{L}_{MLM}}{\partial \text{MaskedLogits}[\text{"cat"}]} \cdot \frac{\partial \text{MaskedLogits}[\text{"cat"}]}{\partial M_{mask}[\text{"cat"}]} \cdot \frac{\partial M_{mask}[\text{"cat"}]}{\partial P(\text{Noun})} \cdot \frac{\partial P(\text{Noun})}{\partial y_{pos}(\text{Noun})}$$
   - This produces a strong gradient pushing the POS classifier to output a higher probability for the `Noun` category, regularizing semantic learning with syntactic structure directly during pretraining.
