# BabyLM Multi-Task pretraining Evaluation Report

This report compares five BERT-Mini models trained on the `BabyLM-2026-Strict-Small` dataset (10M words) with different pretraining tasks.

## Objectives Configuration:
1. **BASEBERT**: MLM + NSP
2. **POSBERT**: POS only
3. **MLMPOSBert**: MLM + POS
4. **NSPPOSBERT**: NSP + POS
5. **MLM_POS_NSPBERT**: MLM + NSP + POS

## Comparative Results Table

| Model Variant | Loss Calculation | BabyLM PPL ⬇️ | WikiText-2 PPL ⬇️ | BLiMP SVA Acc ⬆️ | BLiMP PG Acc ⬆️ | GLUE MRPC Acc ⬆️ | GLUE MRPC F1 ⬆️ | GLUE CoLA MCC ⬆️ | GLUE SST-2 Acc ⬆️ | Multi30k BLEU ⬆️ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASEBERT | $L_{MLM} + L_{NSP}$ | 249.45 | 539.79 | 57.00% | 45.00% | 0.7010 | 0.8152 | 0.0174 | 0.7225 | 5.39 |
| POSBERT | $\alpha L_{POS}$ | 31502.16 | 30933.80 | 45.00% | 37.00% | 0.7010 | 0.8076 | 0.0473 | 0.7603 | 4.69 |
| MLMPOSBert | $L_{MLM} + \alpha L_{POS}$ | 274.23 | 647.22 | 62.00% | 45.00% | 0.6936 | 0.7974 | 0.0122 | 0.7282 | **5.92** |
| NSPPOSBERT | $L_{NSP} + \alpha L_{POS}$ | 31585.45 | 31943.74 | 51.00% | 23.00% | **0.7132** | **0.8157** | **0.1186** | 0.7259 | 4.60 |
| MLM_POS_NSPBERT | $L_{MLM} + L_{NSP} + \alpha L_{POS}$ | 209.88 | **478.46** | 62.00% | 48.00% | 0.6985 | 0.8081 | 0.0465 | **0.7695** | 5.46 |
| MLM_ONLY | $L_{MLM}$ | 297.42 | 645.11 | 56.00% | **54.00%** | 0.6789 | 0.7884 | 0.0752 | 0.7523 | 4.54 |
| MLMPOS_alpha02 | $L_{MLM} + 0.2 L_{POS}$ | 272.86 | 609.42 | 61.00% | 46.00% | 0.6961 | 0.8086 | 0.0046 | 0.7454 | 4.79 |
| MLMPOSNSP_alpha02 | $L_{MLM} + L_{NSP} + 0.2 L_{POS}$ | **208.95** | 494.90 | **63.00%** | 49.00% | 0.6863 | 0.7981 | 0.0497 | 0.7420 | 5.28 |

*Notes:*
- **PPL (Perplexity)**: Lower is better. Computed using pseudo-perplexity from masked tokens.
- **Accuracy / F1 / MCC / BLEU**: Higher is better.
- **BLiMP**: Measured zero-shot grammatical accuracy on minimal pairs.
- **GLUE tasks**: Fine-tuned for 3 epochs.
- **BLEU**: Fine-tuned an Encoder-Decoder model on Multi30k for 3 epochs.

## Visualizations

### 1. Pretraining Perplexity
The following plot shows the pseudo-perplexity values of all models (on log scale due to the extremely high perplexity of POS-only models). Below it, we plot the perplexities on normal scale for the language-modeling variants.
![Perplexity Comparison (All Models - Log Scale)](./babylm-eval/plots/perplexity_comparison.png)

![Perplexity Comparison (LMs Only - Normal Scale)](./babylm-eval/plots/perplexity_lm_only.png)

### 2. Zero-Shot Grammatical Diagnostics
A comparison of the macro-average accuracies across the primary zero-shot linguistic benchmarks (BLiMP Filtered, BLiMP Supplement, COMPS, and Entity Tracking).
![Zero-Shot Grammatical Diagnostics Comparison](./babylm-eval/plots/zeroshot_comparison.png)

### 3. Downstream GLUE Performance
The macro-average fine-tuning performance across downstream GLUE tasks.
![GLUE Downstream Average Performance](./babylm-eval/plots/glue_downstream_comparison.png)

### 4. Seq2Seq Translation Adaptation
The BLEU score achieved by fine-tuning an Encoder-Decoder model on the Multi30k translation dataset.
![Seq2Seq Translation BLEU Score](./babylm-eval/plots/translation_bleu_comparison.png)

## Detailed Metric Descriptions


### 1. Perplexity (PPL)
Perplexity measures how well a probability model predicts a sample. 
- **BabyLM PPL ⬇️**: Pseudo-Perplexity (PLL) computed on the validation split of the in-domain `BabyLM-2026-Strict-Small` corpus. Lower perplexity indicates that the model fits the training domain vocabulary, syntax, and structures better.
- **WikiText-2 PPL ⬇️**: Pseudo-Perplexity computed on the held-out `WikiText-2` validation set. This serves as an out-of-domain benchmark to measure general language modeling capabilities and prevent overfitting to the BabyLM corpus.

### 2. Zero-Shot Grammatical Diagnostics (BLiMP)
The Benchmark of Linguistic Minimal Pairs (BLiMP) evaluates a model's zero-shot grammatical knowledge by comparing the pseudo-log-likelihood (PLL) of a grammatically correct sentence against a matching incorrect minimal pair sentence.
- **BLiMP SVA Acc ⬆️**: Subject-Verb Agreement Accuracy. Evaluates whether the model assigns higher likelihood to sentences with correct subject-verb number agreement (e.g., *regular_plural_subject_verb_agreement_1*).
- **BLiMP PG Acc ⬆️**: Pronoun Gender / Anaphor Agreement Accuracy. Evaluates the model's ability to properly bind gendered pronouns to their antecedents (e.g., *anaphor_gender_agreement*).

### 3. Downstream Fine-Tuning Tasks (GLUE)
Downstream tasks fine-tune the pretrained BERT encoder on classification heads for 3 epochs:
- **GLUE MRPC Acc & F1 ⬆️**: Microsoft Research Paraphrase Corpus. Evaluates semantic equivalence recognition between sentence pairs. Reported in standard Accuracy and F1-score (harmonic mean of precision and recall).
- **GLUE CoLA MCC ⬆️**: Corpus of Linguistic Acceptability. Measures the grammatical acceptability judgments of sentences. Reported using the Matthews Correlation Coefficient (MCC), which accounts for class imbalance (ranging from -1 to +1).
- **GLUE SST-2 Acc ⬆️**: Stanford Sentiment Treebank. Evaluates binary sentiment classification (positive/negative) on movie reviews.

### 4. Sequence-to-Sequence Generative Tasks
- **Multi30k BLEU ⬆️**: Evaluated by training an Encoder-Decoder translation model using the pretrained BERT-Mini encoder and a matching decoder architecture on the Multi30k dataset (English to German). The BLEU score measures the n-gram overlap between the generated translation and the reference translation (higher is better).

