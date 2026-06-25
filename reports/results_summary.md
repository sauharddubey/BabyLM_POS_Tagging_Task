# BabyLM Evaluation Results Summary

This file summarizes the evaluation results across all models for both **Zero-shot** and **Finetuning (GLUE)** tasks.

## Zero-Shot Results

| Task | BASEBERT | MLMPOSBert | MLMPOSNSP_alpha02 | MLMPOS_alpha02 | MLM_ONLY | MLM_POS_NSPBERT | NSPPOSBERT | POSBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BLiMP (filtered) | 51.97 | 48.32 | 51.67 | 50.62 | **53.99** | 51.95 | 52.66 | 47.75 |
| BLiMP Supplement (filtered) | 44.39 | **46.58** | 40.89 | 45.68 | 43.33 | 41.72 | 38.18 | 41.77 |
| COMPS | 49.53 | 49.68 | 49.87 | 49.75 | **50.64** | 50.23 | 50.51 | 49.93 |
| Entity Tracking | 40.74 | 40.81 | 41.71 | 41.37 | **42.14** | 41.34 | 41.53 | 40.76 |
| Reading (Eye Tracking) | 8.14 | 7.04 | **8.79** | 7.99 | 6.69 | 7.80 | 7.75 | 8.07 |
| Reading (Self-Paced) | 2.06 | **2.83** | 2.66 | 2.23 | 2.75 | 2.19 | 2.46 | 2.66 |

## Finetuning Results (GLUE)

| Task | BASEBERT | MLMPOSBert | MLMPOSNSP_alpha02 | MLMPOS_alpha02 | MLM_ONLY | MLM_POS_NSPBERT | NSPPOSBERT | POSBERT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLUE: boolq (accuracy) | 65.99 | 66.61 | **66.79** | 66.61 | 66.67 | **66.79** | 66.24 | 65.14 |
| GLUE: mnli (accuracy) | 43.66 | 43.05 | **45.21** | 42.69 | 43.46 | 44.62 | 41.42 | 41.01 |
| GLUE: mrpc (f1) | 81.57 | 82.01 | 81.99 | 82.99 | 81.05 | 82.01 | **83.23** | 82.54 |
| GLUE: multirc (accuracy) | 58.00 | 58.79 | 58.75 | 58.42 | 58.04 | **59.12** | 59.03 | 58.46 |
| GLUE: qqp (f1) | 60.37 | 60.04 | 60.39 | 60.44 | 59.90 | 60.62 | **60.92** | 60.00 |
| GLUE: rte (accuracy) | 61.15 | 58.99 | 57.55 | 58.27 | **62.59** | 57.55 | 53.24 | 56.83 |
| GLUE: wsc (accuracy) | 61.54 | **65.38** | 61.54 | 61.54 | 63.46 | 61.54 | 63.46 | **65.38** |

## Macro-Average Performance by Category

| Model | Zero-Shot Accuracy Avg | Zero-Shot Reading Avg | GLUE Avg |
| --- | --- | --- | --- |
| BASEBERT | 46.66 | 5.10 | 61.76 |
| MLMPOSBert | 46.35 | 4.94 | 62.12 |
| MLMPOSNSP_alpha02 | 46.04 | **5.72** | 61.75 |
| MLMPOS_alpha02 | 46.86 | 5.11 | 61.56 |
| MLM_ONLY | **47.52** | 4.72 | **62.17** |
| MLM_POS_NSPBERT | 46.31 | 5.00 | 61.75 |
| NSPPOSBERT | 45.72 | 5.11 | 61.08 |
| POSBERT | 45.05 | 5.37 | 61.34 |
