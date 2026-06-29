# tests Directory

This directory contains the pipeline verification and test suite modules. These scripts run integration tests to check data alignment, dimensionality sizes, and gradient calculations.

## 📂 Directory Contents

- **[`verify_pipeline.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/tests/verify_pipeline.py)**: Tests baseline MultiTaskBERT data collation, average pooling dimensions, and backpropagation gradients.
- **[`verify_layered_pipeline.py`](file:///dss/dsshome1/08/ge87ves2/desktop/BabyLM_Challenge/tests/verify_layered_pipeline.py)**: Runs tests verifying the layered vocab reduction configuration (gradient validations across gold, hard, and soft vocab masking).

---

## 🏃 Running the Tests

To run the verification suite:
```bash
python tests/verify_pipeline.py
python tests/verify_layered_pipeline.py
```
Both test files are self-contained and run on mock data splits, requiring no GPUs.
