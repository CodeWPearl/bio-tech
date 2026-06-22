# Cancer Mutation Pathogenicity Predictor

Multi-omics deep learning framework for predicting the pathogenicity
(Pathogenic / Likely Pathogenic / Benign / Likely Benign) of cancer-associated
gene mutations by integrating mutation, gene-expression, DNA-methylation, and
copy-number-variation data.

## Setup

```bash
# 1. Create and activate a virtual environment (Python >=3.10)
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the project in editable mode (enables the train/evaluate/inference commands)
pip install -e .
```

## Quick start

```bash
train --config configs/default.yaml
evaluate --checkpoint results/checkpoints/best_model.ckpt
```

See `CLAUDE.md` for coding standards, architecture rules, and data-source details.
