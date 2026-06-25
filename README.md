# Cancer Mutation Pathogenicity Predictor

A research-grade deep learning framework for predicting the pathogenicity of
cancer-associated gene mutations (**Pathogenic** / **Likely Pathogenic** /
**Benign** / **Likely Benign**) by integrating multi-omics data — mutation
features, gene expression (RNA-seq), DNA methylation, and copy number variation
(CNV) — through cross-attention fusion.

**Target venue:** IEEE / Springer / Nature journal submission.

## Abstract

Accurate classification of cancer-associated mutations as pathogenic or benign is
critical for clinical decision-making and precision oncology. Current approaches
rely on single data modalities and lack uncertainty quantification. We present a
multi-omics deep learning framework that integrates mutation features, gene
expression, DNA methylation, and copy number variation through learned
cross-attention fusion. The model provides calibrated predictions with epistemic
uncertainty estimates, enabling automated flagging of low-confidence cases for
expert review. We evaluate against classical ML baselines (XGBoost, LightGBM,
Random Forest), demonstrate the contribution of each modality through ablation
studies, and validate predictions against the COSMIC Cancer Gene Census.

## Architecture

The model follows an **encode → fuse → classify** pipeline:

```
   Mutation features (42-dim)  ─→  MutationEncoder    ─→  128-dim embedding  ─┐
   Expression (2000-dim)       ─→  DenseAutoencoder   ─→  256-dim embedding  ─┤
   Methylation (2000-dim)      ─→  MethylationEncoder ─→  128-dim embedding  ─┼─→  Cross-Attention  ─→  Classifier  ─→  4 classes
   CNV (200-dim)               ─→  CNVFCEncoder       ─→   64-dim embedding  ─┤      Fusion (256-d)      Head
   Clinical (32-dim)           ─→  ClinicalMLP        ─→   32-dim embedding  ─┘
```

Five fusion strategies are available (configurable via `model.fusion_type`):
early concatenation, late fusion, attention, **cross-attention** (default),
and transformer. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design
rationale.

*Refer to `results/figures/fig01_model_architecture.pdf` (Figure 1) for the
publication-quality architecture diagram.*

## Installation

### Option A: pip (recommended for development)

```bash
# 1. Clone the repository
git clone https://github.com/your-org/cancer-mutation-pathogenicity.git
cd cancer-mutation-pathogenicity

# 2. Create and activate a virtual environment (Python >=3.10)
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib-venn

# 4. Install the project in editable mode
pip install -e .
```

### Option B: Docker (recommended for reproducibility)

```bash
# Build the image
docker build -t cancer-pathogenicity .

# Or using Make
make docker-build
```

### Option C: Make (one command)

```bash
make setup
```

## Data Download

All data sources are **free and publicly available** — no authentication required.

| Source | What it provides | URL |
|--------|-----------------|-----|
| **ClinVar** | Pathogenicity labels (VCF + TSV) | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/ |
| **cBioPortal** | Multi-omics TCGA data (mutations, expression, methylation, CNV) | https://www.cbioportal.org/api |
| **COSMIC CGC** | Known cancer driver gene list (validation) | https://cancer.sanger.ac.uk/census |

```bash
# Download and process all data (~10-20 minutes depending on network)
python scripts/download_data.py --config configs/default.yaml

# Or using Make
make download
```

This will:
1. Download ClinVar `variant_summary.txt.gz` (~50 MB)
2. Fetch multi-omics data from cBioPortal for 5 TCGA PanCancer Atlas studies
3. Merge labels with features, aligned by gene + genomic position
4. Split by **gene** (not variant) into train/val/test sets (70/15/15)
5. Save processed data to `data/processed/` and splits to `data/splits/`

## Training

```bash
# Train with default configuration (cross-attention fusion, focal loss)
python scripts/train.py --config configs/default.yaml

# Train with a specific experiment name
python scripts/train.py --config configs/default.yaml --experiment_name my_experiment

# Train with config overrides
python scripts/train.py --config configs/default.yaml \
    --override training.learning_rate=0.0005 model.fusion_type=transformer

# Train with GPU
python scripts/train.py --config configs/default.yaml --gpus 1

# Train with Docker
docker run --gpus all -v ./data:/app/data -v ./results:/app/results \
    cancer-pathogenicity scripts/train.py --config configs/default.yaml

# Or using Make
make train
```

**Expected output:**
- Training logs printed to console and saved to `results/logs/`
- Model checkpoints saved to `results/checkpoints/` (top-3 by val_auroc)
- Experiment tracked in MLflow (run `mlflow ui` to view at http://localhost:5000)
- Final test metrics saved to `results/tables/final_metrics.json`

**Expected training time:** ~2-4 hours on a single GPU (NVIDIA RTX 3080 or better),
~100 epochs with early stopping (patience=15).

## Evaluation

```bash
# Evaluate the best checkpoint on the test set
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.ckpt

# Skip baseline comparisons (faster)
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.ckpt --skip-baselines

# Include uncertainty estimation with custom MC passes
python scripts/evaluate.py --checkpoint results/checkpoints/best_model.ckpt --mc-passes 100

# Using Make
make evaluate
```

**Output files** (saved to `results/tables/`):
| File | Description |
|------|-------------|
| `test_metrics.json` | All metrics with 95% bootstrap confidence intervals |
| `confusion_matrix.csv` | 4×4 confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `baseline_comparison.csv` | Our model vs. 5 ML baselines |
| `biological_validation.json` | COSMIC driver gene and ClinVar star validation |
| `uncertainty_results.json` | MC Dropout uncertainty + calibration (ECE) |

## Inference

```bash
# Predict pathogenicity for new variants
python scripts/inference.py \
    --checkpoint results/checkpoints/best_model.ckpt \
    --input my_variants.tsv \
    --output predictions.json

# With custom uncertainty settings
python scripts/inference.py \
    --checkpoint results/checkpoints/best_model.ckpt \
    --input my_variants.tsv \
    --mc-passes 100 \
    --temperature 1.5
```

**Input format:** TSV/CSV file with variant information (gene, chromosome,
position, ref/alt alleles, variant type).

**Output per variant:**
```json
{
  "predicted_class": "Pathogenic",
  "confidence": 0.92,
  "uncertainty": 0.03,
  "calibrated_probability": {
    "Pathogenic": 0.92,
    "Likely Pathogenic": 0.05,
    "Benign": 0.02,
    "Likely Benign": 0.01
  },
  "recommendation": "High confidence prediction"
}
```

Predictions with high uncertainty are flagged with
`"recommendation": "Low confidence — manual review recommended"`.

## Hyperparameter Optimisation

```bash
# Run HPO with Optuna (50 trials, TPE sampler)
python scripts/run_hpo.py --config configs/sweep.yaml

# Run with more trials and a timeout
python scripts/run_hpo.py --config configs/sweep.yaml --n-trials 100 --timeout 3600

# Run HPO and retrain with best config
python scripts/run_hpo.py --config configs/sweep.yaml --retrain

# Train with the best-found config
python scripts/train.py --config configs/best.yaml
```

## Ablation Study

```bash
# Run all ablation variants
python scripts/run_ablation.py --base-config configs/default.yaml

# Using Make
make ablation
```

Tests each component's contribution by disabling it and measuring performance
drop: mutation, expression, methylation, CNV, cross-attention, focal loss.

## Figure Generation

```bash
# Generate all 12 publication-quality figures (PDF + PNG at 300 DPI)
python scripts/generate_figures.py

# Skip specific figures
python scripts/generate_figures.py --skip 1 3 9

# Using Make
make figures
```

Generates 12 figures for the paper: architecture diagram, dataset statistics,
learning curves, ROC/PR curves, confusion matrix, baseline comparison, ablation
study, SHAP analysis, attention weights, uncertainty analysis, and biological
validation.

## Reproducibility Verification

```bash
# Train twice with the same seed and compare outputs
python scripts/verify_reproducibility.py --config configs/default.yaml

# Quick verification with fewer epochs
python scripts/verify_reproducibility.py --config configs/default.yaml --max-epochs 5

# Force CPU for bit-exact reproducibility
python scripts/verify_reproducibility.py --config configs/default.yaml --device cpu
```

Reports the maximum absolute difference in predictions between two identical
runs. On CPU, predictions should be bit-for-bit identical. On GPU, differences
< 1e-4 are acceptable due to non-deterministic CUDA operations.

## Full Pipeline (Make)

```bash
make all    # download → train → evaluate → ablation → figures
```

Individual targets:
| Target | Description |
|--------|-------------|
| `make setup` | Create venv, install requirements |
| `make download` | Download ClinVar + cBioPortal data |
| `make train` | Train with default config |
| `make evaluate` | Evaluate best checkpoint |
| `make ablation` | Run ablation study |
| `make figures` | Generate publication figures |
| `make test` | Run pytest test suite |
| `make lint` | Run ruff + mypy |
| `make docker-build` | Build Docker image |
| `make docker-train` | Train in Docker (GPU) |

## Docker

```bash
# Build
docker build -t cancer-pathogenicity .

# Train (GPU)
docker run --gpus all \
    -v ./data:/app/data \
    -v ./results:/app/results \
    cancer-pathogenicity scripts/train.py --config configs/default.yaml

# MLflow UI
docker compose up mlflow
# → Open http://localhost:5000

# Run tests
docker compose up test
```

## Project Structure

```
cancer_mutation_pathogenicity/
├── configs/
│   ├── default.yaml              # Default training configuration
│   ├── sweep.yaml                # Optuna HPO search space
│   ├── ablation/                 # Ablation study configs (8 variants)
│   └── best.yaml                 # Best config from HPO (generated)
├── data/
│   ├── raw/                      # Downloaded raw data (ClinVar, cBioPortal)
│   ├── processed/                # Cleaned, merged data (Parquet)
│   └── splits/                   # Train/val/test splits (by gene)
├── results/
│   ├── checkpoints/              # Model checkpoints (.ckpt)
│   ├── figures/                  # Publication figures (PDF + PNG)
│   ├── tables/                   # Metrics, comparisons, reports (JSON + CSV)
│   └── logs/                     # Training logs
├── src/
│   ├── data/                     # Data loading, merging, PyTorch Dataset/DataModule
│   │   ├── clinvar_loader.py     # ClinVar label processing
│   │   ├── cbioportal_client.py  # cBioPortal REST API client
│   │   ├── data_merger.py        # Multi-omics data merger
│   │   ├── dataset.py            # PyTorch Dataset with modality masking
│   │   └── datamodule.py         # Lightning DataModule
│   ├── features/                 # Per-modality feature extraction
│   │   ├── mutation_features.py  # Variant type, AA properties, conservation
│   │   ├── expression_features.py# RNA-seq feature extraction
│   │   ├── methylation_features.py# DNA methylation features
│   │   ├── cnv_features.py       # Copy number variation features
│   │   └── clinical_features.py  # Clinical metadata features
│   ├── models/
│   │   ├── base.py               # BaseModel abstract class
│   │   ├── classifier.py         # Classification head (MLP → 4 classes)
│   │   ├── full_model.py         # PathogenicityPredictor (end-to-end)
│   │   ├── encoders/             # Per-modality neural network encoders
│   │   │   ├── mutation_encoder.py   # MLP + Transformer variants
│   │   │   ├── expression_encoder.py # Autoencoder + VAE + Transformer
│   │   │   ├── methylation_encoder.py# Autoencoder + VAE + Transformer
│   │   │   └── cnv_encoder.py        # FC + Attention variants
│   │   └── fusion/               # Multi-omics fusion strategies
│   │       ├── early_fusion.py       # Concatenation
│   │       ├── late_fusion.py        # Weighted averaging
│   │       ├── attention_fusion.py   # Self-attention
│   │       ├── cross_attention.py    # Cross-attention (default)
│   │       └── transformer_fusion.py # Full transformer
│   ├── training/                 # Training infrastructure
│   │   ├── losses.py             # Focal loss + weighted cross-entropy
│   │   ├── scheduler.py          # LR schedulers (cosine, plateau, one-cycle)
│   │   ├── callbacks.py          # MLflow logging, gradient monitoring
│   │   └── lightning_module.py   # PyTorch Lightning module
│   ├── evaluation/               # Evaluation and benchmarking
│   │   ├── metrics.py            # 20+ metrics with bootstrap CIs
│   │   ├── benchmarks.py         # Baseline comparisons (XGBoost, LightGBM, etc.)
│   │   └── biological_validation.py  # COSMIC CGC + ClinVar star validation
│   ├── explainability/           # Model interpretability
│   │   ├── shap_explainer.py     # SHAP (global + local)
│   │   ├── integrated_gradients.py   # Captum-based IG attributions
│   │   ├── lime_explainer.py     # LIME local explanations
│   │   └── attention_viz.py      # Attention weight visualisation
│   ├── uncertainty/              # Uncertainty estimation
│   │   ├── mc_dropout.py         # MC Dropout (epistemic uncertainty)
│   │   ├── deep_ensembles.py     # Deep ensemble predictions
│   │   └── calibration.py        # Temperature scaling + ECE
│   └── utils/                    # Shared utilities
│       ├── config.py             # YAML config loading with dot-access
│       ├── reproducibility.py    # seed_everything()
│       └── logging_setup.py      # Console + file logging
├── scripts/
│   ├── train.py                  # CLI: model training
│   ├── evaluate.py               # CLI: model evaluation
│   ├── inference.py              # CLI: prediction on new variants
│   ├── download_data.py          # CLI: data download pipeline
│   ├── run_ablation.py           # CLI: ablation study
│   ├── run_hpo.py                # CLI: hyperparameter optimisation
│   ├── generate_figures.py       # CLI: publication figure generation
│   └── verify_reproducibility.py # CLI: reproducibility verification
├── tests/                        # 545+ pytest tests
│   ├── test_data_loading.py
│   ├── test_data_merger.py
│   ├── test_features.py
│   ├── test_encoders.py
│   ├── test_fusion.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_metrics.py
│   ├── test_biological_validation.py
│   ├── test_explainability.py
│   ├── test_uncertainty.py
│   ├── test_ablation.py
│   └── test_hpo.py
├── Dockerfile                    # Reproducible container build
├── docker-compose.yml            # Multi-service Docker setup
├── Makefile                      # Build automation
├── pyproject.toml                # Python packaging (setuptools)
├── requirements.txt              # Python dependencies
├── ARCHITECTURE.md               # Detailed architecture guide
├── CLAUDE.md                     # Coding standards and rules
└── README.md                     # This file
```

## Testing

```bash
# Run the full test suite (545+ tests)
pytest tests/ -v --tb=short

# Run specific test modules
pytest tests/test_models.py -v
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Using Make
make test
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{cancer_mutation_pathogenicity_2026,
  title     = {Multi-Omics Deep Learning with Cross-Attention Fusion for
               Cancer Mutation Pathogenicity Prediction},
  author    = {Cancer Mutation Pathogenicity Team},
  journal   = {TBD},
  year      = {2026},
  note      = {Under review},
  url       = {https://github.com/your-org/cancer-mutation-pathogenicity}
}
```

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Cancer Mutation Pathogenicity Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
