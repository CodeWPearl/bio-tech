# Architecture Guide

A plain-language walkthrough of **what this project does, why each piece exists,
and how data flows from raw download to a final prediction**. Read this top to
bottom once and the empty `src/` folders will make complete sense.

---

## 1. The problem in one sentence

> Given a **mutation in a gene**, predict whether it is **Pathogenic, Likely
> Pathogenic, Benign, or Likely Benign** — and also explain *why* and say *how
> confident* the model is.

That's a **4-class classification** problem. The twist that makes it research-grade
is *how* we represent each mutation: not just from the DNA change itself, but by
combining several different biological measurements ("multi-omics").

---

## 2. Why "multi-omics"? (the core idea)

A single mutation can look harmless in isolation but be dangerous in context.
So instead of looking at one type of data, we look at **four complementary views**
of the gene/sample. Each view is called a *modality*.

| # | Modality | Plain-English meaning | Where it comes from | Embed dim |
|---|----------|----------------------|---------------------|-----------|
| 1 | **Mutation** | The variant itself: position on the genome, type (missense, nonsense…), the amino-acid change, how conserved that spot is across species | ClinVar + cBioPortal | 128 |
| 2 | **Expression** | How *active* the gene is — how much RNA it produces (RNA-seq) | cBioPortal / TCGA | 256 |
| 3 | **Methylation** | Epigenetic "on/off dimmer switches" chemically attached to DNA | cBioPortal / TCGA | 128 |
| 4 | **CNV** (Copy Number Variation) | Whether the gene is amplified (extra copies) or deleted | cBioPortal / TCGA | 64 |

**Labels** (the 4 classes we predict) come from **ClinVar** — a public database of
expert-curated variant classifications.
**Features** (the 4 modalities above) come from **cBioPortal**, which serves
TCGA PanCancer Atlas data.

> Think of it like diagnosing a patient: you don't decide from one symptom. You
> combine blood tests, imaging, history, and genetics. Each modality is one "test."

---

## 3. End-to-end data flow

This is the single most important diagram. Every folder in `src/` maps to one box.

```
   ClinVar (labels)        cBioPortal / TCGA (omics features)
        │                            │
        └──────────────┬─────────────┘
                       ▼
   ┌───────────────────────────────────────────────┐
   │ src/data/                                       │  ← download, parse, clean
   │  • fetch raw files                              │     align records by
   │  • match each variant to its gene + sample      │     gene + sample
   │  • SPLIT BY GENE (not by variant) → train/val/test
   └───────────────────────────────────────────────┘
                       ▼
   ┌───────────────────────────────────────────────┐
   │ src/features/                                   │  ← turn raw data into
   │  • one feature-extractor per modality           │     numeric feature vectors
   │    (mutation / expression / methylation / cnv)  │
   └───────────────────────────────────────────────┘
                       ▼
   ┌───────────────────────────────────────────────┐
   │ src/models/encoders/                            │  ← one small neural net per
   │  mutation_features    → 128-dim embedding       │     modality, compresses
   │  expression_features  → 256-dim embedding       │     each view into a dense
   │  methylation_features → 128-dim embedding       │     "summary vector"
   │  cnv_features         → 64-dim  embedding       │
   └───────────────────────────────────────────────┘
                       ▼
   ┌───────────────────────────────────────────────┐
   │ src/models/fusion/                              │  ← THE HEART of the model:
   │  combine the 4 embeddings → 1 fused vector (256)│     how to merge the views
   │  strategy chosen in config: fusion_type         │
   └───────────────────────────────────────────────┘
                       ▼
   ┌───────────────────────────────────────────────┐
   │ classifier head → softmax over 4 classes        │  ← final probabilities
   └───────────────────────────────────────────────┘
                       ▼
        Pathogenic / Likely Pathogenic / Benign / Likely Benign
                       │
        ┌──────────────┴───────────────┐
        ▼                              ▼
  src/explainability/            src/uncertainty/
  "WHICH features/omics          "HOW confident is
   drove this prediction?"        this prediction?"
```

### What "encoder → fusion → head" actually means
- **Encoder**: a small neural network that takes one modality's raw numbers and
  squeezes them into a fixed-size vector (an *embedding*) that captures the useful
  signal. Four modalities → four encoders → four embeddings.
- **Fusion**: combines those four embeddings into a *single* vector that represents
  the whole picture.
- **Head**: a final layer that turns the fused vector into 4 probabilities (one per
  class) that sum to 1 (via *softmax*). The biggest probability is the prediction.

---

## 4. The fusion strategies (the key research lever)

How you combine the four modalities is the main thing this project experiments
with. It's a single config switch: `model.fusion_type`. The five options:

| Strategy | How it merges modalities | Trade-off |
|----------|--------------------------|-----------|
| `early` | Glue all raw features together first, then one big network | Simple, but ignores that modalities are different kinds of data |
| `late` | Train a separate model per modality, then average their votes | Robust if a modality is missing, but the modalities never "talk" to each other |
| `attention` | Learn a weight for each modality per sample (how much to trust each) | Adapts per-sample, but still no cross-modality interaction |
| **`cross_attention`** *(default)* | Each modality **attends to** the others, so the model learns *interactions* | Most expressive — e.g. "this mutation matters *only when* expression is high" |
| `transformer` | Treat the 4 embeddings as a 4-token sequence fed to a transformer | Powerful, more data/compute hungry |

> **Why cross-attention is the default:** biology is full of interactions. A mutation's
> effect often depends on whether the gene is even being expressed, or whether it's
> been amplified. Cross-attention lets the model condition one modality on another
> instead of treating them independently.

---

## 5. The supporting subsystems (rest of `src/`)

| Folder | Job | Why it matters |
|--------|-----|----------------|
| `src/training/` | The training loop (PyTorch Lightning). Uses **focal loss** + stratified sampling for class imbalance, and **early stopping** (`patience`). | Pathogenic variants are rarer than benign ones; focal loss stops the model from just predicting the majority class. |
| `src/evaluation/` | Metrics, ROC/PR curves, confusion matrices → saved to `results/`. | How we prove the model actually works. |
| `src/explainability/` | SHAP / LIME / captum — "which features and which omics drove this prediction?" | Clinicians and journal reviewers won't trust a black box. This makes predictions *interpretable*. |
| `src/uncertainty/` | Confidence + calibration estimates (e.g. MC-dropout). | A prediction without a confidence level is dangerous in a clinical setting. |
| (baselines) | XGBoost / LightGBM / scikit-learn classical models. | We must show the deep model **beats** simpler methods, or it isn't worth the complexity. |
| `src/utils/` | config, reproducibility (seeds), logging. **← implemented now** | The plumbing everything else depends on. |

External tools (from `requirements.txt`):
- **Optuna** → automatic hyperparameter search.
- **MLflow** → logs every experiment (config + seed + metrics) to `mlruns/` so runs
  are tracked and comparable.

---

## 6. Critical design rules (and the reasoning)

These come from `CLAUDE.md` and are baked into the architecture:

- **Split by gene, not by variant.** Two variants in the same gene share a lot of
  context. If one lands in train and another in test, the model "cheats" by
  memorizing the gene — *data leakage*. Splitting whole genes into one side only
  gives an honest estimate of real-world performance.
- **GRCh38 genome build everywhere.** Mixing genome builds = mismatched coordinates
  = silent, catastrophic bugs.
- **Entrez Gene ID as the primary key** (HUGO symbol only for display). Symbols
  change over time; numeric IDs are stable.
- **Everything is config-driven and reproducible.** No hardcoded paths or magic
  numbers; one seed controls all randomness (`src/utils/reproducibility.py`); the
  config + seed are logged to MLflow so any run can be reproduced exactly.
- **Document missingness; use modality masking.** Not every sample has all four
  omics. The fusion layer is designed to handle a missing modality rather than crash.

---

## 7. What's built *right now* vs. later

This repo is currently a **scaffold**. The decision was to first lay down a clean,
reproducible skeleton, then fill in the science.

**✅ Built and working now:**
- Full directory tree + Python packaging (`pyproject.toml`, `requirements.txt`).
- `configs/default.yaml` — all the knobs (embed dims, fusion type, training params).
- `src/utils/config.py` — loads YAML with dot-access (`cfg.model.fusion_dim`),
  CLI overrides, and validation.
- `src/utils/reproducibility.py` — `seed_everything()`.
- `src/utils/logging_setup.py` — console + file logging.
- `scripts/train.py`, `evaluate.py`, `inference.py` — CLI entry points (currently
  stubs that load config + logging and tell you the pipeline isn't built yet).
- **`src/data/` — the full data layer is built:** `clinvar_loader.py` (download +
  clean the labels), `cbioportal_client.py` (download the multi-omics features),
  `data_merger.py` (join labels↔features on gene + position, attach omics,
  split **by gene** into train/val/test), `dataset.py` (PyTorch Dataset with
  modality masking and custom collate), `datamodule.py` (Lightning DataModule
  with feature caching and class-weighted sampling). `scripts/download_data.py`
  runs the whole thing end-to-end from one command.
- **`src/features/` — per-modality feature extraction is built:** five extractors
  (mutation, expression, methylation, CNV, clinical) orchestrated by
  `FeaturePipeline` with fit/transform separation and save/load support.

**🔜 Empty packages, to be implemented in future sessions (in dependency order):**
1. `src/models/encoders/` — the four modality encoders.
3. `src/models/fusion/` — the five fusion strategies.
4. `src/training/` — Lightning training loop + focal loss.
5. `src/evaluation/`, `src/explainability/`, `src/uncertainty/` — analysis layers.

---

## 8. How to read the config (your control panel)

Open `configs/default.yaml`. Mapping config → architecture:

```yaml
model:
  mutation_embed_dim: 128      # size of the mutation encoder's output (box 3 above)
  expression_embed_dim: 256    # size of the expression encoder's output
  methylation_embed_dim: 128   # size of the methylation encoder's output
  cnv_embed_dim: 64            # size of the CNV encoder's output
  fusion_dim: 256              # size of the single fused vector (box 4)
  num_classes: 4               # the 4 pathogenicity labels (final head)
  fusion_type: "cross_attention"  # which fusion strategy (section 4)
training:
  focal_loss_gamma: 2.0        # how hard focal loss focuses on rare/hard cases
  patience: 15                 # stop if no improvement for 15 epochs (early stopping)
```

Change a value here, and (once the pipeline is built) the whole model rebuilds
around it — no code edits needed. That's the "config-driven" rule in action.

---

*For coding standards, data-source URLs, and git conventions, see `CLAUDE.md`.*
*For the build/setup commands, see `README.md`.*
