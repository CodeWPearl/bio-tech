# Beginner's Guide & Session Log

> **Who this is for:** someone who has *never* used Python, machine learning, a
> terminal, or any of these tools. Everything is explained in plain English. No
> prior knowledge assumed.
>
> **This file is a living document.** Every time work is done on the project, a new
> entry is added to the [Session Log](#part-4--session-log) at the bottom, recording
> exactly what commands were run and what changed.

---

## Part 0 — The absolute basics (start here)

### What is a "terminal"?
A **terminal** (also called a "command line", "shell", "PowerShell", or "console")
is a window where you type commands as text instead of clicking buttons. You type
a command, press **Enter**, and the computer does it.

- On **Windows** the terminal is called **PowerShell**. Open it by pressing the
  Windows key, typing `PowerShell`, and hitting Enter.
- The blinking spot where you type is called the **prompt**.

### What does it mean to "run a command"?
It means: type the command exactly as written, then press **Enter**. That's it.

### One rule that saves you: be in the right folder
Commands run *inside whatever folder the terminal is currently "looking at."*
This project lives in:

```
c:\Users\Pearl Queen Ray\Desktop\bio pro\cancer_mutation_pathogenicity
```

To move the terminal into that folder, run this **first, every session**:

```powershell
cd "c:\Users\Pearl Queen Ray\Desktop\bio pro\cancer_mutation_pathogenicity"
```

- `cd` = **c**hange **d**irectory ("directory" is just another word for "folder").
- The part in quotes is the folder's address. Quotes are needed because the path
  has spaces in it.

After this, the terminal is "inside" the project and all other commands will work.

---

## Part 1 — What is this project, in one breath?

It's a program that looks at a **genetic mutation** (a tiny change in someone's DNA)
and predicts whether that mutation is **dangerous (causes cancer)** or **harmless**.
It does this using **AI** trained on real public medical data. For the full
science, read [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Part 2 — The tech stack, explained like you're new (and *why* we use each)

"Tech stack" just means "the collection of tools the project is built from."
Here's every major one, what it is, and why it's here.

### The language
| Tool | What it is (plain English) | Why we use it |
|------|----------------------------|---------------|
| **Python** | A programming language — the language all our code is written in. Famous for being readable and dominant in AI. | It's the standard language for AI/medical research; nearly every tool below is built for it. |

### The "brain" — deep learning
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **PyTorch** (`torch`) | The engine that builds and trains neural networks (the "AI brain"). It does the heavy math. | Industry + research standard; flexible enough for our custom multi-omics model. |
| **PyTorch Lightning** | A neat "wrapper" around PyTorch that handles the boring, repetitive training code for us (saving progress, using the GPU, stopping early). | Lets us focus on the *science*, not plumbing; makes experiments reproducible. |

### The "comparison" models — classical machine learning
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **scikit-learn** | A toolbox of classic, simpler AI methods + utilities for splitting data and measuring accuracy. | We need simple baselines and standard metrics to *prove* our fancy model is actually better. |
| **XGBoost**, **LightGBM** | Two very strong "decision-tree" models — a different, non-neural style of AI. | They're famously hard to beat on table-style data, so beating them is a real result worth publishing. |

### The "why did it decide that?" tools — explainability
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **SHAP**, **LIME**, **captum** | Tools that open the AI's "black box" and show *which pieces of data* pushed it toward its decision. | Doctors and scientific reviewers will not trust a prediction they can't understand. This is essential for medical AI. |

### The "tuning & tracking" tools
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **Optuna** | Automatically tries many settings to find the best-performing ones. | Hand-tuning is slow and biased; Optuna searches smartly for us. |
| **MLflow** | A logbook that records every experiment — its settings, and how well it did. | Research demands reproducibility: we must be able to look up and re-run any past result. |

### The "data handling & charts" tools
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **pandas** | Works with data tables (like Excel, but in code). | All our medical data starts as big tables; pandas loads and cleans them. |
| **numpy** | Fast math on big lists of numbers. | The foundation everything numerical is built on. |
| **pyarrow** | Reads/writes **Parquet** files — a compact, fast table format. | We save the cleaned ClinVar data as a `.parquet` file so it loads quickly later. |
| **matplotlib**, **seaborn**, **plotly** | Make charts and graphs (static and interactive). | We need figures for the paper and to understand results visually. |
| **requests** | Downloads files/data from the internet. | We fetch the medical data from public servers (ClinVar, cBioPortal). |
| **tqdm** | Shows a progress bar for slow tasks. | So you can see something is working and how long it'll take. |
| **PyYAML** | Reads our settings file (`configs/default.yaml`). | All the project's knobs live in that file; PyYAML loads them. |

### The "keep the code clean & correct" tools
| Tool | What it is | Why we use it |
|------|------------|---------------|
| **pytest** | Runs automated tests that check our code does what it should. | Catches bugs early, before they ruin an experiment. |
| **ruff** | Checks code style and formats it neatly, automatically. | Keeps the codebase consistent and readable. |
| **mypy** | Checks that we're using data of the right *type* (e.g. not treating text as a number). | Catches a whole class of mistakes before the code even runs. |
| **git** | Saves snapshots of the code over time (version control). | So we can undo mistakes and track every change. |

---

## Part 3 — The commands you'll actually run

Each command is shown, then explained **word by word**. Run them from inside the
project folder (see [Part 0](#one-rule-that-saves-you-be-in-the-right-folder)).

### 3.1 One-time setup (only done once, on a new computer)

**Step 1 — create a "virtual environment":**
```powershell
python -m venv .venv
```
- `python` = run the Python program.
- `-m venv` = use Python's built-in "venv" tool. A **virtual environment** is a
  private, isolated box that holds this project's tools, so they don't clash with
  other projects on your computer.
- `.venv` = the name of the folder that box lives in.

**Step 2 — "activate" that environment:**
```powershell
.venv\Scripts\Activate.ps1
```
- This switches the terminal to *use* the private box you just made. You'll know it
  worked because `(.venv)` appears at the start of your prompt. Do this **every
  session** before running project commands.

**Step 3 — install all the tools the project needs:**
```powershell
pip install -r requirements.txt
```
- `pip` = Python's tool **installer** (it downloads and installs tools).
- `install` = the action: install something.
- `-r requirements.txt` = "**r**ead the shopping list in the file `requirements.txt`
  and install everything on it." That file lists every tool from Part 2.

**Step 4 — install *this project itself* so its commands work:**
```powershell
pip install -e .
```
- `-e` = "**e**ditable" — install it in a way where your code edits take effect
  immediately, without reinstalling.
- `.` = "the project in the current folder."
- After this, the words `train`, `evaluate`, and `inference` become usable commands.

### 3.2 Every-session startup (the routine)
```powershell
cd "c:\Users\Pearl Queen Ray\Desktop\bio pro\cancer_mutation_pathogenicity"
.venv\Scripts\Activate.ps1
```
That's it — move into the folder, activate the environment, and you're ready.

### 3.3 The main project commands
> ⚠️ These are wired up but **not fully functional yet** — the AI pipeline is built
> step-by-step in future sessions. For now they load settings and report they're stubs.

**Train the AI model:**
```powershell
train --config configs/default.yaml
```
- `train` = the command that teaches the AI from data.
- `--config configs/default.yaml` = "use the settings written in this file."

**Test how good a trained model is:**
```powershell
evaluate --checkpoint results/checkpoints/best_model.ckpt
```
- `evaluate` = measure the model's accuracy.
- `--checkpoint ...` = "use the saved model file at this location." A
  **checkpoint** (`.ckpt`) is a saved snapshot of a trained model.

**Use the model on brand-new mutations:**
```powershell
inference --checkpoint results/checkpoints/best_model.ckpt --input my_variants.tsv
```
- `inference` = make predictions on new data the model has never seen.
- `--input ...` = the file of new mutations you want predictions for.

### 3.4 Code-quality commands (for keeping things healthy)
```powershell
pytest tests/ -v          # run all automated tests (-v = verbose, show each one)
ruff check src/ scripts/  # check code style for problems
ruff format src/ scripts/ # auto-tidy the code's formatting
mypy src/                 # check for type mistakes
```

### 3.5 Saving your work with git
```powershell
git status                # see what changed
git add .                 # stage all changes ("." means everything)
git commit -m "message"   # save a snapshot with a short description
```
- A **commit** is a saved snapshot. The `-m "message"` is a short note describing
  what you changed, e.g. `-m "feat: add mutation encoder"`.

---

## Part 4 — Session Log

> A dated record of every work session: what was done, what commands were run, and
> what to do next. **Newest at the top.** This section is updated at the end of
> every session.

### Session 5 — Feature extraction modules — *2026-06-23*

**Goal:** Turn each modality's raw numbers (mutation calls, gene expression,
methylation beta-values, copy-number GISTIC scores, clinical attributes) into
clean, numeric **feature vectors** that the AI encoders can consume. This is
the bridge between "raw data table" and "model input."

**Plain-English background (what the new words mean):**
- **Feature extraction** — converting raw data (text, categories, wide number
  matrices) into fixed-width columns of numbers. AI models need numbers, not
  text like "Missense_Mutation" or "Stage IIIA."
- **One-hot encoding** — turning a category into a row of 0s with a single 1.
  For example, chromosome "17" becomes 24 columns, all 0 except position 17.
- **Z-score standardisation** — subtracting the average and dividing by the
  spread, so every gene column is centred at 0 with similar scale. Learned
  from training data only (never peeking at test data).
- **Quantile normalisation** — forces all samples to share the same
  statistical shape, removing technical variation between samples.
- **M-value transform** — converts methylation beta-values (0–1) into an
  unbounded scale that behaves better for statistical models.
- **Grantham score / BLOSUM62** — biochemical distance metrics that quantify
  how "different" one amino acid is from another. A mutation that swaps two
  similar amino acids (low score) is less likely to be damaging.
- **COSMIC Cancer Gene Census** — a curated list of ~730 genes known to drive
  cancer; being in this list is itself a useful feature.
- **fit / transform pattern** — the standard approach: first `fit` learns
  statistics (means, categories) from training data, then `transform` applies
  those same statistics to any split (train/val/test) without data leakage.

**What was created/changed:**
- `src/features/mutation_features.py` — **MutationFeatureExtractor**: produces
  ~42 numeric features per mutation:
  - Variant type one-hot (8 categories + Other)
  - Amino acid change properties: Grantham score, BLOSUM62 score,
    hydrophobicity/charge/size deltas (full lookup tables for all 20 amino
    acids and 190 Grantham + 210 BLOSUM62 pairs bundled in the module)
  - Gene-level: COSMIC census membership, mutation frequency (learned),
    normalised gene length
  - Positional: chromosome one-hot (24), normalised position within gene

- `src/features/expression_features.py` — **ExpressionFeatureExtractor**:
  log2(x+1) → quantile normalisation → z-score per gene. Imputes missing
  values with per-gene median from training data.

- `src/features/methylation_features.py` — **MethylationFeatureExtractor**:
  clips beta values to [0.001, 0.999] → M-value transform → z-score per
  gene. Imputes missing with median.

- `src/features/cnv_features.py` — **CNVFeatureExtractor**: supports ordinal
  mode (keep -2..2 values) or one-hot mode (5 binary columns per gene). Can
  intersect with the mutation gene set to reduce dimensionality. Missing → 0.

- `src/features/clinical_features.py` — **ClinicalFeatureExtractor**: age
  min-max normalised, sex binary, stage ordinal (1–4, with regex for
  sub-stages), cancer type one-hot. Missing imputed with training median/mode.

- `src/features/__init__.py` — **FeaturePipeline**: orchestrates all five
  extractors. `fit(df)` on training data, `transform(df)` on any split.
  Returns a dict of per-modality arrays or a combined matrix. Supports
  `save()`/`load()` via pickle for reproducible inference.

- `tests/test_features.py` — **56 new tests** (now 117 total) with synthetic
  data: output shapes, feature-name consistency, missing-value imputation,
  edge cases (unknown variants, unparseable protein changes, extreme betas,
  unseen cancer types), fit/transform separation, pipeline save/load
  round-trip, and one-hot row-sum validation.

**Commands run this session (and what they did):**
```powershell
# Ran the automated tests for all feature extractors (and all earlier ones):
python -m pytest tests/ -v                                     # → 117 passed
```

> ℹ️ **No new tools to install:** the feature extractors use only `numpy` and
> `pandas`, which were already installed. No scikit-learn needed.

**Status:** ✅ Done and verified — all 117 tests pass offline; each extractor
handles missing data gracefully and produces no NaN values.

**What's next (Session 6):** Build `src/models/encoders/` — the per-modality
neural network encoders that compress each feature vector into a learned
embedding, and the fusion module that combines them for the final prediction.

---

### Session 4 — Cross-database merger + train/val/test splits — *2026-06-22*

**Goal:** Connect the two halves of the data. ClinVar (Session 2) tells us *which
mutations are dangerous* (the labels). cBioPortal (Session 3) gives us the *clues*
about each tumour sample (gene activity, DNA tags, copy-number, patient info). This
session **joins them into one table** the AI can learn from, then **splits that
table into three piles** for honest training and testing.

**Plain-English background (what the new words mean):**
- **Merge / join** — gluing two tables together by matching rows. Here we match a
  mutation in cBioPortal to its verdict in ClinVar using the **gene name** and the
  **position on the DNA**. Positions can differ by a few letters between databases
  (different ways of writing the same change), so we allow a **±5 letter wiggle
  room** ("fuzzy match").
- **Feature** — a clue the AI learns from (gene activity, etc.). **Label** — the
  answer we want it to predict (Pathogenic / Benign / …).
- **Top-2000 most variable genes** — there are ~20,000 genes; using all of them is
  too much. We keep the 2,000 that *differ the most* between samples (the genes
  that carry the most information), for expression and methylation.
- **Train / validation / test split** — three piles of data: one to *teach* the AI
  (train), one to *tune* it (validation), one to *grade* it fairly on data it has
  never seen (test).
- **Split BY GENE (not by variant)** — the golden rule here. Two mutations in the
  *same* gene are very alike, so if one is in the "teach" pile and its near-twin is
  in the "grade" pile, the AI looks smarter than it is (it basically memorised the
  gene). This is **data leakage**. We prevent it by keeping *every* mutation of a
  gene together in *one* pile. We also **stratify** — keep the mix of answer-classes
  roughly equal across the three piles.

**What was created/changed:**
- `src/data/data_merger.py` — the new module. In plain English it:
  1. **`DataMerger.merge_clinvar_with_mutations(...)`** — matches each cBioPortal
     mutation to its ClinVar verdict (same gene + chromosome, position within 5
     letters; if several fit, the closest wins). Logs how many matched, how many
     didn't, and the match rate. Unmatched mutations are dropped.
  2. **`DataMerger.attach_omics_features(...)`** — for each matched mutation, looks
     up that sample's expression / methylation / copy-number / clinical data and
     glues it on. Missing data is handled gracefully with `has_expression`,
     `has_methylation`, `has_cnv`, `has_clinical` true/false columns. Expression
     and methylation are trimmed to their **top-2000 most variable genes**. Each
     modality's gene columns are prefixed (`expr_`, `meth_`, `cnv_`) so they never
     clash.
  3. **`DataSplitter.split_by_gene(...)`** — splits the table into train/val/test
     **by gene** (no gene leakage), stratified by label, deterministic from the
     seed. Saves the three piles as `.parquet` files in `data/splits/` plus the
     gene lists in `data/splits/gene_splits.json`.
  4. **`DataSplitter.print_split_statistics(...)`** — logs each pile's size, gene
     count and label mix, and **verifies no gene appears in two piles**.
  5. **`run_full_pipeline(...)`** — does all of the above end-to-end from the
     already-downloaded files, and writes `data/processed/merged_dataset.parquet`.
- `scripts/download_data.py` — a **one-command, end-to-end data pipeline**: it
  downloads + cleans ClinVar, downloads all configured cBioPortal studies, then
  merges and splits everything. Use `--skip-download` to only rebuild the merge +
  splits from data already on disk.
- `tests/test_data_merger.py` — **24 new tests** (now 61 total) with tiny fake data
  (no internet): exact + fuzzy + no-match merges, closest-match selection, missing
  modalities, top-variable-gene trimming, split-by-gene **leakage checks**, the
  statistics, and the multi-study loader.

**Commands run this session (and what they did):**
```powershell
# Ran the automated tests for the new merger (and all earlier ones):
python -m pytest tests/ -q                                     # → 61 passed
```

**How to build the full dataset yourself (after ClinVar + cBioPortal downloads):**
```powershell
# Everything at once (download → clean → merge → split):
python scripts/download_data.py --config configs/default.yaml
# Or, if the raw data is already downloaded, just rebuild the merge + splits:
python scripts/download_data.py --skip-download
```

> ℹ️ **No new tools to install:** the gene-level split uses only Python's built-in
> `random` (no scikit-learn needed yet), plus `pandas`/`pyarrow` from earlier
> sessions.

**Status:** ✅ Done and verified — all 61 tests pass offline; the end-to-end
`run_full_pipeline` was smoke-tested on synthetic on-disk data (merge → attach →
split → save all working, no gene leakage).

**What's next (Session 5):** Build `src/features/` — turn each modality's raw
numbers into clean, model-ready feature vectors (the step before the AI encoders).

---

### Session 3 — cBioPortal multi-omics client — *2026-06-22*

**Goal:** Build the second piece of the data pipeline — the code that downloads
the **multi-omics** cancer data (gene activity, DNA tagging, copy-number, and
patient info) from **cBioPortal**. This is the "evidence" that pairs with the
ClinVar "answer key" from Session 2, so the AI can later learn from both.

**Plain-English background (what the new words mean):**
- **cBioPortal** — a big, free public website + API holding real cancer-patient
  data from the **TCGA** project (a landmark cancer-genomics dataset). No login
  needed for public studies.
- **API** — a way for our code to *ask a server for data* over the internet and
  get a tidy machine-readable answer back (instead of clicking around a website).
- **The four "omics" we fetch (think of them as four kinds of clues per patient):**
  - **Mutations** — the actual DNA spelling changes in the tumour.
  - **Expression (RNA-seq)** — how *active* each gene is (turned up or down).
  - **Methylation** — chemical "tags" on DNA that switch genes on/off.
  - **CNV (copy-number)** — whether chunks of DNA were *duplicated or deleted*.
- **Clinical data** — the patient facts: cancer type, age, sex, stage, grade, and
  survival (how long they lived, and whether they were alive at last check).
- **Rate limiting** — politely not hammering the server: we cap ourselves at
  **10 requests per second** so we're a good internet citizen.
- **Retry with backoff** — if the server hiccups, we automatically try again a
  few times, waiting a little longer each time, instead of giving up.
- **Pagination** — the server hands back huge results in *pages* (max 10,000 rows
  each); our code keeps asking for the next page until there are no more.

**What was created/changed:**
- `src/data/cbioportal_client.py` — the new module. It provides one class,
  `CBioPortalClient`, that in plain English:
  1. **Connects politely** — caps itself to 10 requests/second and auto-retries
     (3 times, waiting longer each time) when the server is briefly unhappy.
  2. **Fetches mutations** for a study, automatically flipping through all the
     pages, and returns a tidy table (gene, position, protein change, type, etc.).
  3. **Fetches expression, methylation, and copy-number** as *matrices* (one row
     per patient-sample, one column per gene). Because different studies name
     these datasets slightly differently, it **tries several known names** and
     uses the first that exists.
  4. **Fetches clinical data** and tidies it into one row per sample with the
     columns we care about (cancer type, age, sex, stage, grade, survival).
  5. **Handles missing data gracefully** — if a study simply doesn't have, say,
     methylation, it logs a warning and returns an *empty* table instead of
     crashing the whole run.
  6. **`download_study(...)`** — fetches all five of the above for one study and
     saves each as a fast `.parquet` file under `data/raw/{study_id}/`.
  7. **`download_all_studies(...)`** — loops over many studies, keeps going even
     if one fails, and writes a `manifest.json` summarising what was downloaded.
- `tests/test_data_loading.py` — appended **18 new tests** (now 37 total) that
  fake the internet (no real network) and check: the retry settings, the rate
  limiter, page-flipping, the graceful "dataset missing" (404) handling, the
  table-reshaping (pivoting), the clinical tidy-up, and the save/manifest logic.

**Commands run this session (and what they did):**
```powershell
# Ran the automated tests for the new cBioPortal client (and the old ones):
python -m pytest tests/test_data_loading.py -q                 # → 37 passed
```

**How to use the new client (example, run inside the project + venv):**
```python
from src.data.cbioportal_client import CBioPortalClient

client = CBioPortalClient()
# Download every modality for one TCGA breast-cancer study into data/raw/:
client.download_study("brca_tcga_pan_can_atlas_2018", "data/raw")
# Or several at once (writes data/raw/manifest.json too):
client.download_all_studies(
    ["brca_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018"],
    "data/raw",
)
```

> ℹ️ **No new tools to install:** this uses `requests` (downloads) and `pandas`/
> `pyarrow` (tables/Parquet), which were already installed in Session 2.

**Status:** ✅ Done and verified — all 37 tests pass offline. (The client talks to
the live cBioPortal API only when you actually call `download_study`; the tests
never touch the network.)

**What's next (Session 4):** Join the ClinVar labels with the cBioPortal omics
features into a single training-ready dataset, and build the train/val/test split
(split *by gene*, not by variant, to avoid data leakage — see `CLAUDE.md`).

---

### Session 2 — ClinVar data loader — *2026-06-22*

**Goal:** Build the first real piece of the data pipeline — the code that
downloads the public **ClinVar** database (the "answer key" of which mutations are
dangerous) and cleans it into a tidy table the AI can later learn from.

**What was created/changed:**
- `src/data/clinvar_loader.py` — the new module. In plain English it:
  1. **Downloads** ClinVar's `variant_summary.txt.gz` from NCBI (~419 MB) with a
     progress bar. It **skips** the download if you already have a complete copy,
     and — importantly — if the download gets cut off partway (NCBI's servers do
     this often), it **resumes from where it stopped** instead of starting over.
  2. **Cleans** the table — keeps only GRCh38 (one genome build), only clear
     verdicts (Pathogenic / Likely pathogenic / Benign / Likely benign), drops
     "uncertain"/"conflicting" rows, requires a trustworthy review (≥1 star),
     drops rows with no gene, and keeps only small mutations (not huge ones).
     This is done in **chunks** (a slice at a time) so it doesn't run out of
     memory on the ~9-million-row file.
  3. **Labels** each row with a number (0–3) the AI can use.
  4. **Removes duplicates**, keeping the best-supported copy of each mutation.
  5. **Saves** the result to `data/processed/clinvar_processed.parquet`.
  6. **Logs a summary** (how many of each class, top genes, etc.).
- `tests/test_data_loading.py` — 19 automated tests checking every step with tiny
  fake data (the real download is faked, so tests run offline in seconds),
  including tests that the download resumes and retries after a dropped connection.
- `requirements.txt` — added **pyarrow** (needed to save the `.parquet` file).

**Commands run this session (and what they did):**
```powershell
# Installed the tools this step needs (data tables, Parquet, downloads, tests):
python -m pip install pandas pyarrow requests tqdm pytest   # → installed

# Ran the automated tests for the new data loader:
python -m pytest tests/test_data_loading.py -q                # → 19 passed

# Downloaded the real ClinVar database and built the clean dataset:
python -m src.data.clinvar_loader                            # → wrote the .parquet
```

**How to run the data pipeline yourself again (it's already been run once):**
```powershell
python -m src.data.clinvar_loader   # skips the download (already have it), rebuilds the table
```

**Real result (this machine, 2026-06-22):** Built
`data/processed/clinvar_processed.parquet` (70 MB) holding **1,367,228** clean,
labelled, de-duplicated GRCh38 variants:

| Class | Count |
|-------|-------|
| Likely benign | 960,951 |
| Benign | 190,089 |
| Pathogenic | 136,624 |
| Likely pathogenic | 79,564 |

(Top genes by variant count: TTN, BRCA2, NF1, NEB, BRCA1, ATM, … — the usual
large/cancer-relevant genes, which is a good sanity check.)

> ⚠️ **Note on class balance:** "Likely benign" hugely outnumbers the rest. That's
> a real-world *class imbalance* the model will need to handle (the project plans
> to use focal loss + stratified sampling for exactly this — see `CLAUDE.md`).

**Status:** ✅ Done and verified — all 19 tests pass, the real dataset is built and
checked (0 duplicate loci, all GRCh38, labels 0–3 present).

**What's next (Session 3):** Build the cBioPortal loader (`src/data/`) to fetch
the multi-omics TCGA data (gene expression, methylation, copy-number) that pairs
with these ClinVar labels.

---

### Session 1 — Project scaffold + utilities — *2026-06-22*

**Goal:** Build the empty skeleton of the project and the core "plumbing" tools, so
future sessions have a clean, reproducible foundation.

**What was created:**
- The full folder structure (`src/`, `scripts/`, `tests/`, `configs/`, `data/`,
  `results/`, etc.).
- Packaging files: `pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`.
- The settings file: `configs/default.yaml`.
- Three working utility modules in `src/utils/`:
  - `config.py` — reads the settings file.
  - `reproducibility.py` — locks in randomness so results repeat exactly.
  - `logging_setup.py` — records what the program does to screen + a log file.
- Command stubs: `scripts/train.py`, `evaluate.py`, `inference.py`.
- Documentation: `ARCHITECTURE.md` (the science) and this `GUIDE.md`.

**Commands run this session (and what they did):**
```powershell
# Verified the code can be imported without errors (the success check):
python -c "from src.utils.config import load_config; print('OK')"   # → printed OK

# Installed just the settings-reader tool to test the config loader for real:
python -m pip install pyyaml                                          # → installed
# Then confirmed settings load, overrides work, and bad input is rejected → all OK
```

**Status:** ✅ Scaffold complete and verified. The project is a clean, empty shell
with working utilities. No AI model exists yet.

**What's next (Session 2):** Build `src/data/` — the code that downloads the real
medical data (ClinVar labels + cBioPortal omics) and prepares it for training.

---

*Template for future sessions (copy this when starting a new entry):*

```markdown
### Session N — <short title> — *<date>*

**Goal:** <why this session exists, in one sentence>

**What was created/changed:**
- <file or feature> — <plain-English purpose>

**Commands run this session (and what they did):**
` ` `powershell
<command>   # what it did
` ` `

**Status:** <done / in progress / blocked>

**What's next:** <the next session's goal>
```
