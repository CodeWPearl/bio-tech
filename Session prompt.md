# ═══════════════════════════════════════════════════════════════════════════════
# CLAUDE CODE SESSION PROMPTS
# Deep Learning Framework for Cancer Mutation Pathogenicity Prediction
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW TO USE THIS FILE:
# 1. Each "SESSION" is one Claude Code conversation
# 2. Start a FRESH Claude Code session for each one
# 3. Copy the prompt text between the ``` markers
# 4. Paste into Claude Code
# 5. Review and accept changes
# 6. Run verification commands shown at the end of each session
# 7. Git commit before moving to the next session
#
# ESTIMATED TOTAL TIME: 25-35 sessions across 4-6 weeks
# ═══════════════════════════════════════════════════════════════════════════════


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 1: PROJECT SCAFFOLDING                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
You are setting up a research-grade Python project for predicting pathogenicity
of cancer-associated gene mutations using multi-omics deep learning.

Create the complete project structure:

1. Initialize git repo in the current directory
2. Create the full directory tree:
   - src/ with subpackages: data, features, models (with encoders/ and fusion/
     subpackages), training, evaluation, explainability, uncertainty, utils
   - scripts/, notebooks/, tests/, configs/, data/raw/, data/processed/,
     data/splits/, results/figures/, results/tables/, results/checkpoints/,
     results/logs/, paper/
   - All __init__.py files

3. Create pyproject.toml with:
   - Project name: cancer-mutation-pathogenicity
   - Python >=3.10
   - [project.scripts] entry points for train, evaluate, inference

4. Create requirements.txt with pinned versions:
   torch>=2.1.0
   pytorch-lightning>=2.1.0
   scikit-learn>=1.3.0
   xgboost>=2.0.0
   lightgbm>=4.1.0
   shap>=0.43.0
   lime>=0.2.0
   captum>=0.7.0
   optuna>=3.4.0
   mlflow>=2.9.0
   pandas>=2.1.0
   numpy>=1.26.0
   matplotlib>=3.8.0
   seaborn>=0.13.0
   plotly>=5.18.0
   requests>=2.31.0
   pyyaml>=6.0
   tqdm>=4.66.0
   pytest>=7.4.0
   ruff>=0.1.0
   mypy>=1.7.0

5. Create .gitignore (Python, data files, checkpoints, mlflow, __pycache__,
   .env, *.ckpt, data/raw/*, results/checkpoints/*, *.egg-info)

6. Create configs/default.yaml with placeholder structure:
   data:
     clinvar_url: "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
     cbioportal_url: "https://www.cbioportal.org/api"
     studies: ["brca_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018",
               "coadread_tcga_pan_can_atlas_2018", "ucec_tcga_pan_can_atlas_2018",
               "ov_tcga_pan_can_atlas_2018"]
     test_size: 0.15
     val_size: 0.15
     random_seed: 42
   model:
     mutation_embed_dim: 128
     expression_embed_dim: 256
     methylation_embed_dim: 128
     cnv_embed_dim: 64
     fusion_dim: 256
     num_classes: 4
     dropout: 0.3
     fusion_type: "cross_attention"  # early, late, attention, cross_attention, transformer
   training:
     max_epochs: 100
     batch_size: 64
     learning_rate: 0.001
     weight_decay: 0.0001
     patience: 15
     focal_loss_gamma: 2.0
     num_workers: 4
   experiment:
     name: "default"
     tracking_uri: "mlruns"

7. Create src/utils/config.py that loads YAML configs with OmegaConf-style
   dot-access, CLI override support, and config validation.

8. Create src/utils/reproducibility.py with seed_everything() that sets
   seeds for random, numpy, torch, torch.cuda, and makes cudnn deterministic.

9. Create src/utils/logging_setup.py with setup_logging() that configures
   Python logging with both console and file handlers, timestamps, and
   log level from config.

10. Create a minimal README.md with project title, one-line description,
    and "Setup" section showing pip install commands.

After creating everything, run: python -c "from src.utils.config import load_config; print('OK')"
to verify imports work.
```

# VERIFY: python -c "from src.utils.config import load_config; print('Config OK')"
# VERIFY: python -c "from src.utils.reproducibility import seed_everything; seed_everything(42); print('Seed OK')"
# COMMIT: git add -A && git commit -m "feat: project scaffolding with config, reproducibility, logging"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 2: CLINVAR DATA PIPELINE                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md for project context.

Implement src/data/clinvar_loader.py — the ClinVar data download and processing pipeline.

This module must:

1. Download ClinVar variant_summary.txt.gz from:
   https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
   - Show progress bar with tqdm
   - Skip download if file already exists (check file size)
   - Save to data/raw/variant_summary.txt.gz

2. Parse the TSV file and extract these columns:
   - VariationID, Name, GeneSymbol, GeneID (Entrez)
   - ClinicalSignificance, ClinSigSimple
   - ReviewStatus, NumberSubmitters
   - Type (variant type: single nucleotide variant, deletion, etc.)
   - Chromosome, Start, Stop, Assembly
   - PhenotypeList, Origin

3. Filter to keep ONLY:
   - Assembly == "GRCh38" (never mix genome builds)
   - Origin contains "germline" OR "somatic"
   - ClinicalSignificance is exactly one of:
     "Pathogenic", "Likely pathogenic", "Benign", "Likely benign"
   - Remove "Conflicting interpretations" and "Uncertain significance"
   - ReviewStatus has at least 1 star (not "no assertion criteria provided")
   - GeneSymbol is not empty or "-"
   - Type is "single nucleotide variant" OR "Deletion" OR "Insertion" OR
     "Indel" OR "Duplication" (skip large structural variants)

4. Create a clean label column mapping:
   - "Pathogenic" → 0
   - "Likely pathogenic" → 1
   - "Benign" → 2
   - "Likely benign" → 3

5. Remove duplicate variants (same GeneSymbol + Chromosome + Start + Stop),
   keeping the record with the highest NumberSubmitters.

6. Save processed dataframe to data/processed/clinvar_processed.parquet

7. Print summary statistics:
   - Total variants per class
   - Top 20 genes by variant count
   - Variant type distribution
   - Review status distribution

8. Write comprehensive unit tests in tests/test_data_loading.py:
   - Test download function with mocked HTTP (don't actually download in tests)
   - Test filtering logic with synthetic DataFrame
   - Test label mapping
   - Test deduplication
   - Test edge cases: empty gene symbol, missing columns

Implement the download function, the processing function, and a main()
that runs the full pipeline. All functions must have type hints and docstrings.

Run the tests after implementation.
```

# VERIFY: pytest tests/test_data_loading.py -v
# VERIFY: python -m src.data.clinvar_loader (actually download and process)
# COMMIT: git add -A && git commit -m "feat: ClinVar data download and processing pipeline"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 3: CBIOPORTAL API CLIENT                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/data/clinvar_loader.py for context.

Implement src/data/cbioportal_client.py — a Python client for the cBioPortal REST API.

The cBioPortal API is at https://www.cbioportal.org/api (no authentication needed).
API docs: https://www.cbioportal.org/api/swagger-ui/index.html

This module must provide:

1. class CBioPortalClient:
   - __init__(self, base_url: str = "https://www.cbioportal.org/api")
   - Handles rate limiting (max 10 requests/second, exponential backoff)
   - Session with retry logic (3 retries, backoff_factor=0.5)
   - Proper User-Agent header

2. Methods to fetch data for a given study_id:

   a) get_mutations(study_id) -> pd.DataFrame
      - Endpoint: /molecular-profiles/{study_id}_mutations/mutations
      - Use sampleListId: {study_id}_all
      - projection: "DETAILED"
      - Extract: gene symbol, entrez gene id, mutation type, variant type,
        protein change, chromosome, start position, end position,
        reference allele, variant allele, variant classification
      - Handle pagination (API returns max 10000 per request)

   b) get_expression(study_id) -> pd.DataFrame
      - Endpoint: /molecular-profiles/{study_id}_rna_seq_v2_mrna/molecular-data
      - Returns gene × sample matrix of RSEM normalized expression values
      - Pivot to wide format: rows=samples, columns=genes, values=expression

   c) get_methylation(study_id) -> pd.DataFrame
      - Endpoint: /molecular-profiles/{study_id}_methylation_hm450/molecular-data
      - Returns gene × sample methylation beta values
      - Pivot to wide format

   d) get_cnv(study_id) -> pd.DataFrame
      - Endpoint: /molecular-profiles/{study_id}_gistic/molecular-data
      - Returns GISTIC2 discrete copy number values (-2,-1,0,1,2)
      - Pivot to wide format

   e) get_clinical(study_id) -> pd.DataFrame
      - Endpoint: /studies/{study_id}/clinical-data
      - Extract: patient/sample ID, cancer type, age, sex, stage, grade,
        overall survival months, overall survival status

3. A download_study(study_id, output_dir) method that:
   - Calls all five methods above
   - Saves each as a parquet file in output_dir/{study_id}/
   - Logs progress and data dimensions
   - Returns a dict of DataFrames

4. A download_all_studies(study_ids, output_dir) method that:
   - Iterates through studies, calls download_study for each
   - Handles failures gracefully (log error, continue to next study)
   - Saves a manifest.json listing what was downloaded

IMPORTANT API NOTES:
- For expression and methylation, the API may not return data for all studies.
  Some TCGA studies use different molecular profile IDs. The client should
  try common suffixes: _rna_seq_v2_mrna, _rna_seq_mrna, _mrna for expression.
  For methylation: _methylation_hm450, _methylation_hm27.
  For CNV: _gistic, _cna, _linear_CNA.
- If a profile doesn't exist, catch the 404 and log a warning, don't crash.
- The molecular-data endpoint can be slow for whole-study queries.
  Use fetchAllMolecularDataInMolecularProfileUsingPOST with sampleListId
  for efficiency.

Write tests in tests/test_data_loading.py (append to existing):
- Mock the API responses with unittest.mock.patch on requests.Session.get
- Test retry logic
- Test pagination handling
- Test graceful 404 handling
- Test DataFrame pivoting

Run all tests after implementation.
```

# VERIFY: pytest tests/test_data_loading.py -v
# VERIFY: python -c "from src.data.cbioportal_client import CBioPortalClient; c = CBioPortalClient(); print(c)"
# COMMIT: git add -A && git commit -m "feat: cBioPortal REST API client for multi-omics data"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 4: DATA INTEGRATION & SPLITTING                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md, @src/data/clinvar_loader.py, and @src/data/cbioportal_client.py.

Implement src/data/data_merger.py — the cross-database integration module.

The challenge: ClinVar has pathogenicity LABELS for variants, and cBioPortal
has the multi-omics FEATURES for tumor samples. We need to connect them.

Strategy:
- ClinVar tells us "variant X in gene Y is Pathogenic"
- cBioPortal tells us "sample S has a mutation in gene Y, and here's
  that sample's expression/methylation/CNV/clinical data"
- We JOIN on: gene symbol + variant position + variant type

Implementation:

1. class DataMerger:
   - merge_clinvar_with_mutations(clinvar_df, mutations_df) -> pd.DataFrame
     * Join on: GeneSymbol (exact match) AND Chromosome + Start position
       (fuzzy match within 5bp window to handle different representations)
     * For each matched mutation, we now have: the mutation details,
       the pathogenicity label from ClinVar, and the sample ID
     * Log: how many mutations matched, how many didn't, match rate

   - attach_omics_features(merged_df, expression_df, methylation_df,
                           cnv_df, clinical_df) -> pd.DataFrame
     * For each sample in merged_df, look up its expression/methylation/
       CNV/clinical data
     * Handle missing data: some samples won't have all omics types.
       Add boolean columns: has_expression, has_methylation, has_cnv
     * For expression and methylation: only keep the TOP 2000 most
       variable genes (by variance across samples) to manage dimensionality

2. class DataSplitter:
   - split_by_gene(df, test_size, val_size, random_seed) -> dict
     * CRITICAL: Split by GENE, not by individual variant
     * This prevents data leakage (variants in the same gene are correlated)
     * Stratify by label distribution
     * Return {"train": df, "val": df, "test": df}
     * Save splits to data/splits/ as parquet files
     * Save split gene lists to data/splits/gene_splits.json

   - print_split_statistics(splits_dict)
     * Label distribution per split
     * Number of unique genes per split
     * Number of samples per split
     * Verify no gene overlap between splits

3. A run_full_pipeline() function that:
   - Loads ClinVar processed data
   - Loads cBioPortal data for configured studies
   - Merges everything
   - Creates splits
   - Saves everything
   - Prints comprehensive statistics

4. Also create scripts/download_data.py that:
   - Reads config from configs/default.yaml
   - Runs ClinVar download + processing
   - Runs cBioPortal download for all configured studies
   - Runs the merger
   - Creates splits
   - Total end-to-end data pipeline in one script

Write tests for:
- Merge logic with synthetic data (including edge cases)
- Split-by-gene correctness (verify no gene leakage)
- Statistics calculation
- Missing data handling

Run tests after implementation.
```

# VERIFY: pytest tests/test_data_loading.py -v
# VERIFY: python scripts/download_data.py --config configs/default.yaml (run actual pipeline)
# COMMIT: git add -A && git commit -m "feat: data integration, gene-stratified splitting, download pipeline"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 5: FEATURE ENGINEERING                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/data/data_merger.py to understand the merged data format.

Implement ALL feature extraction modules in src/features/.

Each module must:
- Have a class with fit() and transform() methods (sklearn-style)
- Handle missing values explicitly
- Include type hints and docstrings
- Be independently testable

1. src/features/mutation_features.py — MutationFeatureExtractor:
   Input columns: GeneSymbol, VariantClassification, ProteinChange,
                  Chromosome, StartPosition, VariantType, ReferenceAllele,
                  VariantAllele

   Features to extract:
   a) Variant type one-hot: Missense, Nonsense, Frameshift, Splice_Site,
      In_Frame_Del, In_Frame_Ins, Silent, Other
   b) Amino acid change properties (if protein change available):
      - Grantham score (biochemical distance between amino acids)
      - BLOSUM62 score
      - Hydrophobicity change
      - Charge change
      - Size change
      Include a lookup table of amino acid properties.
   c) Gene-level features:
      - Gene length (use a bundled gene_info dict or fetch from Entrez)
      - Is the gene in COSMIC Cancer Gene Census? (boolean)
      - Mutation frequency in ClinVar for this gene (count from our data)
   d) Positional features:
      - Chromosome one-hot (22 autosomes + X + Y)
      - Normalized position within gene (0.0 to 1.0)

   Output: numpy array of shape (n_samples, n_mutation_features)
   Also return feature_names list for interpretability.

2. src/features/expression_features.py — ExpressionFeatureExtractor:
   Input: expression DataFrame (samples × top-2000 genes, RSEM values)

   Processing:
   a) Log2(x + 1) transform
   b) Quantile normalization across samples
   c) Z-score standardization per gene
   d) Handle missing values: impute with gene median

   Output: numpy array of shape (n_samples, 2000)

3. src/features/methylation_features.py — MethylationFeatureExtractor:
   Input: methylation DataFrame (samples × top-2000 genes, beta values)

   Processing:
   a) Beta values are already 0-1, but clip to [0.001, 0.999]
   b) M-value transform: M = log2(beta / (1 - beta)) for statistical analysis
   c) Z-score standardization per gene
   d) Handle missing: impute with gene median

   Output: numpy array of shape (n_samples, 2000)

4. src/features/cnv_features.py — CNVFeatureExtractor:
   Input: CNV DataFrame (samples × genes, GISTIC values: -2,-1,0,1,2)

   Processing:
   a) Keep discrete values (they're already categorical)
   b) One-hot encode each gene's CNV state (5 categories)
   c) OR: keep as ordinal features (just the -2 to 2 values)
   d) Focus on genes that overlap with our mutation gene set

   Output: numpy array of shape (n_samples, n_cnv_features)

5. src/features/clinical_features.py — ClinicalFeatureExtractor:
   Input: clinical DataFrame

   Processing:
   a) Age: normalize to [0, 1] range
   b) Sex: binary encode
   c) Cancer type: one-hot encode
   d) Stage: ordinal encode (I=1, II=2, III=3, IV=4)
   e) Handle missing: impute age with median, sex with mode, stage with mode

   Output: numpy array of shape (n_samples, n_clinical_features)

6. Create a FeaturePipeline class in src/features/__init__.py that:
   - Orchestrates all extractors
   - Runs fit() on training data, transform() on all splits
   - Saves fitted extractors (pickle) for inference
   - Returns a dict of feature arrays keyed by modality
   - Also returns a combined feature matrix for baseline models

Write tests in tests/test_features.py covering each extractor with synthetic data.

Run all tests.
```

# VERIFY: pytest tests/test_features.py -v
# COMMIT: git add -A && git commit -m "feat: multi-omics feature extraction pipeline"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 6: PYTORCH DATASET & DATAMODULE                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md, @src/features/__init__.py, and @configs/default.yaml.

Implement PyTorch data loading infrastructure.

1. src/data/dataset.py — MultiOmicsDataset(torch.utils.data.Dataset):
   - __init__(self, mutation_features, expression_features,
              methylation_features, cnv_features, clinical_features,
              labels, modality_mask)
   - Each feature input is a numpy array or None
   - modality_mask: boolean array [has_expression, has_methylation, has_cnv]
     per sample — critical for handling missing modalities
   - __getitem__ returns a dict:
     {
       "mutation": torch.FloatTensor,
       "expression": torch.FloatTensor (zeros if missing),
       "methylation": torch.FloatTensor (zeros if missing),
       "cnv": torch.FloatTensor (zeros if missing),
       "clinical": torch.FloatTensor,
       "modality_mask": torch.BoolTensor (which modalities are present),
       "label": torch.LongTensor
     }
   - __len__ returns number of samples
   - custom collate_fn that handles variable-presence modalities

2. src/data/datamodule.py — PathogenicityDataModule(LightningDataModule):
   - __init__(self, config: DictConfig)
   - prepare_data(): download if not exists (calls download pipeline)
   - setup(stage): load splits, run feature pipeline, create Dataset objects
   - train_dataloader(): DataLoader with shuffle=True, class-weighted sampler
   - val_dataloader(): DataLoader with shuffle=False
   - test_dataloader(): DataLoader with shuffle=False
   - The class-weighted sampler uses inverse class frequency weights
     to handle imbalanced labels

3. Make sure the DataModule:
   - Caches processed features after first run (don't re-extract every time)
   - Logs dataset statistics on setup
   - Is fully configurable via the YAML config

Write tests in tests/test_data_loading.py (append):
- Test Dataset __getitem__ with synthetic data
- Test collate_fn produces correct batch shapes
- Test modality masking (missing modalities → zeros + mask=False)
- Test DataModule setup with a tiny synthetic dataset
- Test class-weighted sampler produces balanced batches

Run all tests.
```

# VERIFY: pytest tests/test_data_loading.py -v
# COMMIT: git add -A && git commit -m "feat: PyTorch Dataset and Lightning DataModule"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 7: MODALITY ENCODERS                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/data/dataset.py to understand input shapes.

Implement all modality encoders in src/models/encoders/.

First create src/models/base.py:
- class BaseModel(nn.Module): abstract base with encode() and forward()
- Common methods: count_parameters(), get_device()

Then implement each encoder:

1. src/models/encoders/mutation_encoder.py — MutationEncoder(BaseModel):
   Architecture: MLP with learned embeddings
   - Input: mutation feature vector (n_mutation_features,)
   - Linear(n_features, 256) → BatchNorm → ReLU → Dropout(0.3)
   - Linear(256, 128) → BatchNorm → ReLU → Dropout(0.2)
   - Output: embedding of shape (mutation_embed_dim,) — default 128

   Also implement a SECOND version using a small Transformer:
   - Embed each feature group (variant type, AA properties, gene features,
     position) as separate tokens
   - 2-layer Transformer encoder with 4 heads
   - [CLS] token output as the embedding
   Name it MutationTransformerEncoder.

2. src/models/encoders/expression_encoder.py — ExpressionEncoder(BaseModel):
   Implement THREE versions:

   a) DenseAutoencoder:
      Encoder: 2000 → 512 → 256
      Decoder: 256 → 512 → 2000 (for pretraining)
      During pathogenicity prediction, use encoder only.

   b) VariationalAutoencoder (VAE):
      Encoder: 2000 → 512 → (mu: 256, logvar: 256)
      Reparameterization trick
      Decoder: 256 → 512 → 2000
      Returns: (z, mu, logvar) for KL loss

   c) TransformerEncoder:
      Chunk 2000 genes into 40 groups of 50
      Project each group to embed_dim
      4-layer Transformer with 8 heads
      [CLS] token → 256-dim output

3. src/models/encoders/methylation_encoder.py — MethylationEncoder(BaseModel):
   Same architecture options as expression encoder but with:
   - Input dim matched to methylation features
   - Separate batch normalization statistics

4. src/models/encoders/cnv_encoder.py — CNVEncoder(BaseModel):
   a) FCEncoder: simple MLP
      Input → 128 → 64 → cnv_embed_dim
   b) AttentionEncoder:
      Treat each gene's CNV as a token
      Self-attention to learn gene-gene CNV interactions
      Mean pooling → cnv_embed_dim

Each encoder must:
- Accept embed_dim as a constructor parameter
- Have a get_output_dim() method returning the embedding dimension
- Handle variable input sizes gracefully
- Include dropout for regularization

Write tests in tests/test_encoders.py:
- For each encoder: verify output shape with random input
- Test with batch_size=1 and batch_size=32
- Test that parameters are trainable (gradient flows)
- Test get_output_dim() matches actual output

Run all tests.
```

# VERIFY: pytest tests/test_encoders.py -v
# COMMIT: git add -A && git commit -m "feat: modality encoders (MLP, Autoencoder, VAE, Transformer)"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 8: FUSION STRATEGIES                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/models/encoders/ to understand encoder output shapes.

Implement ALL five fusion strategies in src/models/fusion/.

Every fusion module must:
- Inherit from nn.Module
- Accept a modality_mask tensor (batch, num_modalities) to handle
  missing modalities (mask out absent modalities in attention/pooling)
- Accept a dict of embeddings: {"mutation": Tensor, "expression": Tensor,
  "methylation": Tensor, "cnv": Tensor, "clinical": Tensor}
- Output a fused representation of shape (batch, fusion_dim)
- Include type hints and docstrings

1. src/models/fusion/early_fusion.py — EarlyFusion:
   - Concatenate all embeddings along feature dimension
   - Apply modality mask: zero out missing modality embeddings before concat
   - Linear projection: concat_dim → fusion_dim
   - BatchNorm → ReLU → Dropout
   Simple but strong baseline.

2. src/models/fusion/late_fusion.py — LateFusion:
   - Each modality gets its own classification head: embed_dim → num_classes
   - Final prediction: weighted average of per-modality logits
   - Learnable weights per modality (softmax-normalized)
   - Modality mask: exclude missing modalities from the weighted average
   - Also output intermediate per-modality predictions for interpretability

3. src/models/fusion/attention_fusion.py — AttentionFusion:
   - Stack embeddings as a sequence: (batch, num_modalities, embed_dim)
     All embeddings must be projected to the same dimension first.
   - Multi-head self-attention (4 heads) over modalities
   - Apply modality mask in attention: -inf for missing modalities
   - Weighted pooling using attention scores
   - Layer normalization + residual connection
   - Output: (batch, fusion_dim)
   - Save attention weights for interpretability (which omics matters most)

4. src/models/fusion/cross_attention.py — CrossAttentionFusion:
   - Each modality attends to every other modality (pairwise cross-attention)
   - For modality i, Q comes from modality i, K and V from all modalities
   - Stack cross-attended outputs
   - Mean pooling → fusion_dim
   - Modality mask applied in K/V attention masking
   This is the most expressive fusion — the paper's PRIMARY contribution.

5. src/models/fusion/transformer_fusion.py — TransformerFusion:
   - Add learned modality-type embeddings to each modality embedding
   - Add a [CLS] token
   - 2-layer Transformer encoder over the modality sequence
   - [CLS] output → fusion_dim
   - Modality mask as padding mask in Transformer

Write tests in tests/test_fusion.py:
- Each fusion: correct output shape
- Each fusion: works with missing modalities (2 of 5 absent)
- Each fusion: gradient flows through all paths
- Attention-based fusions: attention weights sum to ~1.0

Run all tests.
```

# VERIFY: pytest tests/test_fusion.py -v
# COMMIT: git add -A && git commit -m "feat: five fusion strategies (early, late, attention, cross-attention, transformer)"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 9: FULL MODEL + CLASSIFICATION HEAD                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md, @src/models/encoders/, and @src/models/fusion/.

Implement the complete assembled model.

1. src/models/classifier.py — ClassificationHead(nn.Module):
   - Input: fused representation (batch, fusion_dim)
   - Architecture:
     Linear(fusion_dim, 128) → BatchNorm → ReLU → Dropout(0.3)
     Linear(128, 64) → BatchNorm → ReLU → Dropout(0.2)
     Linear(64, num_classes)
   - Output: logits (batch, num_classes)
   - Method: predict_proba() that applies softmax
   - Method: predict() that returns argmax class

2. src/models/full_model.py — PathogenicityPredictor(BaseModel):
   - __init__(self, config):
     * Instantiate all encoders based on config
     * Instantiate chosen fusion module based on config.model.fusion_type
     * Instantiate classification head
   - forward(self, batch: dict) -> dict:
     * Extract features from batch dict
     * Run each modality through its encoder
     * Get modality_mask from batch
     * Run fusion
     * Run classifier
     * Return {
         "logits": Tensor (batch, num_classes),
         "probabilities": Tensor (batch, num_classes),
         "predicted_class": Tensor (batch,),
         "fused_embedding": Tensor (batch, fusion_dim),
         "modality_embeddings": dict of per-modality embeddings,
         "attention_weights": Tensor or None (from fusion, for interpretability)
       }
   - A from_config(cls, config) classmethod for clean instantiation
   - A summary() method printing parameter counts per component

3. Verify the full model works end-to-end:
   - Create a synthetic batch matching the Dataset format
   - Forward pass through the model
   - Backward pass (compute loss, call .backward())
   - Check gradients exist on all parameters
   - Print model summary

Write tests in tests/test_models.py:
- Full model forward pass with synthetic batch
- Full model backward pass (gradient check)
- Model with different fusion types (test all five)
- Model with missing modalities in the batch
- Parameter count is reasonable (not exploding)
- Test from_config() class method

Run all tests.
```

# VERIFY: pytest tests/ -v  (run ALL tests)
# COMMIT: git add -A && git commit -m "feat: full model assembly with classifier head"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 10: TRAINING PIPELINE                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md, @src/models/full_model.py, @src/data/datamodule.py, and
@configs/default.yaml.

Implement the complete training infrastructure.

1. src/training/losses.py:
   - FocalLoss(nn.Module): implements focal loss for class imbalance
     FL(p) = -alpha * (1-p)^gamma * log(p)
     Parameters: alpha (per-class weights), gamma (focusing parameter)
   - WeightedCrossEntropy: wrapper around nn.CrossEntropyLoss with
     automatic class weight computation from label distribution

2. src/training/scheduler.py:
   - get_scheduler(optimizer, config) function that returns:
     CosineAnnealingWarmRestarts with configurable warmup steps
   - Also support: ReduceLROnPlateau, OneCycleLR via config

3. src/training/callbacks.py:
   - MetricLogger callback: logs all metrics to MLflow at each epoch
   - GradientMonitor callback: logs gradient norms to detect exploding/vanishing
   - EarlyStoppingWithPatience: custom early stopping on val_auroc

4. src/training/lightning_module.py — PathogenicityLightningModule(LightningModule):
   - __init__(self, config, model, class_weights):
     * Wraps PathogenicityPredictor
     * Sets up loss function based on config (focal, weighted_ce, or ce)
     * Initializes torchmetrics: Accuracy, F1, AUROC, Precision, Recall
       for both per-step and per-epoch tracking
   - training_step(batch, batch_idx):
     * Forward pass
     * Compute loss
     * Log: train_loss, train_accuracy (per step)
     * Return loss
   - validation_step(batch, batch_idx):
     * Forward pass
     * Compute loss + all metrics
     * Accumulate predictions for epoch-level AUROC
   - on_validation_epoch_end():
     * Compute epoch-level: val_loss, val_accuracy, val_auroc, val_f1,
       val_precision, val_recall, val_mcc
     * Log all to MLflow
   - test_step: same as validation_step but with test_ prefix
   - configure_optimizers:
     * AdamW with weight_decay from config
     * LR scheduler from config

5. scripts/train.py — Main training entry point:
   - Parse CLI args: --config, --experiment_name, --gpus
   - Load config, merge CLI overrides
   - Set seeds
   - Initialize: DataModule, Model, LightningModule
   - Initialize: MLflow tracking
   - Initialize: Trainer with callbacks:
     * ModelCheckpoint(monitor="val_auroc", mode="max", save_top_k=3)
     * EarlyStopping(monitor="val_auroc", patience=config.training.patience)
     * LearningRateMonitor
     * GradientMonitor
   - trainer.fit()
   - trainer.test(ckpt_path="best")
   - Log final results to MLflow
   - Save final metrics to results/tables/

Write tests in tests/test_training.py:
- FocalLoss: correct gradient with known inputs
- Lightning module: training_step runs without error on synthetic batch
- Lightning module: validation metrics are computed correctly
- Full training loop: 2 epochs on tiny synthetic data completes without error

Run all tests.
```

# VERIFY: pytest tests/test_training.py -v
# VERIFY: python scripts/train.py --config configs/default.yaml  (on real data if downloaded)
# COMMIT: git add -A && git commit -m "feat: complete training pipeline with Lightning, MLflow, focal loss"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 11: EVALUATION METRICS                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/training/lightning_module.py.

Implement comprehensive evaluation.

1. src/evaluation/metrics.py — compute_all_metrics(y_true, y_pred, y_prob):
   - Input: true labels, predicted labels, prediction probabilities (n, 4)
   - Compute:
     * Overall accuracy
     * Per-class precision, recall, F1
     * Macro-averaged and weighted-averaged precision, recall, F1
     * Matthews Correlation Coefficient (MCC)
     * ROC-AUC (one-vs-rest, macro average)
     * PR-AUC (one-vs-rest, macro average)
     * Cohen's Kappa
     * Top-1 and Top-2 accuracy
     * Expected Calibration Error (ECE) with 10 bins
   - Return as a dict with descriptive keys
   - Also: confusion_matrix(), classification_report_df()
   - Bootstrap confidence intervals: compute_ci(y_true, y_pred, y_prob,
     n_bootstrap=1000, ci=0.95) returning (lower, mean, upper) for each metric

2. src/evaluation/benchmarks.py — run_baselines(X_train, y_train, X_test, y_test):
   Implement and evaluate these baseline models:
   a) Logistic Regression (sklearn)
   b) Random Forest (sklearn, n_estimators=500)
   c) XGBoost (xgboost)
   d) LightGBM (lightgbm)
   e) Multi-Layer Perceptron (sklearn MLPClassifier)

   For each baseline:
   - Use the FLATTENED combined feature vector (all modalities concatenated)
   - 5-fold cross-validation on training set for hyperparameter selection
   - Evaluate on test set
   - Compute all metrics from compute_all_metrics()
   - Return a DataFrame comparing all baselines + our model

3. src/evaluation/biological_validation.py:
   - Load COSMIC Cancer Gene Census list
   - For variants in known cancer driver genes:
     * What fraction are correctly predicted as Pathogenic/Likely Pathogenic?
   - For variants in non-cancer genes:
     * What fraction are correctly predicted as Benign/Likely Benign?
   - Cross-reference with ClinVar review stars:
     * Is our model more confident (higher probability) for higher-star variants?
   - Gene-level analysis: for each gene, compute the model's accuracy
   - Report: precision, recall, F1 specifically for known cancer driver mutations

4. scripts/evaluate.py:
   - Load best checkpoint
   - Run on test set
   - Compute all metrics with confidence intervals
   - Run baseline comparison
   - Run biological validation
   - Save all results to results/tables/ as CSV
   - Print formatted summary

Write tests for metrics (verify against sklearn on known inputs) and
biological validation logic.

Run all tests.
```

# VERIFY: pytest tests/test_evaluation.py -v
# COMMIT: git add -A && git commit -m "feat: comprehensive evaluation with baselines and biological validation"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 12: EXPLAINABILITY                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/models/full_model.py.

Implement ALL explainability methods.

1. src/explainability/shap_explainer.py — SHAPExplainer:
   - Uses SHAP DeepExplainer or KernelExplainer on our model
   - compute_global_importance(model, test_data, n_samples=500):
     * Compute SHAP values for a sample of test instances
     * Aggregate to get global feature importance
     * Group by modality (which omics contributes most?)
     * Return: per-feature importance, per-modality importance
   - compute_local_explanation(model, single_sample):
     * SHAP values for one specific variant prediction
     * Return: feature attribution dict
   - generate_shap_plots(shap_values, feature_names, output_dir):
     * Summary plot (beeswarm)
     * Bar plot (top 30 features)
     * Modality importance bar chart

2. src/explainability/integrated_gradients.py — IGExplainer:
   - Uses captum.attr.IntegratedGradients
   - compute_attributions(model, batch, target_class):
     * Compute IG attributions for each input feature
     * Aggregate by modality
   - compute_modality_importance(model, test_loader):
     * Average IG attributions across test set per modality
     * Return ranked modality importance

3. src/explainability/attention_viz.py — AttentionVisualizer:
   - Extract attention weights from the fusion module
   - For attention/cross-attention/transformer fusion:
     * Get attention weight matrix (batch, heads, n_modalities, n_modalities)
     * Average across heads
     * Create heatmap: which modality attends to which
   - plot_attention_heatmap(attention_weights, modality_names, output_path)
   - plot_attention_distribution(attention_weights_across_test_set, output_path):
     * Box plot of attention weights per modality across all test samples

4. src/explainability/lime_explainer.py — LIMEExplainer:
   - Use lime.lime_tabular.LimeTabularExplainer
   - Wrapper that converts our model to a function LIME can call
   - explain_instance(model, sample, feature_names):
     * Return top contributing features for/against each class

5. Scripts integration in scripts/generate_figures.py (stub — flesh out later):
   - Run all explainability methods on test set
   - Save all plots to results/figures/

Write tests:
- SHAP: runs on tiny model without crashing
- IG: produces attributions with correct shape
- Attention viz: produces valid heatmap data
- LIME: produces explanation for single sample

Run tests.
```

# VERIFY: pytest tests/ -v --ignore=tests/test_training.py (skip slow training tests)
# COMMIT: git add -A && git commit -m "feat: explainability (SHAP, IG, attention viz, LIME)"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 13: UNCERTAINTY ESTIMATION                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/models/full_model.py.

Implement uncertainty estimation methods.

1. src/uncertainty/mc_dropout.py — MCDropoutPredictor:
   - __init__(self, model, n_forward_passes=50)
   - predict_with_uncertainty(batch):
     * Enable dropout at inference time (model.train() mode)
     * Run n_forward_passes forward passes
     * Collect all softmax outputs
     * Compute: mean prediction, predictive variance, entropy
     * Return: {
         "mean_probs": Tensor (batch, num_classes),
         "predicted_class": Tensor (batch,),
         "epistemic_uncertainty": Tensor (batch,) — variance of predictions,
         "predictive_entropy": Tensor (batch,) — entropy of mean prediction,
         "all_predictions": Tensor (n_passes, batch, num_classes)
       }

2. src/uncertainty/deep_ensembles.py — DeepEnsemblePredictor:
   - __init__(self, model_paths: list[str], config)
   - Loads n models (n=5 by default, each trained with different seed)
   - predict_with_uncertainty(batch):
     * Forward pass through each ensemble member
     * Compute: mean prediction, variance, mutual information
     * Return same format as MC Dropout

3. src/uncertainty/calibration.py:
   - TemperatureScaling:
     * Learn a temperature parameter T on validation set
     * Calibrated probs = softmax(logits / T)
     * optimize_temperature(logits, labels) using NLL minimization
   - compute_ece(y_true, y_prob, n_bins=15):
     * Expected Calibration Error
   - compute_reliability_diagram(y_true, y_prob, n_bins=15):
     * Return (mean_predicted_prob, true_fraction, bin_counts) for plotting
   - apply_calibration(model, val_loader):
     * Fit temperature on val set
     * Return calibrated model wrapper

4. Update scripts/evaluate.py to also:
   - Run MC Dropout uncertainty estimation on test set
   - Compute ECE before and after temperature scaling
   - Report uncertainty statistics (mean, std of uncertainty per class)
   - Flag high-uncertainty predictions for manual review
   - Save uncertainty-augmented predictions to results/tables/

5. Update scripts/inference.py:
   - Single variant prediction endpoint
   - Input: variant information (gene, mutation type, position, etc.)
   - Output: {
       "predicted_class": "Pathogenic",
       "confidence": 0.87,
       "uncertainty": 0.05,
       "calibrated_probability": [0.87, 0.08, 0.03, 0.02],
       "explanation": "Top contributing features: ...",
       "recommendation": "High confidence prediction" | "Low confidence — manual review recommended"
     }

Write tests for MC Dropout and calibration.

Run all tests.
```

# VERIFY: pytest tests/ -v
# COMMIT: git add -A && git commit -m "feat: uncertainty estimation (MC Dropout, ensembles, calibration)"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 14: ABLATION STUDY                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @src/models/full_model.py.

Implement the ablation study framework.

1. Create configs/ablation/ directory with YAML configs for each ablation:
   - no_mutation.yaml: disable mutation encoder (zero out mutation embeddings)
   - no_expression.yaml: disable expression encoder
   - no_methylation.yaml: disable methylation encoder
   - no_cnv.yaml: disable CNV encoder
   - no_attention.yaml: use early_fusion instead of cross_attention
   - single_mutation_only.yaml: only mutation features, no other omics
   - single_expression_only.yaml: only expression features
   - no_focal_loss.yaml: use standard cross_entropy instead of focal loss

2. Modify PathogenicityPredictor to support modality ablation:
   - Add a disabled_modalities parameter
   - When a modality is disabled, replace its embedding with a zero vector
     AND set its modality_mask to False (so fusion knows it's absent)
   - This must be controlled via config, not code changes

3. scripts/run_ablation.py:
   - Iterate through all ablation configs
   - For each: train the model (same seeds, same data splits)
   - Evaluate on test set
   - Collect all metrics into a comparison DataFrame
   - Compute: performance drop (%) relative to full model for each metric
   - Save to results/tables/ablation_results.csv
   - Print formatted comparison table

4. The ablation table should have columns:
   Configuration | Accuracy | F1-Macro | AUROC | PR-AUC | MCC | Δ vs Full
   Full Model    | 0.XX     | 0.XX     | 0.XX  | 0.XX   | 0.XX| —
   No Mutation   | 0.XX     | 0.XX     | 0.XX  | 0.XX   | 0.XX| -X.X%
   No Expression | ...
   etc.

This is a critical section for the paper — it demonstrates the value of
each component.

Run the ablation study (this takes time — can be done overnight).
```

# VERIFY: python scripts/run_ablation.py --config configs/default.yaml
# COMMIT: git add -A && git commit -m "feat: ablation study framework with automated comparison"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 15: PUBLICATION FIGURES                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md.

Implement scripts/generate_figures.py — generate ALL publication-quality figures.

Use matplotlib + seaborn with these settings for journal quality:
- Figure DPI: 300
- Font: Arial/Helvetica (or DejaVu Sans as fallback)
- Font size: 10pt for labels, 8pt for tick labels
- Save as both PDF (for LaTeX) and PNG (for preview)
- Use colorblind-friendly palette (e.g., seaborn "colorblind" or "Set2")

Generate these figures:

1. Figure 1: Model Architecture Diagram
   - Schematic showing: inputs → encoders → fusion → classifier → output
   - Use matplotlib patches and arrows
   - Show dimensionality at each stage

2. Figure 2: Dataset Statistics (multi-panel)
   - (a) Class distribution bar chart
   - (b) Top 20 genes by variant count
   - (c) Variant type distribution
   - (d) Data availability Venn diagram (which samples have which omics)

3. Figure 3: Learning Curves
   - (a) Training loss vs. validation loss over epochs
   - (b) Validation AUROC over epochs
   (Load from MLflow or training logs)

4. Figure 4: ROC Curves (multi-class)
   - One-vs-rest ROC for each class
   - Micro-average and macro-average ROC
   - Include AUROC values in legend

5. Figure 5: PR Curves (multi-class)
   - Same format as ROC curves
   - Include AP values in legend

6. Figure 6: Confusion Matrix
   - Heatmap with counts AND percentages
   - Proper class labels on axes

7. Figure 7: Baseline Comparison
   - Grouped bar chart: model vs all baselines on key metrics
   - Include 95% CI error bars

8. Figure 8: Ablation Study Results
   - Horizontal bar chart showing performance drop per ablation
   - Ordered by impact magnitude

9. Figure 9: SHAP Analysis (multi-panel)
   - (a) Global feature importance (top 30)
   - (b) Modality importance comparison
   - (c) SHAP beeswarm for top features

10. Figure 10: Attention Weights
    - Heatmap showing cross-modality attention patterns
    - Average across test set

11. Figure 11: Uncertainty Analysis (multi-panel)
    - (a) Calibration plot (before and after temperature scaling)
    - (b) Uncertainty distribution by prediction correctness
    - (c) Accuracy vs. confidence histogram

12. Figure 12: Biological Validation
    - Agreement rate with ClinVar review stars
    - Performance on known driver genes vs. non-driver genes

Save all to results/figures/ with descriptive filenames.
Print a summary of all generated figures.
```

# VERIFY: python scripts/generate_figures.py
# VERIFY: ls results/figures/ (check all files exist)
# COMMIT: git add -A && git commit -m "feat: publication-quality figures for all analyses"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 16: HYPERPARAMETER OPTIMIZATION                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and @scripts/train.py.

Implement hyperparameter optimization with Optuna.

1. Create scripts/run_hpo.py:
   - Define an Optuna objective function that:
     * Suggests hyperparameters:
       - learning_rate: log-uniform [1e-5, 1e-2]
       - batch_size: categorical [32, 64, 128]
       - dropout: uniform [0.1, 0.5]
       - fusion_type: categorical ["early", "attention", "cross_attention", "transformer"]
       - mutation_embed_dim: categorical [64, 128, 256]
       - expression_embed_dim: categorical [128, 256, 512]
       - focal_loss_gamma: uniform [0.5, 5.0]
       - weight_decay: log-uniform [1e-6, 1e-2]
       - num_attention_heads: categorical [2, 4, 8]
     * Trains the model for max 30 epochs (with early stopping)
     * Returns validation AUROC (maximize)
   - Uses Optuna's TPE sampler with 50-100 trials
   - Pruning: MedianPruner to stop unpromising trials early
   - Save study results to results/hpo_study.db (SQLite)
   - Generate: parameter importance plot, optimization history, parallel coordinate plot
   - Save best config to configs/best.yaml

2. Integrate with MLflow: log each trial as an MLflow run with
   hyperparameters as parameters and final val_auroc as metric.

3. After HPO, retrain final model with best config on full train+val data,
   evaluate on test set.

Make this configurable via configs/sweep.yaml.
```

# VERIFY: python scripts/run_hpo.py --n_trials 5 (quick test with few trials)
# COMMIT: git add -A && git commit -m "feat: Optuna hyperparameter optimization with MLflow integration"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 17: REPRODUCIBILITY & DOCKER                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md.

Set up complete reproducibility infrastructure.

1. Create Dockerfile:
   - Base: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   - Install all requirements
   - Copy source code
   - Set working directory
   - Entry point: scripts/train.py
   - Also support: scripts/evaluate.py, scripts/inference.py

2. Create docker-compose.yml:
   - Service for training (with GPU support)
   - Service for MLflow UI (port 5000)
   - Volume mounts for data/ and results/

3. Create Makefile with targets:
   - make setup: create venv, install requirements
   - make download: run download_data.py
   - make train: run train.py with default config
   - make evaluate: run evaluate.py
   - make ablation: run run_ablation.py
   - make figures: run generate_figures.py
   - make test: run pytest
   - make lint: run ruff + mypy
   - make all: download → train → evaluate → ablation → figures
   - make docker-build: build Docker image
   - make docker-train: train in Docker

4. Update README.md with COMPLETE documentation:
   - Project title and abstract
   - Architecture diagram (reference to Figure 1)
   - Installation (pip and Docker)
   - Data download instructions
   - Training instructions (with expected output)
   - Evaluation instructions
   - Inference usage example
   - Project structure explanation
   - Citation (BibTeX)
   - License (MIT)

5. Create scripts/verify_reproducibility.py:
   - Train the model twice with the same config and seed
   - Compare outputs: should be identical (or very close with GPU non-determinism)
   - Report max absolute difference in predictions

Make everything robust and well-documented.
```

# VERIFY: make test && make lint
# COMMIT: git add -A && git commit -m "feat: Docker, Makefile, full reproducibility infrastructure"


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SESSION 18: PAPER SCAFFOLD                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

```
Read @CLAUDE.md and review results in @results/tables/ and @results/figures/.

Create the research paper scaffold in paper/.

1. paper/manuscript.tex — LaTeX manuscript following IEEE/Nature format:
   - Title: "Multi-Omics Deep Learning Framework with Cross-Attention Fusion
     for Predicting Pathogenicity of Cancer-Associated Gene Mutations"
   - Sections (with substantive drafts, not just headers):
     * Abstract (250 words)
     * Introduction (motivation, gap, contribution)
     * Related Work (survey existing pathogenicity predictors)
     * Methods
       - Data collection and preprocessing
       - Feature extraction (each modality)
       - Model architecture (with figure reference)
       - Fusion strategies
       - Training procedure
       - Evaluation metrics
       - Explainability and uncertainty
     * Experiments
       - Dataset statistics
       - Implementation details
       - Main results (table of metrics)
       - Baseline comparison
       - Ablation study
       - Fusion strategy comparison
     * Results and Discussion
       - Performance analysis
       - Explainability insights
       - Uncertainty analysis
       - Biological validation
     * Limitations
     * Conclusion and Future Work
     * References

2. paper/references.bib — BibTeX with references for:
   - ClinVar, COSMIC, cBioPortal, TCGA
   - PyTorch, PyTorch Lightning
   - SHAP, LIME, Integrated Gradients
   - Focal Loss (Lin et al.)
   - Attention mechanisms (Vaswani et al.)
   - Related pathogenicity predictors (SIFT, PolyPhen-2, CADD, REVEL, AlphaMissense)
   - Multi-omics integration methods

3. Include \input{} references to figure and table files.

The paper should be a complete first draft that can be iterated on.
```

# VERIFY: Check paper/manuscript.tex compiles (if LaTeX installed)
# COMMIT: git add -A && git commit -m "docs: complete research paper first draft"


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERIFICATION CHECKLIST
# ═══════════════════════════════════════════════════════════════════════════════
#
# Before delivering to client, verify:
#
# □ make download  — data downloads successfully
# □ make train     — training completes without errors
# □ make evaluate  — evaluation produces all metrics
# □ make ablation  — ablation study runs all configurations
# □ make figures   — all 12+ figures generated
# □ make test      — all tests pass
# □ make lint      — no ruff or mypy errors
# □ Dockerfile builds and runs
# □ README.md is complete with usage instructions
# □ results/tables/ has CSV files with all metric tables
# □ results/figures/ has all publication figures
# □ paper/manuscript.tex is a complete first draft
# □ Git history is clean with conventional commits
# □ No API keys or credentials in the codebase
# □ No data files committed to git (only in .gitignore'd directories)
# ═══════════════════════════════════════════════════════════════════════════════