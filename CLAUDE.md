# Cancer Mutation Pathogenicity Predictor

## Project Overview
Research-grade deep learning framework predicting pathogenicity of cancer-associated
gene mutations (Pathogenic / Likely Pathogenic / Benign / Likely Benign) using
multi-omics integration. Target: IEEE/Springer/Nature journal submission.

## Tech Stack
- Python 3.10+, PyTorch 2.x, PyTorch Lightning 2.x
- scikit-learn, XGBoost, LightGBM (baselines)
- SHAP, LIME, captum (explainability)
- Optuna (hyperparameter optimization), MLflow (experiment tracking)
- pandas, numpy, matplotlib, seaborn, plotly

## Build Commands
- Install: `pip install -r requirements.txt`
- Train: `python scripts/train.py --config configs/default.yaml`
- Evaluate: `python scripts/evaluate.py --checkpoint results/best_model.ckpt`
- Test: `pytest tests/ -v --tb=short`
- Lint: `ruff check src/ scripts/ api/ && mypy src/ --ignore-missing-imports`
- Format: `ruff format src/ scripts/ api/`
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
- Dashboard: `streamlit run webapp/app.py` (requires API running on port 8000)

## Code Standards
- Type hints on ALL function signatures (parameters + return)
- Google-style docstrings on every class and public method
- snake_case variables/functions, PascalCase classes
- Max line length: 100 chars
- Use pathlib for paths, never os.path strings
- No hardcoded file paths — everything via config YAML or CLI args
- All random seeds set via src/utils/reproducibility.py
- Log with Python logging module, never print()

## Architecture Rules
- All models inherit from `src.models.base.BaseModel`
- All data modules inherit from `pytorch_lightning.LightningDataModule`
- Configuration: YAML files in configs/ loaded by src/utils/config.py
- Feature extraction: each omics type has its own module in src/features/
- Model encoders: each modality encoder in src/models/encoders/
- Fusion strategies: src/models/fusion/
- Every experiment must be reproducible (seed, config logged to MLflow)

## REST API (api/)
- FastAPI application in api/main.py with lifespan-based startup
- Pydantic schemas in api/schemas.py for request/response validation
- Route modules: api/routes/predict.py, explore.py, explain.py
- Service layer: api/services/model_service.py (singleton, thread-safe),
  api/services/feature_service.py (feature extraction from requests)
- Endpoints: /health, /version, /predict, /predict/batch, /genes,
  /genes/{symbol}, /stats, /model/info, /explain/shap, /explain/attention,
  /explain/global
- CORS enabled for all origins (research demo)
- Tests in tests/test_api.py (29 tests using mocked services)

## Data Sources (ALL FREE, NO AUTHENTICATION WALLS)
- **ClinVar** (pathogenicity labels): FTP download, VCF + TSV, public domain
  - URL: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/
  - Use: variant_summary.txt.gz for labels, clinvar.vcf.gz for variant details
- **cBioPortal** (multi-omics TCGA data): REST API, no auth for public studies
  - URL: https://www.cbioportal.org/api
  - Studies: TCGA PanCancer Atlas studies (suffix _tcga_pan_can_atlas_2018)
  - Provides: mutations, expression (RNA-seq), methylation, CNV, clinical data
- **COSMIC Cancer Gene Census** (validation): requires free academic registration
  - URL: https://cancer.sanger.ac.uk/census
  - Use: known driver gene list for biological validation

## Data Processing Rules
- Genome build: GRCh38 throughout (never mix builds)
- Gene identifiers: Entrez Gene ID as primary key, HUGO symbol as display
- Missing data: document missingness rates, use modality masking in fusion
- Train/val/test split: stratified by label, split by GENE not by variant
  (prevents data leakage from correlated variants in the same gene)
- Class imbalance: use focal loss + stratified sampling

## Testing Rules
- Every new module needs unit tests in tests/
- Use pytest fixtures in tests/conftest.py for shared test data
- Test with small synthetic data — never require full dataset downloads
- Minimum: test input/output shapes, edge cases, missing data handling

## Streamlit Dashboard (webapp/)
- Entry point: `streamlit run webapp/app.py` (port 8501)
- Communicates with FastAPI backend via HTTP (API_URL env var, default localhost:8000)
- Pages: home, single_prediction, batch_analysis, model_performance, data_explorer, about
- Shared utilities: webapp/utils/api_client.py (APIClient class), webapp/utils/styling.py
- Dependencies: streamlit, plotly, requests, reportlab (PDF), openpyxl (Excel)
- Single prediction page is the core feature: input form + results panel with
  prediction card, class probabilities, uncertainty, explanations, biological context, exports

## Git Conventions
- Conventional commits: feat:, fix:, refactor:, docs:, test:, data:
- Branch: feature/, bugfix/, experiment/
- Never commit data files — use .gitignore
- Commit working code only — run tests before committing