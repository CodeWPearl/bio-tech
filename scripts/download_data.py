"""End-to-end data pipeline: download, process, merge and split everything.

This single script wires the whole data layer together so the training-ready
dataset can be built from scratch with one command:

1. Download + process **ClinVar** (pathogenicity labels) — see
   :mod:`src.data.clinvar_loader`.
2. Download all configured **cBioPortal** studies (multi-omics features) — see
   :mod:`src.data.cbioportal_client`.
3. **Merge** the labels onto the mutations and attach the omics features, then
   create gene-level **train/val/test splits** — see :mod:`src.data.data_merger`.

Everything is driven by ``configs/default.yaml`` (URLs, study list, split sizes),
so there are no hardcoded paths beyond the documented defaults (per ``CLAUDE.md``).

Usage:
    python scripts/download_data.py --config configs/default.yaml
    python scripts/download_data.py --skip-download   # rebuild merge/splits only
"""

from __future__ import annotations

import argparse

from src.data.cbioportal_client import CBioPortalClient
from src.data.clinvar_loader import (
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_PATH,
    download_clinvar,
    process_clinvar,
)
from src.data.data_merger import DEFAULT_CBIOPORTAL_DIR, run_full_pipeline
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the data pipeline."""
    parser = argparse.ArgumentParser(
        description="Download, process, merge and split all project data."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download ClinVar even if an up-to-date copy already exists.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip all downloads and only (re)build the merged dataset and splits.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete data pipeline end to end."""
    args = parse_args()
    logger = setup_logging(level="INFO", name=__name__)
    cfg = load_config(args.config)
    studies = list(cfg.data.studies)

    if not args.skip_download:
        logger.info("Step 1/3: ClinVar download + processing")
        download_clinvar(cfg.data.clinvar_url, DEFAULT_RAW_PATH, force=args.force_download)
        process_clinvar(DEFAULT_RAW_PATH, DEFAULT_PROCESSED_PATH)

        logger.info("Step 2/3: cBioPortal download for %d studies", len(studies))
        client = CBioPortalClient(cfg.data.cbioportal_url)
        client.download_all_studies(studies, DEFAULT_CBIOPORTAL_DIR)
    else:
        logger.info("Skipping downloads (--skip-download); rebuilding merge + splits only")

    logger.info("Step 3/3: merge ClinVar + cBioPortal and create gene-level splits")
    result = run_full_pipeline(config_path=args.config)

    merged = result["merged"]
    logger.info(
        "Data pipeline complete: %d merged rows x %d columns; splits saved to data/splits/",
        merged.shape[0],
        merged.shape[1],
    )


if __name__ == "__main__":
    main()
