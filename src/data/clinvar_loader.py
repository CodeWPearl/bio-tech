"""ClinVar download and processing pipeline.

This module fetches the public ``variant_summary.txt.gz`` table from NCBI ClinVar
and turns it into a clean, labelled, deduplicated dataset of single-nucleotide and
small indel variants restricted to genome build **GRCh38** (see ``CLAUDE.md`` —
never mix genome builds).

The pipeline has three stages, each exposed as a standalone, type-hinted function
so it can be unit-tested in isolation:

1. :func:`download_clinvar` — stream the gzipped TSV to ``data/raw`` with a tqdm
   progress bar, skipping the download when an up-to-date copy already exists.
2. :func:`process_clinvar` — read, filter, label, and deduplicate the table, then
   persist it as a Parquet file under ``data/processed``.
3. :func:`summarize` — log class balance, top genes, and other distributions.

Per project standards we log via :mod:`logging` rather than calling ``print()``;
the summary statistics required by the task are emitted at ``INFO`` level.

Example:
    >>> from src.data.clinvar_loader import main
    >>> main()  # downloads (if needed) and writes data/processed/clinvar_processed.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

#: Public NCBI ClinVar download URL for the tab-delimited variant summary.
CLINVAR_URL: str = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

#: Default on-disk locations (relative to the project root).
DEFAULT_RAW_PATH: Path = Path("data/raw/variant_summary.txt.gz")
DEFAULT_PROCESSED_PATH: Path = Path("data/processed/clinvar_processed.parquet")

#: Columns extracted from the raw ClinVar TSV. Order is preserved in the output.
COLUMNS: tuple[str, ...] = (
    "VariationID",
    "Name",
    "GeneSymbol",
    "GeneID",
    "ClinicalSignificance",
    "ClinSigSimple",
    "ReviewStatus",
    "NumberSubmitters",
    "Type",
    "Chromosome",
    "Start",
    "Stop",
    "Assembly",
    "PhenotypeList",
    "Origin",
)

#: Genome build kept throughout the project.
TARGET_ASSEMBLY: str = "GRCh38"

#: Clinical significance values kept, mapped to integer training labels.
LABEL_MAP: dict[str, int] = {
    "Pathogenic": 0,
    "Likely pathogenic": 1,
    "Benign": 2,
    "Likely benign": 3,
}

#: Reverse mapping for human-readable summaries.
CLASS_NAMES: dict[int, str] = {value: key for key, value in LABEL_MAP.items()}

#: Small-scale variant types kept (large structural variants are dropped).
ALLOWED_TYPES: frozenset[str] = frozenset(
    {
        "single nucleotide variant",
        "Deletion",
        "Insertion",
        "Indel",
        "Duplication",
    }
)

#: ReviewStatus strings carrying zero stars; rows with these are dropped because
#: the task requires at least a 1-star assertion. Compared case-insensitively.
ZERO_STAR_REVIEW_STATUSES: frozenset[str] = frozenset(
    {
        "no assertion criteria provided",
        "no assertion provided",
        "no classification provided",
        "no classification for the individual variant",
        "no classifications from unflagged records",
        "no interpretation for the single variant",
        "no assertion for the individual variant",
    }
)

#: Columns used to detect duplicate variants.
DEDUP_KEYS: tuple[str, ...] = ("GeneSymbol", "Chromosome", "Start", "Stop")


# --- Download ----------------------------------------------------------------


def download_clinvar(
    url: str = CLINVAR_URL,
    dest: str | Path = DEFAULT_RAW_PATH,
    *,
    force: bool = False,
    chunk_size: int = 1 << 16,
    session: Any | None = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> Path:
    """Download the ClinVar variant summary, resuming and retrying as needed.

    The remote ``Content-Length`` is compared against the size of any existing
    file; the download is skipped when they match (and ``force`` is ``False``).
    A partially downloaded file is *resumed* via an HTTP ``Range`` request rather
    than restarted, and dropped connections (NCBI servers reset them often) are
    retried up to ``max_retries`` times, continuing from the last byte received.
    Progress is shown with a tqdm bar.

    Args:
        url: Source URL of the gzipped variant summary TSV.
        dest: Destination path; parent directories are created as needed.
        force: If ``True``, re-download from scratch even when a file exists.
        chunk_size: Number of bytes streamed per write.
        session: Object exposing a ``requests``-style ``get`` (for dependency
            injection in tests). Defaults to the :mod:`requests` module.
        timeout: Per-request timeout in seconds.
        max_retries: Maximum number of resume attempts after a connection drop.

    Returns:
        The path to the downloaded (or already-present) file.

    Raises:
        requests.HTTPError: If the server returns an unsuccessful status code.
        requests.exceptions.ChunkedEncodingError: If the connection keeps dropping
            past ``max_retries`` (or the server does not report a total size).
    """
    # Imported lazily so the module imports without third-party deps installed.
    import time

    import requests
    from requests.exceptions import ChunkedEncodingError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from tqdm import tqdm

    http = session if session is not None else requests
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe the remote with a plain (non-range) request to learn the total size.
    response = http.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0) or 0)

    existing = dest_path.stat().st_size if (dest_path.exists() and not force) else 0
    if total > 0 and existing == total:
        logger.info(
            "ClinVar file already present and complete (%d bytes); skipping download: %s",
            total,
            dest_path,
        )
        response.close()
        return dest_path

    # Resume a partial file by re-requesting only the missing byte range.
    if 0 < existing < total:
        logger.info("Resuming download from %d / %d bytes", existing, total)
        response.close()
        response = http.get(
            url, stream=True, timeout=timeout, headers={"Range": f"bytes={existing}-"}
        )
        response.raise_for_status()
        mode, downloaded = "ab", existing
    else:
        logger.info("Downloading ClinVar variant summary from %s", url)
        mode, downloaded = "wb", 0

    with dest_path.open(mode) as handle, tqdm(
        total=total or None,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="variant_summary.txt.gz",
    ) as bar:
        attempt = 0
        while True:
            try:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    bar.update(len(chunk))
                break  # stream consumed cleanly
            except (ChunkedEncodingError, RequestsConnectionError) as exc:
                attempt += 1
                if attempt > max_retries or total == 0:
                    raise
                handle.flush()
                logger.warning(
                    "Connection dropped at %d/%d bytes; resuming (attempt %d/%d): %s",
                    downloaded,
                    total,
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(min(2 * attempt, 10))
                response.close()
                response = http.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    headers={"Range": f"bytes={downloaded}-"},
                )
                response.raise_for_status()

    logger.info("Saved ClinVar file to %s (%d bytes)", dest_path, dest_path.stat().st_size)
    return dest_path


# --- Read / filter / label / deduplicate ------------------------------------


def read_variant_summary(path: str | Path) -> pd.DataFrame:
    """Read the gzipped ClinVar TSV, keeping only the columns in :data:`COLUMNS`.

    Args:
        path: Path to ``variant_summary.txt.gz`` (gzip auto-detected by suffix).

    Returns:
        A DataFrame restricted to :data:`COLUMNS`, with all values read as
        strings to avoid silent type coercion before explicit cleaning.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"ClinVar file not found: {file_path}")

    logger.info("Reading ClinVar variant summary from %s", file_path)
    frame = pd.read_csv(
        file_path,
        sep="\t",
        usecols=list(COLUMNS),
        dtype=str,
        na_filter=False,
        compression="infer",
    )
    logger.info("Read %d raw variant rows", len(frame))
    return frame[list(COLUMNS)]


def _require_columns(df: pd.DataFrame) -> None:
    """Raise if any required column is absent from ``df``.

    Args:
        df: DataFrame to validate.

    Raises:
        KeyError: If one or more of :data:`COLUMNS` is missing.
    """
    missing = [column for column in COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required ClinVar columns: {missing}")


def filter_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all ClinVar inclusion filters and return the surviving rows.

    Keeps rows that satisfy every one of: ``Assembly == "GRCh38"``; an ``Origin``
    mentioning germline or somatic; an exact :data:`LABEL_MAP` clinical
    significance; a ReviewStatus with at least one star; a non-empty
    ``GeneSymbol`` other than ``"-"``; and a small-scale variant ``Type``.

    Args:
        df: Raw variant DataFrame containing at least :data:`COLUMNS`.

    Returns:
        A filtered copy of ``df`` with ``NumberSubmitters`` coerced to ``int`` and
        a fresh ``RangeIndex``.

    Raises:
        KeyError: If required columns are missing.
    """
    _require_columns(df)
    out = df.copy()

    # Normalise the numeric submitter count up front; it drives deduplication and
    # may arrive as strings or with blanks from the raw TSV.
    out["NumberSubmitters"] = (
        pd.to_numeric(out["NumberSubmitters"], errors="coerce").fillna(0).astype(int)
    )

    significance = out["ClinicalSignificance"].astype(str).str.strip()
    gene = out["GeneSymbol"].astype(str).str.strip()
    review = out["ReviewStatus"].astype(str).str.strip().str.lower()
    origin = out["Origin"].astype(str).str.lower()
    variant_type = out["Type"].astype(str).str.strip()
    assembly = out["Assembly"].astype(str).str.strip()

    mask = (
        (assembly == TARGET_ASSEMBLY)
        & (origin.str.contains("germline") | origin.str.contains("somatic"))
        & significance.isin(LABEL_MAP.keys())
        & (~review.isin(ZERO_STAR_REVIEW_STATUSES))
        & (gene != "") & (gene != "-")
        & variant_type.isin(ALLOWED_TYPES)
    )

    filtered = out.loc[mask].reset_index(drop=True)
    logger.info("Filtering kept %d of %d rows", len(filtered), len(df))
    return filtered


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add an integer ``label`` column derived from ``ClinicalSignificance``.

    Args:
        df: DataFrame containing a ``ClinicalSignificance`` column whose values
            are already restricted to the keys of :data:`LABEL_MAP`.

    Returns:
        A copy of ``df`` with an added integer ``label`` column.

    Raises:
        ValueError: If any clinical significance value cannot be mapped.
    """
    out = df.copy()
    labels = out["ClinicalSignificance"].astype(str).str.strip().map(LABEL_MAP)
    if labels.isna().any():
        unmapped = sorted(
            out.loc[labels.isna(), "ClinicalSignificance"].astype(str).unique()
        )
        raise ValueError(f"Unmapped ClinicalSignificance values: {unmapped}")
    out["label"] = labels.astype(int)
    return out


def deduplicate_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate variants, keeping the best-supported record per locus.

    Duplicates are identified by :data:`DEDUP_KEYS` (gene symbol plus genomic
    coordinates). Among duplicates, the row with the highest ``NumberSubmitters``
    is retained.

    Args:
        df: DataFrame containing :data:`DEDUP_KEYS` and ``NumberSubmitters``.

    Returns:
        A deduplicated copy of ``df`` with a fresh ``RangeIndex``.
    """
    out = df.copy()
    out["NumberSubmitters"] = (
        pd.to_numeric(out["NumberSubmitters"], errors="coerce").fillna(0).astype(int)
    )
    # Stable sort by support descending, then keep the first of each locus group.
    deduped = (
        out.sort_values("NumberSubmitters", ascending=False, kind="stable")
        .drop_duplicates(subset=list(DEDUP_KEYS), keep="first")
        .sort_index()
        .reset_index(drop=True)
    )
    logger.info("Deduplication kept %d of %d rows", len(deduped), len(df))
    return deduped


# --- Summary -----------------------------------------------------------------


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Log and return summary statistics for a processed ClinVar DataFrame.

    Emits, at ``INFO`` level: counts per class, the top 20 genes by variant
    count, the variant-type distribution, and the review-status distribution.

    Args:
        df: Processed DataFrame containing ``label``, ``GeneSymbol``, ``Type``,
            and ``ReviewStatus`` columns.

    Returns:
        A dictionary of the computed distributions (useful for testing/logging).
    """
    class_counts = df["label"].map(CLASS_NAMES).value_counts()
    top_genes = df["GeneSymbol"].value_counts().head(20)
    type_counts = df["Type"].value_counts()
    review_counts = df["ReviewStatus"].value_counts()

    logger.info("=== ClinVar processed summary (%d variants) ===", len(df))
    logger.info("Variants per class:\n%s", class_counts.to_string())
    logger.info("Top 20 genes by variant count:\n%s", top_genes.to_string())
    logger.info("Variant type distribution:\n%s", type_counts.to_string())
    logger.info("Review status distribution:\n%s", review_counts.to_string())

    return {
        "total": int(len(df)),
        "class_counts": class_counts.to_dict(),
        "top_genes": top_genes.to_dict(),
        "type_counts": type_counts.to_dict(),
        "review_counts": review_counts.to_dict(),
    }


# --- Orchestration -----------------------------------------------------------


def process_clinvar(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    out_path: str | Path = DEFAULT_PROCESSED_PATH,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Run the full read → filter → label → deduplicate → save pipeline.

    The raw TSV (~9M rows) is read and filtered in chunks so peak memory stays
    bounded; only the small set of surviving variants is held in full. Filtering
    each chunk drops the bulk of rows (non-GRCh38 builds, uncertain/conflicting
    significance, large structural variants) before any expensive whole-frame
    work happens.

    Args:
        raw_path: Path to the raw ``variant_summary.txt.gz`` file.
        out_path: Destination Parquet path for the processed dataset.
        chunksize: Number of raw rows read per chunk.

    Returns:
        The processed DataFrame that was written to ``out_path``.

    Raises:
        FileNotFoundError: If ``raw_path`` does not exist.
    """
    raw_file = Path(raw_path)
    if not raw_file.is_file():
        raise FileNotFoundError(f"ClinVar file not found: {raw_file}")

    logger.info("Reading and filtering %s in chunks of %d rows", raw_file, chunksize)
    reader = pd.read_csv(
        raw_file,
        sep="\t",
        usecols=list(COLUMNS),
        dtype=str,
        na_filter=False,
        compression="infer",
        chunksize=chunksize,
    )

    filtered_parts: list[pd.DataFrame] = []
    total_rows = 0
    for chunk in reader:
        total_rows += len(chunk)
        filtered_parts.append(filter_variants(chunk[list(COLUMNS)]))

    filtered = (
        pd.concat(filtered_parts, ignore_index=True)
        if filtered_parts
        else pd.DataFrame(columns=list(COLUMNS))
    )
    logger.info("Filtering kept %d of %d total raw rows", len(filtered), total_rows)

    labelled = add_labels(filtered)
    deduped = deduplicate_variants(labelled)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_parquet(out, index=False)
    logger.info("Wrote processed ClinVar dataset to %s", out)

    summarize(deduped)
    return deduped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ClinVar pipeline."""
    parser = argparse.ArgumentParser(description="Download and process ClinVar variants.")
    parser.add_argument("--url", default=CLINVAR_URL, help="ClinVar download URL.")
    parser.add_argument(
        "--raw-path",
        default=str(DEFAULT_RAW_PATH),
        help="Where to store the raw downloaded TSV.",
    )
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_PROCESSED_PATH),
        help="Where to write the processed Parquet file.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if an up-to-date file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Console entry point: download (if needed) then process ClinVar."""
    from src.utils.logging_setup import setup_logging

    args = parse_args()
    setup_logging(level="INFO", name=__name__)
    download_clinvar(args.url, args.raw_path, force=args.force_download)
    process_clinvar(args.raw_path, args.out_path)


if __name__ == "__main__":
    main()
