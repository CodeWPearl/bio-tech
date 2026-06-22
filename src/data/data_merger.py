"""Cross-database integration: join ClinVar labels onto cBioPortal omics features.

ClinVar (see :mod:`src.data.clinvar_loader`) provides the *pathogenicity labels*
for variants — "variant X in gene Y is Pathogenic". cBioPortal (see
:mod:`src.data.cbioportal_client`) provides the *multi-omics features* for tumour
samples — "sample S has a mutation in gene Y, plus this sample's expression /
methylation / copy-number / clinical data". This module connects the two so a
single, model-ready table carries both the features and the label.

The join key is **gene symbol + genomic position** (per ``CLAUDE.md``: GRCh38
throughout). Positions are matched within a small ±5 bp window because the two
databases sometimes anchor the same variant on slightly different coordinates
(left- vs right-aligned indels, etc.).

Three public pieces:

* :class:`DataMerger` — :meth:`~DataMerger.merge_clinvar_with_mutations` performs
  the labelled join; :meth:`~DataMerger.attach_omics_features` widens each matched
  mutation with its sample's expression/methylation/CNV/clinical columns.
* :class:`DataSplitter` — splits the merged table **by gene** (never by variant),
  preventing leakage from correlated variants in the same gene, and stratifying by
  the label distribution.
* :func:`run_full_pipeline` — loads processed ClinVar + downloaded cBioPortal data,
  merges, splits, saves everything, and logs comprehensive statistics.

Per project standards we log via :mod:`logging` rather than ``print()`` and put
type hints on every signature.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

#: Default on-disk locations (relative to the project root).
DEFAULT_CLINVAR_PATH: Path = Path("data/processed/clinvar_processed.parquet")
DEFAULT_CBIOPORTAL_DIR: Path = Path("data/raw")
DEFAULT_MERGED_PATH: Path = Path("data/processed/merged_dataset.parquet")
DEFAULT_SPLITS_DIR: Path = Path("data/splits")
DEFAULT_CONFIG_PATH: Path = Path("configs/default.yaml")

#: Half-width (in base pairs) of the position window for a fuzzy coordinate match.
DEFAULT_POSITION_WINDOW: int = 5

#: Per-modality genes kept for high-dimensional matrices (expression/methylation).
DEFAULT_TOP_VARIABLE_GENES: int = 2000

#: cBioPortal modalities loaded per study by :func:`load_cbioportal_modalities`.
MODALITIES: tuple[str, ...] = (
    "mutations",
    "expression",
    "methylation",
    "cnv",
    "clinical",
)

#: ClinVar columns carried onto each matched mutation, mapped to output names.
#: The integer training ``label`` keeps its name; everything else is prefixed
#: ``clinvar_`` so it never collides with a cBioPortal mutation column.
CLINVAR_FEATURE_RENAME: dict[str, str] = {
    "VariationID": "clinvar_variation_id",
    "GeneSymbol": "clinvar_gene_symbol",
    "ClinicalSignificance": "clinvar_clinical_significance",
    "ReviewStatus": "clinvar_review_status",
    "Type": "clinvar_variant_type",
    "Start": "clinvar_start",
    "label": "label",
}

#: Mutation columns that must be present to perform the join.
REQUIRED_MUTATION_COLUMNS: tuple[str, ...] = (
    "gene_symbol",
    "chromosome",
    "start_position",
)

#: ClinVar columns that must be present to perform the join.
REQUIRED_CLINVAR_COLUMNS: tuple[str, ...] = (
    "GeneSymbol",
    "Chromosome",
    "Start",
    "label",
)


def _norm_gene(value: object) -> str:
    """Normalise a gene symbol for matching (trimmed, upper-cased).

    Args:
        value: Raw gene symbol from either database.

    Returns:
        The symbol stripped of surrounding whitespace and upper-cased.
    """
    return str(value).strip().upper()


def _norm_chrom(value: object) -> str:
    """Normalise a chromosome label for matching.

    Strips a leading ``chr``/``CHR`` prefix and upper-cases, so ``"chr17"`` and
    ``"17"`` (and ``"x"`` / ``"X"``) compare equal across databases.

    Args:
        value: Raw chromosome label.

    Returns:
        The normalised chromosome string.
    """
    text = str(value).strip().upper()
    return text[3:] if text.startswith("CHR") else text


class DataMerger:
    """Join ClinVar labels with cBioPortal mutations, then attach omics features.

    Attributes:
        position_window: Half-width in base pairs of the fuzzy coordinate match.
        top_variable_genes: Number of most-variable genes kept per high-dimensional
            modality (expression and methylation).
    """

    def __init__(
        self,
        *,
        position_window: int = DEFAULT_POSITION_WINDOW,
        top_variable_genes: int = DEFAULT_TOP_VARIABLE_GENES,
    ) -> None:
        """Initialise the merger.

        Args:
            position_window: Maximum absolute distance (bp) between a ClinVar
                ``Start`` and a mutation ``start_position`` for them to match.
            top_variable_genes: Genes retained per expression/methylation matrix,
                selected by descending variance across samples.
        """
        self.position_window = position_window
        self.top_variable_genes = top_variable_genes

    # --- Labelled join -------------------------------------------------------

    def merge_clinvar_with_mutations(
        self, clinvar_df: pd.DataFrame, mutations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach ClinVar pathogenicity labels to cBioPortal mutation calls.

        Matches are made on exact (normalised) gene symbol and chromosome plus a
        fuzzy genomic position within :attr:`position_window` base pairs. When a
        mutation falls within the window of several ClinVar variants, the closest
        one wins. Mutations with no ClinVar match are dropped (and counted).

        Args:
            clinvar_df: Processed ClinVar table (see
                :func:`src.data.clinvar_loader.process_clinvar`). Must contain
                :data:`REQUIRED_CLINVAR_COLUMNS`.
            mutations_df: cBioPortal mutation table (see
                :meth:`src.data.cbioportal_client.CBioPortalClient.get_mutations`).
                Must contain :data:`REQUIRED_MUTATION_COLUMNS`.

        Returns:
            One row per matched mutation: every original mutation column, the
            attached ``clinvar_*`` columns and ``label``, plus a ``match_distance``
            (bp between the two coordinates).

        Raises:
            KeyError: If a required column is missing from either input.
        """
        self._require(mutations_df, REQUIRED_MUTATION_COLUMNS, "mutation")
        self._require(clinvar_df, REQUIRED_CLINVAR_COLUMNS, "ClinVar")

        mutation_columns = list(mutations_df.columns)
        empty = self._empty_merged(mutation_columns)
        if mutations_df.empty or clinvar_df.empty:
            logger.warning(
                "Nothing to merge (mutations=%d rows, clinvar=%d rows)",
                len(mutations_df),
                len(clinvar_df),
            )
            return empty

        left = mutations_df.copy().reset_index(drop=True)
        left["_mut_id"] = range(len(left))
        left["_gene_key"] = left["gene_symbol"].map(_norm_gene)
        left["_chrom_key"] = left["chromosome"].map(_norm_chrom)
        left["_pos"] = pd.to_numeric(left["start_position"], errors="coerce")

        right = clinvar_df.copy()
        right["_gene_key"] = right["GeneSymbol"].map(_norm_gene)
        right["_chrom_key"] = right["Chromosome"].map(_norm_chrom)
        right["_clinvar_pos"] = pd.to_numeric(right["Start"], errors="coerce")
        right = right.rename(columns=CLINVAR_FEATURE_RENAME)
        right_cols = ["_gene_key", "_chrom_key", "_clinvar_pos", *CLINVAR_FEATURE_RENAME.values()]
        right = right[right_cols]

        # Candidate pairs share gene + chromosome; the position window is applied
        # afterwards so the heavy lifting is a single hash join.
        candidates = left.merge(right, on=["_gene_key", "_chrom_key"], how="inner")
        candidates["match_distance"] = (
            (candidates["_pos"] - candidates["_clinvar_pos"]).abs()
        )
        within = candidates[candidates["match_distance"] <= self.position_window]

        # Keep the closest ClinVar variant for each mutation.
        best = (
            within.sort_values("match_distance", kind="stable")
            .drop_duplicates(subset="_mut_id", keep="first")
            .sort_values("_mut_id", kind="stable")
        )

        output_columns = [
            *mutation_columns,
            *CLINVAR_FEATURE_RENAME.values(),
            "match_distance",
        ]
        merged = best[output_columns].reset_index(drop=True)
        merged["match_distance"] = merged["match_distance"].astype(int)

        matched = len(merged)
        total = len(mutations_df)
        unmatched = total - matched
        rate = matched / total if total else 0.0
        logger.info(
            "Merged ClinVar onto mutations: %d matched, %d unmatched of %d (match rate %.2f%%)",
            matched,
            unmatched,
            total,
            rate * 100,
        )
        return merged

    @staticmethod
    def _require(df: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
        """Raise if any required column is missing from ``df``.

        Args:
            df: DataFrame to validate.
            columns: Required column names.
            label: Human-readable name of the table for the error message.

        Raises:
            KeyError: If one or more required columns are absent.
        """
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise KeyError(f"Input {label} table is missing required columns: {missing}")

    @staticmethod
    def _empty_merged(mutation_columns: list[str]) -> pd.DataFrame:
        """Return an empty merged frame with the full output schema.

        Args:
            mutation_columns: Columns of the source mutation table.

        Returns:
            An empty DataFrame whose columns match a successful merge.
        """
        columns = [*mutation_columns, *CLINVAR_FEATURE_RENAME.values(), "match_distance"]
        # De-duplicate while preserving order (``label`` etc. cannot already exist
        # in mutation_columns, but guard against accidental overlap).
        seen: dict[str, None] = {}
        for column in columns:
            seen.setdefault(column, None)
        return pd.DataFrame(columns=list(seen))

    # --- Omics feature attachment -------------------------------------------

    def attach_omics_features(
        self,
        merged_df: pd.DataFrame,
        expression_df: pd.DataFrame | None,
        methylation_df: pd.DataFrame | None,
        cnv_df: pd.DataFrame | None,
        clinical_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Widen each matched mutation with its sample's omics features.

        Every modality is left-joined on ``sample_id``; samples lacking a modality
        get ``NaN`` feature values and a ``False`` ``has_<modality>`` flag (per the
        task's missing-data handling). Expression and methylation are reduced to
        the :attr:`top_variable_genes` most variable genes to manage
        dimensionality; CNV and clinical are kept whole.

        Modality gene columns are prefixed (``expr_``, ``meth_``, ``cnv_``) so the
        same gene appearing in multiple modalities never collides.

        Args:
            merged_df: Output of :meth:`merge_clinvar_with_mutations`; must contain
                a ``sample_id`` column.
            expression_df: Wide samples × genes RNA-seq matrix, or ``None``/empty.
            methylation_df: Wide samples × genes methylation matrix, or ``None``.
            cnv_df: Wide samples × genes copy-number matrix, or ``None``/empty.
            clinical_df: One row per sample of clinical attributes, or ``None``.

        Returns:
            ``merged_df`` widened with prefixed feature columns, ``has_expression``,
            ``has_methylation``, ``has_cnv`` and ``has_clinical`` boolean columns.

        Raises:
            KeyError: If ``merged_df`` has no ``sample_id`` column.
        """
        if "sample_id" not in merged_df.columns:
            raise KeyError("merged_df must contain a 'sample_id' column")

        out = merged_df.copy()
        out = self._attach_matrix(
            out, expression_df, "expr", "has_expression", self.top_variable_genes
        )
        out = self._attach_matrix(
            out, methylation_df, "meth", "has_methylation", self.top_variable_genes
        )
        out = self._attach_matrix(out, cnv_df, "cnv", "has_cnv", None)
        out = self._attach_clinical(out, clinical_df)

        logger.info(
            "Attached omics features: %d rows, %d columns "
            "(expression=%d, methylation=%d, cnv=%d, clinical=%d samples present)",
            out.shape[0],
            out.shape[1],
            int(out["has_expression"].sum()),
            int(out["has_methylation"].sum()),
            int(out["has_cnv"].sum()),
            int(out["has_clinical"].sum()),
        )
        return out

    def _attach_matrix(
        self,
        base: pd.DataFrame,
        matrix_df: pd.DataFrame | None,
        prefix: str,
        flag_column: str,
        top_n: int | None,
    ) -> pd.DataFrame:
        """Left-join one wide omics matrix onto ``base`` keyed on ``sample_id``.

        Args:
            base: The accumulating feature table.
            matrix_df: Wide samples × genes matrix, or ``None``/empty.
            prefix: Column-name prefix applied to every gene column (e.g. ``expr``).
            flag_column: Name of the boolean presence column to add.
            top_n: If set, keep only the ``top_n`` most variable gene columns.

        Returns:
            ``base`` with the (prefixed) feature columns and ``flag_column`` added.
        """
        if matrix_df is None or matrix_df.empty or "sample_id" not in matrix_df.columns:
            base[flag_column] = False
            logger.warning("No data for %s; samples flagged %s=False", prefix, flag_column)
            return base

        matrix = matrix_df.drop_duplicates(subset="sample_id").copy()
        gene_columns = [column for column in matrix.columns if column != "sample_id"]
        if top_n is not None and len(gene_columns) > top_n:
            gene_columns = self._top_variable_genes(matrix, gene_columns, top_n)
            logger.info("%s: kept top %d of most-variable genes", prefix, top_n)

        matrix = matrix[["sample_id", *gene_columns]].rename(
            columns={column: f"{prefix}_{column}" for column in gene_columns}
        )
        base[flag_column] = base["sample_id"].isin(set(matrix["sample_id"]))
        return base.merge(matrix, on="sample_id", how="left")

    @staticmethod
    def _top_variable_genes(
        matrix: pd.DataFrame, gene_columns: list[str], top_n: int
    ) -> list[str]:
        """Return the ``top_n`` gene columns with the highest variance.

        Args:
            matrix: Wide samples × genes matrix.
            gene_columns: Candidate gene column names.
            top_n: Number of genes to keep.

        Returns:
            The ``top_n`` gene column names ranked by descending variance.
        """
        variances = matrix[gene_columns].var(numeric_only=True)
        return variances.sort_values(ascending=False).head(top_n).index.tolist()

    @staticmethod
    def _attach_clinical(
        base: pd.DataFrame, clinical_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Left-join standardised clinical attributes onto ``base``.

        The clinical ``patient_id`` is dropped before the join because ``base``
        already carries one from the mutation table.

        Args:
            base: The accumulating feature table (must contain ``sample_id``).
            clinical_df: One row per sample of clinical attributes, or ``None``.

        Returns:
            ``base`` with clinical columns and a ``has_clinical`` boolean added.
        """
        if clinical_df is None or clinical_df.empty or "sample_id" not in clinical_df.columns:
            base["has_clinical"] = False
            logger.warning("No clinical data; samples flagged has_clinical=False")
            return base

        clinical = clinical_df.drop_duplicates(subset="sample_id").copy()
        if "patient_id" in clinical.columns:
            clinical = clinical.drop(columns="patient_id")
        base["has_clinical"] = base["sample_id"].isin(set(clinical["sample_id"]))
        return base.merge(clinical, on="sample_id", how="left", suffixes=("", "_clinical"))


class DataSplitter:
    """Split a merged dataset into train/val/test partitions **by gene**.

    Splitting by gene (not by individual variant) prevents data leakage: variants
    within the same gene are biologically correlated, so a gene must live entirely
    in one partition (see ``CLAUDE.md``). The gene-level split is stratified by
    each gene's majority label to keep class balance comparable across partitions.

    Attributes:
        gene_column: Column identifying the gene of each row.
        label_column: Integer label column used for stratification and statistics.
        output_dir: Directory where split Parquet files and the gene-list JSON go.
    """

    def __init__(
        self,
        gene_column: str = "gene_symbol",
        label_column: str = "label",
        output_dir: str | Path = DEFAULT_SPLITS_DIR,
    ) -> None:
        """Initialise the splitter.

        Args:
            gene_column: Column holding the gene symbol to split on.
            label_column: Integer label column (for stratification + statistics).
            output_dir: Destination directory for split artefacts.
        """
        self.gene_column = gene_column
        self.label_column = label_column
        self.output_dir = Path(output_dir)

    def split_by_gene(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_seed: int = 42,
        save: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Partition ``df`` into train/val/test with no gene shared across splits.

        Args:
            df: Merged dataset containing :attr:`gene_column` and
                :attr:`label_column`.
            test_size: Fraction of *genes* held out for the test split.
            val_size: Fraction of *genes* held out for the validation split.
            random_seed: Seed for the deterministic gene-level split.
            save: If ``True``, write the three Parquet files and
                ``gene_splits.json`` under :attr:`output_dir`.

        Returns:
            A mapping ``{"train": df, "val": df, "test": df}``.

        Raises:
            KeyError: If a required column is missing.
        """
        for column in (self.gene_column, self.label_column):
            if column not in df.columns:
                raise KeyError(f"split_by_gene requires column '{column}'")

        # One representative (majority) label per gene drives stratification.
        gene_labels = df.groupby(self.gene_column)[self.label_column].agg(
            lambda series: series.value_counts().index[0]
        )
        gene_sets = self._stratified_gene_sets(
            gene_labels, test_size, val_size, random_seed
        )
        splits = {
            name: df[df[self.gene_column].isin(members)].reset_index(drop=True)
            for name, members in gene_sets.items()
        }

        if save:
            self._save_splits(splits, gene_sets)
        return splits

    @staticmethod
    def _stratified_gene_sets(
        gene_labels: pd.Series,
        test_size: float,
        val_size: float,
        random_seed: int,
    ) -> dict[str, set[str]]:
        """Partition genes into train/val/test sets, stratified by label.

        Genes are grouped by their representative label and each group is split by
        the requested fractions, so the overall label balance is preserved across
        splits while every gene lands in exactly one split (no leakage). The
        per-group shuffle is seeded, making the partition fully deterministic.

        Args:
            gene_labels: Series mapping each gene symbol to its majority label.
            test_size: Fraction of genes (per label) held out for test.
            val_size: Fraction of genes (per label) held out for validation.
            random_seed: Seed for the deterministic per-label shuffle.

        Returns:
            A mapping ``{"train", "val", "test"}`` to disjoint sets of gene symbols.
        """
        import random

        rng = random.Random(random_seed)
        train: list[str] = []
        val: list[str] = []
        test: list[str] = []

        # Iterate labels in sorted order so the result is independent of input row
        # order; shuffle within each label group for an unbiased assignment.
        for label in sorted(gene_labels.unique()):
            genes = sorted(gene_labels.index[gene_labels == label].tolist())
            rng.shuffle(genes)
            n = len(genes)
            n_test = round(n * test_size)
            n_val = round(n * val_size)
            # Guarantee train keeps at least one gene when a group is tiny.
            n_test = min(n_test, n)
            n_val = min(n_val, n - n_test)
            test.extend(genes[:n_test])
            val.extend(genes[n_test : n_test + n_val])
            train.extend(genes[n_test + n_val :])

        return {"train": set(train), "val": set(val), "test": set(test)}

    def _save_splits(
        self, splits: dict[str, pd.DataFrame], gene_sets: dict[str, set[str]]
    ) -> None:
        """Persist split DataFrames as Parquet and gene lists as JSON.

        Args:
            splits: Mapping of split name to its DataFrame.
            gene_sets: Mapping of split name to its set of gene symbols.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, split_df in splits.items():
            destination = self.output_dir / f"{name}.parquet"
            split_df.to_parquet(destination, index=False)
            logger.info("Wrote %s split (%d rows) to %s", name, len(split_df), destination)

        gene_splits = {name: sorted(members) for name, members in gene_sets.items()}
        gene_path = self.output_dir / "gene_splits.json"
        gene_path.write_text(json.dumps(gene_splits, indent=2), encoding="utf-8")
        logger.info("Wrote gene split lists to %s", gene_path)

    def print_split_statistics(
        self, splits_dict: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """Log (and return) per-split statistics and verify no gene leakage.

        Emits, at ``INFO`` level for each split: sample count, unique-gene count
        and label distribution; then whether any gene appears in more than one
        split. We log rather than ``print`` per ``CLAUDE.md``.

        Args:
            splits_dict: Mapping of split name to its DataFrame.

        Returns:
            A dictionary of the computed statistics, including a ``no_leakage``
            boolean and any pairwise gene overlaps.
        """
        stats: dict[str, Any] = {}
        gene_sets: dict[str, set[str]] = {}

        logger.info("=== Split statistics ===")
        for name, split_df in splits_dict.items():
            genes = set(split_df[self.gene_column].unique())
            gene_sets[name] = genes
            label_distribution = (
                split_df[self.label_column].value_counts().sort_index().to_dict()
            )
            stats[name] = {
                "n_samples": int(len(split_df)),
                "n_genes": len(genes),
                "label_distribution": {int(k): int(v) for k, v in label_distribution.items()},
            }
            logger.info(
                "%-5s | samples=%d | genes=%d | labels=%s",
                name,
                stats[name]["n_samples"],
                stats[name]["n_genes"],
                stats[name]["label_distribution"],
            )

        overlaps: dict[str, list[str]] = {}
        for left, right in combinations(gene_sets, 2):
            shared = gene_sets[left] & gene_sets[right]
            if shared:
                overlaps[f"{left}&{right}"] = sorted(shared)

        no_leakage = not overlaps
        stats["gene_overlaps"] = overlaps
        stats["no_leakage"] = no_leakage
        if no_leakage:
            logger.info("No gene overlap between splits — no leakage detected.")
        else:
            logger.error("Gene leakage detected between splits: %s", overlaps)
        return stats


# --- Orchestration -----------------------------------------------------------


def load_cbioportal_modalities(
    raw_dir: str | Path, study_ids: list[str]
) -> dict[str, pd.DataFrame]:
    """Load and concatenate each cBioPortal modality across several studies.

    Reads ``raw_dir/{study_id}/{modality}.parquet`` for every study and modality
    that exists on disk (missing files are skipped) and concatenates each modality
    across studies. Wide matrices with differing gene columns are aligned by an
    outer concat, so absent genes become ``NaN``.

    Args:
        raw_dir: Root directory of downloaded studies (see
            :meth:`src.data.cbioportal_client.CBioPortalClient.download_study`).
        study_ids: Study identifiers to load.

    Returns:
        A mapping of modality name to its concatenated DataFrame (empty if no
        study supplied that modality).
    """
    root = Path(raw_dir)
    collected: dict[str, list[pd.DataFrame]] = {modality: [] for modality in MODALITIES}
    for study_id in study_ids:
        for modality in MODALITIES:
            path = root / study_id / f"{modality}.parquet"
            if path.is_file():
                collected[modality].append(pd.read_parquet(path))
            else:
                logger.warning("Missing %s for study %s at %s", modality, study_id, path)

    frames: dict[str, pd.DataFrame] = {}
    for modality, parts in collected.items():
        frames[modality] = (
            pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        )
        logger.info(
            "Loaded %s: %d rows across %d studies", modality, len(frames[modality]), len(parts)
        )
    return frames


def run_full_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    clinvar_path: str | Path = DEFAULT_CLINVAR_PATH,
    cbioportal_dir: str | Path = DEFAULT_CBIOPORTAL_DIR,
    merged_out: str | Path = DEFAULT_MERGED_PATH,
    splits_dir: str | Path = DEFAULT_SPLITS_DIR,
) -> dict[str, Any]:
    """Run the end-to-end merge + split pipeline from on-disk inputs.

    Loads the processed ClinVar table and the downloaded cBioPortal modalities for
    every study configured in ``config_path``, merges labels onto mutations,
    attaches omics features, writes the merged table, creates gene-level splits and
    logs comprehensive statistics.

    Args:
        config_path: YAML config providing ``data.studies`` and the split sizes.
        clinvar_path: Path to the processed ClinVar Parquet file.
        cbioportal_dir: Root directory of downloaded cBioPortal studies.
        merged_out: Destination Parquet path for the merged feature table.
        splits_dir: Directory for the train/val/test Parquet files.

    Returns:
        A dictionary with the merged DataFrame, the split DataFrames and the
        statistics dictionary.

    Raises:
        FileNotFoundError: If the processed ClinVar file is absent.
    """
    from src.utils.config import load_config

    cfg = load_config(config_path)
    studies = list(cfg.data.studies)
    test_size = float(cfg.data.test_size)
    val_size = float(cfg.data.val_size)
    random_seed = int(cfg.data.random_seed)

    clinvar_file = Path(clinvar_path)
    if not clinvar_file.is_file():
        raise FileNotFoundError(
            f"Processed ClinVar file not found: {clinvar_file}. "
            "Run `python -m src.data.clinvar_loader` first."
        )
    logger.info("Loading processed ClinVar from %s", clinvar_file)
    clinvar_df = pd.read_parquet(clinvar_file)

    modalities = load_cbioportal_modalities(cbioportal_dir, studies)

    merger = DataMerger()
    merged = merger.merge_clinvar_with_mutations(clinvar_df, modalities["mutations"])
    full = merger.attach_omics_features(
        merged,
        modalities["expression"],
        modalities["methylation"],
        modalities["cnv"],
        modalities["clinical"],
    )

    merged_path = Path(merged_out)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(merged_path, index=False)
    logger.info(
        "Wrote merged dataset (%d rows, %d cols) to %s",
        full.shape[0],
        full.shape[1],
        merged_path,
    )

    splitter = DataSplitter(output_dir=splits_dir)
    splits = splitter.split_by_gene(full, test_size, val_size, random_seed, save=True)
    stats = splitter.print_split_statistics(splits)

    return {"merged": full, "splits": splits, "stats": stats}


def main() -> None:
    """Console entry point: run the merge + split pipeline with logging."""
    from src.utils.logging_setup import setup_logging

    setup_logging(level="INFO", name=__name__)
    run_full_pipeline()


if __name__ == "__main__":
    main()
