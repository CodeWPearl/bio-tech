"""Unit tests for :mod:`src.data.data_merger`.

All tests use small synthetic DataFrames (no network, no full datasets — see
``CLAUDE.md``). They cover the labelled merge (including fuzzy-position and
no-match edge cases), omics attachment with missing modalities and top-variable
gene selection, split-by-gene correctness (no gene leakage), and the statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import data_merger as dm


# --- Synthetic builders ------------------------------------------------------


def _clinvar_row(**overrides: object) -> dict[str, object]:
    """A single processed-ClinVar row (only the columns the merger needs)."""
    base: dict[str, object] = {
        "VariationID": "1",
        "GeneSymbol": "TP53",
        "Chromosome": "17",
        "Start": "7676154",
        "ClinicalSignificance": "Pathogenic",
        "ReviewStatus": "criteria provided, single submitter",
        "Type": "single nucleotide variant",
        "label": 0,
    }
    base.update(overrides)
    return base


def _mutation_row(**overrides: object) -> dict[str, object]:
    """A single cBioPortal mutation row (subset of MUTATION_COLUMNS)."""
    base: dict[str, object] = {
        "sample_id": "TCGA-AA-0001-01",
        "patient_id": "TCGA-AA-0001",
        "gene_symbol": "TP53",
        "chromosome": "17",
        "start_position": 7676154,
        "variant_type": "SNP",
        "protein_change": "R175H",
    }
    base.update(overrides)
    return base


# --- merge_clinvar_with_mutations -------------------------------------------


def test_merge_exact_position_match() -> None:
    """A mutation at the exact ClinVar coordinate inherits its label."""
    clinvar = pd.DataFrame([_clinvar_row()])
    mutations = pd.DataFrame([_mutation_row()])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)

    assert len(merged) == 1
    assert merged.loc[0, "label"] == 0
    assert merged.loc[0, "clinvar_clinical_significance"] == "Pathogenic"
    assert merged.loc[0, "sample_id"] == "TCGA-AA-0001-01"
    assert merged.loc[0, "match_distance"] == 0


def test_merge_fuzzy_within_window() -> None:
    """A mutation 3 bp away (≤5) still matches; the distance is recorded."""
    clinvar = pd.DataFrame([_clinvar_row(Start="7676154")])
    mutations = pd.DataFrame([_mutation_row(start_position=7676157)])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)

    assert len(merged) == 1
    assert merged.loc[0, "match_distance"] == 3


def test_merge_outside_window_drops_mutation() -> None:
    """A mutation 6 bp away (>5) does not match and is dropped."""
    clinvar = pd.DataFrame([_clinvar_row(Start="7676154")])
    mutations = pd.DataFrame([_mutation_row(start_position=7676160)])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert merged.empty


def test_merge_requires_gene_match() -> None:
    """Same position but a different gene must not match."""
    clinvar = pd.DataFrame([_clinvar_row(GeneSymbol="TP53")])
    mutations = pd.DataFrame([_mutation_row(gene_symbol="KRAS")])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert merged.empty


def test_merge_requires_chromosome_match() -> None:
    """Same gene/position but a different chromosome must not match."""
    clinvar = pd.DataFrame([_clinvar_row(Chromosome="17")])
    mutations = pd.DataFrame([_mutation_row(chromosome="7")])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert merged.empty


def test_merge_normalises_chr_prefix_and_case() -> None:
    """``chr17``/``17`` and gene-case differences still match."""
    clinvar = pd.DataFrame([_clinvar_row(Chromosome="17", GeneSymbol="TP53")])
    mutations = pd.DataFrame(
        [_mutation_row(chromosome="chr17", gene_symbol="tp53")]
    )
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert len(merged) == 1


def test_merge_picks_closest_clinvar_variant() -> None:
    """Among several in-window ClinVar variants, the closest one wins."""
    clinvar = pd.DataFrame(
        [
            _clinvar_row(VariationID="far", Start="7676150", label=2),
            _clinvar_row(VariationID="near", Start="7676155", label=0),
        ]
    )
    mutations = pd.DataFrame([_mutation_row(start_position=7676154)])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)

    assert len(merged) == 1
    assert merged.loc[0, "clinvar_variation_id"] == "near"
    assert merged.loc[0, "label"] == 0


def test_merge_preserves_mutation_columns() -> None:
    """Every original mutation column survives the merge."""
    clinvar = pd.DataFrame([_clinvar_row()])
    mutations = pd.DataFrame([_mutation_row()])
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    for column in mutations.columns:
        assert column in merged.columns


def test_merge_empty_mutations_returns_typed_empty() -> None:
    """No mutations yields an empty frame carrying the full output schema."""
    clinvar = pd.DataFrame([_clinvar_row()])
    mutations = pd.DataFrame(
        columns=["sample_id", "patient_id", "gene_symbol", "chromosome", "start_position"]
    )
    merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert merged.empty
    assert "label" in merged.columns
    assert "match_distance" in merged.columns


def test_merge_missing_required_column_raises() -> None:
    """A mutation frame lacking a join column fails loudly."""
    clinvar = pd.DataFrame([_clinvar_row()])
    mutations = pd.DataFrame([{"sample_id": "S1", "gene_symbol": "TP53"}])
    with pytest.raises(KeyError, match="missing required columns"):
        dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)


def test_merge_unmatched_count_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Match/no-match counts and rate are reported."""
    clinvar = pd.DataFrame([_clinvar_row(Start="100")])
    mutations = pd.DataFrame(
        [
            _mutation_row(sample_id="hit", start_position=100),
            _mutation_row(sample_id="miss", start_position=99999),
        ]
    )
    with caplog.at_level("INFO"):
        merged = dm.DataMerger().merge_clinvar_with_mutations(clinvar, mutations)
    assert len(merged) == 1
    assert "1 matched, 1 unmatched of 2" in caplog.text


# --- attach_omics_features ---------------------------------------------------


def _merged_two_samples() -> pd.DataFrame:
    """A tiny merged frame with two samples for omics attachment tests."""
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "gene_symbol": ["TP53", "KRAS"],
            "label": [0, 2],
        }
    )


def test_attach_omics_flags_and_columns() -> None:
    """Present samples are flagged True; features are prefixed by modality."""
    merged = _merged_two_samples()
    expression = pd.DataFrame({"sample_id": ["S1", "S2"], "EGFR": [1.0, 5.0]})
    methylation = pd.DataFrame({"sample_id": ["S1"], "MLH1": [0.2]})  # S2 missing
    cnv = pd.DataFrame({"sample_id": ["S1", "S2"], "MYC": [-1, 2]})
    clinical = pd.DataFrame(
        {"sample_id": ["S1", "S2"], "patient_id": ["P1", "P2"], "age": [61.0, 50.0]}
    )

    out = dm.DataMerger().attach_omics_features(
        merged, expression, methylation, cnv, clinical
    )

    assert out["has_expression"].tolist() == [True, True]
    assert out["has_methylation"].tolist() == [True, False]
    assert out["has_cnv"].tolist() == [True, True]
    assert out["has_clinical"].tolist() == [True, True]
    # Prefixed feature columns, no collision across modalities.
    assert "expr_EGFR" in out.columns
    assert "meth_MLH1" in out.columns
    assert "cnv_MYC" in out.columns
    assert "age" in out.columns
    # S2 has no methylation -> NaN feature value.
    s2 = out.set_index("sample_id").loc["S2"]
    assert pd.isna(s2["meth_MLH1"])
    # cBioPortal patient_id is dropped from clinical (mutation one would remain).
    assert "patient_id" not in out.columns or out["age"].notna().all()


def test_attach_omics_missing_modalities() -> None:
    """Empty/None modalities flag all samples False and add no feature columns."""
    merged = _merged_two_samples()
    out = dm.DataMerger().attach_omics_features(
        merged, pd.DataFrame(), None, pd.DataFrame(), None
    )
    assert out["has_expression"].tolist() == [False, False]
    assert out["has_methylation"].tolist() == [False, False]
    assert out["has_cnv"].tolist() == [False, False]
    assert out["has_clinical"].tolist() == [False, False]


def test_attach_omics_keeps_top_variable_genes() -> None:
    """Expression/methylation are reduced to the top-N most variable genes."""
    merged = _merged_two_samples()
    # LOW has ~0 variance, MID small, HIGH large -> top-2 keeps HIGH and MID.
    expression = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "LOW": [1.0, 1.0],
            "MID": [1.0, 3.0],
            "HIGH": [0.0, 100.0],
        }
    )
    merger = dm.DataMerger(top_variable_genes=2)
    out = merger.attach_omics_features(merged, expression, None, None, None)

    assert "expr_HIGH" in out.columns
    assert "expr_MID" in out.columns
    assert "expr_LOW" not in out.columns


def test_attach_omics_requires_sample_id() -> None:
    """A merged frame without sample_id cannot have omics attached."""
    merged = pd.DataFrame({"gene_symbol": ["TP53"], "label": [0]})
    with pytest.raises(KeyError, match="sample_id"):
        dm.DataMerger().attach_omics_features(merged, None, None, None, None)


# --- DataSplitter.split_by_gene ---------------------------------------------


def _multi_gene_df(n_variants_per_gene: int = 4) -> pd.DataFrame:
    """A dataset of several genes, each with multiple variants and a fixed label."""
    genes = {
        "GENE_A": 0, "GENE_B": 0, "GENE_C": 1, "GENE_D": 1,
        "GENE_E": 2, "GENE_F": 2, "GENE_G": 3, "GENE_H": 3,
        "GENE_I": 0, "GENE_J": 1, "GENE_K": 2, "GENE_L": 3,
    }
    rows = []
    for gene, label in genes.items():
        for i in range(n_variants_per_gene):
            rows.append({"gene_symbol": gene, "label": label, "variant": f"{gene}_{i}"})
    return pd.DataFrame(rows)


def test_split_by_gene_no_leakage(tmp_path: Path) -> None:
    """No gene appears in more than one split (the core anti-leakage guarantee)."""
    df = _multi_gene_df()
    splitter = dm.DataSplitter(output_dir=tmp_path)
    splits = splitter.split_by_gene(df, test_size=0.25, val_size=0.25, random_seed=42)

    train_genes = set(splits["train"]["gene_symbol"])
    val_genes = set(splits["val"]["gene_symbol"])
    test_genes = set(splits["test"]["gene_symbol"])

    assert train_genes & val_genes == set()
    assert train_genes & test_genes == set()
    assert val_genes & test_genes == set()


def test_split_by_gene_preserves_all_rows(tmp_path: Path) -> None:
    """Every input row lands in exactly one split (none lost or duplicated)."""
    df = _multi_gene_df()
    splits = dm.DataSplitter(output_dir=tmp_path).split_by_gene(
        df, test_size=0.25, val_size=0.25
    )
    total = sum(len(part) for part in splits.values())
    assert total == len(df)


def test_split_by_gene_keeps_variants_together(tmp_path: Path) -> None:
    """All variants of a gene stay in the same split."""
    df = _multi_gene_df()
    splits = dm.DataSplitter(output_dir=tmp_path).split_by_gene(
        df, test_size=0.25, val_size=0.25
    )
    gene_to_split: dict[str, str] = {}
    for name, part in splits.items():
        for gene in part["gene_symbol"].unique():
            assert gene not in gene_to_split  # never seen in another split
            gene_to_split[gene] = name


def test_split_by_gene_is_deterministic(tmp_path: Path) -> None:
    """The same seed reproduces the same gene partition."""
    df = _multi_gene_df()
    a = dm.DataSplitter(output_dir=tmp_path).split_by_gene(df, 0.25, 0.25, random_seed=7)
    b = dm.DataSplitter(output_dir=tmp_path).split_by_gene(df, 0.25, 0.25, random_seed=7)
    assert set(a["test"]["gene_symbol"]) == set(b["test"]["gene_symbol"])


def test_split_by_gene_writes_artifacts(tmp_path: Path) -> None:
    """Parquet splits and the gene-list JSON are written to output_dir."""
    df = _multi_gene_df()
    dm.DataSplitter(output_dir=tmp_path).split_by_gene(df, 0.25, 0.25, save=True)

    for name in ("train", "val", "test"):
        assert (tmp_path / f"{name}.parquet").is_file()
    gene_splits = json.loads((tmp_path / "gene_splits.json").read_text(encoding="utf-8"))
    assert set(gene_splits) == {"train", "val", "test"}
    # The JSON gene lists do not overlap either.
    assert set(gene_splits["train"]).isdisjoint(gene_splits["test"])


def test_split_by_gene_missing_column_raises(tmp_path: Path) -> None:
    """A missing gene/label column is reported clearly."""
    df = pd.DataFrame({"label": [0, 1]})
    with pytest.raises(KeyError, match="gene_symbol"):
        dm.DataSplitter(output_dir=tmp_path).split_by_gene(df)


# --- DataSplitter.print_split_statistics ------------------------------------


def test_print_split_statistics_values(tmp_path: Path) -> None:
    """Statistics report per-split sizes, gene counts, labels and no-leakage."""
    df = _multi_gene_df()
    splitter = dm.DataSplitter(output_dir=tmp_path)
    splits = splitter.split_by_gene(df, 0.25, 0.25, random_seed=42)
    stats = splitter.print_split_statistics(splits)

    assert stats["no_leakage"] is True
    assert stats["gene_overlaps"] == {}
    for name in ("train", "val", "test"):
        assert stats[name]["n_samples"] == len(splits[name])
        assert stats[name]["n_genes"] == splits[name]["gene_symbol"].nunique()
        assert sum(stats[name]["label_distribution"].values()) == len(splits[name])


def test_print_split_statistics_detects_leakage(tmp_path: Path) -> None:
    """A deliberately leaked gene is detected and reported."""
    shared = pd.DataFrame({"gene_symbol": ["GENE_X"], "label": [0]})
    splits = {
        "train": shared,
        "val": pd.DataFrame({"gene_symbol": ["GENE_Y"], "label": [1]}),
        "test": shared,  # GENE_X leaks into test
    }
    stats = dm.DataSplitter(output_dir=tmp_path).print_split_statistics(splits)
    assert stats["no_leakage"] is False
    assert "GENE_X" in stats["gene_overlaps"]["train&test"]


# --- load_cbioportal_modalities ---------------------------------------------


def test_load_cbioportal_modalities_concatenates(tmp_path: Path) -> None:
    """Per-study modality files are loaded and concatenated; missing ones skipped."""
    for study in ("study_a", "study_b"):
        study_dir = tmp_path / study
        study_dir.mkdir(parents=True)
        pd.DataFrame({"gene_symbol": [study]}).to_parquet(
            study_dir / "mutations.parquet", index=False
        )
    # Only study_a has expression.
    pd.DataFrame({"sample_id": ["S1"], "EGFR": [1.0]}).to_parquet(
        tmp_path / "study_a" / "expression.parquet", index=False
    )

    frames = dm.load_cbioportal_modalities(tmp_path, ["study_a", "study_b"])
    assert len(frames["mutations"]) == 2
    assert set(frames["mutations"]["gene_symbol"]) == {"study_a", "study_b"}
    assert len(frames["expression"]) == 1
    assert frames["methylation"].empty
