"""Unit tests for all feature extraction modules in ``src.features``.

Every test uses small synthetic DataFrames (no real data, no network — per
``CLAUDE.md``).  Coverage: output shapes, feature-name consistency, missing-
value handling, edge cases (unknown variant types, unparseable protein
changes, unseen cancer types at transform time), fit/transform separation,
and the FeaturePipeline orchestrator including save/load round-trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FeaturePipeline
from src.features.clinical_features import ClinicalFeatureExtractor, _parse_stage
from src.features.cnv_features import CNVFeatureExtractor
from src.features.expression_features import ExpressionFeatureExtractor
from src.features.methylation_features import MethylationFeatureExtractor
from src.features.mutation_features import MutationFeatureExtractor


# ---- Synthetic data builders ------------------------------------------------


def _mutation_df(**overrides: object) -> pd.DataFrame:
    """Two-row mutation DataFrame with known values."""
    rows = [
        {
            "sample_id": "S1",
            "gene_symbol": "TP53",
            "variant_classification": "Missense_Mutation",
            "protein_change": "p.R175H",
            "chromosome": "17",
            "start_position": 7676154,
            "variant_type": "SNP",
            "reference_allele": "C",
            "variant_allele": "T",
        },
        {
            "sample_id": "S2",
            "gene_symbol": "KRAS",
            "variant_classification": "Nonsense_Mutation",
            "protein_change": "p.G12V",
            "chromosome": "12",
            "start_position": 25245350,
            "variant_type": "SNP",
            "reference_allele": "G",
            "variant_allele": "A",
        },
    ]
    df = pd.DataFrame(rows)
    for key, val in overrides.items():
        df[key] = val
    return df


def _expression_df(n_samples: int = 5, n_genes: int = 10) -> pd.DataFrame:
    """Synthetic expression matrix with ``expr_`` prefix."""
    rng = np.random.default_rng(42)
    data = {"sample_id": [f"S{i}" for i in range(n_samples)]}
    for g in range(n_genes):
        data[f"expr_GENE{g}"] = rng.exponential(50, size=n_samples)
    return pd.DataFrame(data)


def _methylation_df(n_samples: int = 5, n_genes: int = 10) -> pd.DataFrame:
    """Synthetic methylation DataFrame with ``meth_`` prefix."""
    rng = np.random.default_rng(7)
    data = {"sample_id": [f"S{i}" for i in range(n_samples)]}
    for g in range(n_genes):
        data[f"meth_GENE{g}"] = rng.uniform(0.0, 1.0, size=n_samples)
    return pd.DataFrame(data)


def _cnv_df(n_samples: int = 5, genes: list[str] | None = None) -> pd.DataFrame:
    """Synthetic CNV DataFrame with ``cnv_`` prefix, GISTIC values."""
    rng = np.random.default_rng(99)
    genes = genes or [f"GENE{i}" for i in range(8)]
    data = {"sample_id": [f"S{i}" for i in range(n_samples)]}
    for g in genes:
        data[f"cnv_{g}"] = rng.choice([-2, -1, 0, 1, 2], size=n_samples)
    return pd.DataFrame(data)


def _clinical_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(n)],
        "age": [45, 60, 55, None, 70],
        "sex": ["Male", "Female", "Female", None, "Male"],
        "cancer_type": ["BRCA", "LUAD", "BRCA", "COAD", "LUAD"],
        "stage": ["Stage IIA", "Stage III", "Stage I", None, "Stage IV"],
    })


def _merged_df() -> pd.DataFrame:
    """Minimal merged DataFrame with all modality columns."""
    rng = np.random.default_rng(0)
    n = 4
    data: dict[str, object] = {
        "sample_id": [f"S{i}" for i in range(n)],
        "gene_symbol": ["TP53", "KRAS", "EGFR", "BRAF"],
        "variant_classification": [
            "Missense_Mutation", "Nonsense_Mutation", "Silent", "Splice_Site",
        ],
        "protein_change": ["p.R175H", "p.G12V", None, "p.V600E"],
        "chromosome": ["17", "12", "7", "7"],
        "start_position": [7676154, 25245350, 55249071, 140753336],
        "variant_type": ["SNP", "SNP", "SNP", "SNP"],
        "reference_allele": ["C", "G", "A", "T"],
        "variant_allele": ["T", "A", "G", "A"],
        "label": [0, 1, 2, 3],
        "age": [50, 65, 45, 70],
        "sex": ["Male", "Female", "Female", "Male"],
        "cancer_type": ["BRCA", "LUAD", "BRCA", "COAD"],
        "stage": ["Stage II", "Stage III", "Stage I", "Stage IV"],
    }
    for g in range(3):
        data[f"expr_GENE{g}"] = rng.exponential(50, size=n).tolist()
        data[f"meth_GENE{g}"] = rng.uniform(0.0, 1.0, size=n).tolist()
        data[f"cnv_GENE{g}"] = rng.choice([-2, -1, 0, 1, 2], size=n).tolist()
    return pd.DataFrame(data)


# ===== MutationFeatureExtractor =============================================


class TestMutationFeatureExtractor:
    def test_output_shape(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape == (2, len(ext.feature_names))

    def test_feature_names_match_width(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape[1] == len(ext.feature_names)

    def test_variant_type_onehot(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        missense_idx = ext.feature_names.index("vtype_Missense_Mutation")
        nonsense_idx = ext.feature_names.index("vtype_Nonsense_Mutation")
        assert out[0, missense_idx] == 1.0
        assert out[0, nonsense_idx] == 0.0
        assert out[1, nonsense_idx] == 1.0
        assert out[1, missense_idx] == 0.0

    def test_unknown_variant_type_maps_to_other(self) -> None:
        df = _mutation_df(variant_classification="WEIRD_TYPE")
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        other_idx = ext.feature_names.index("vtype_Other")
        assert out[0, other_idx] == 1.0

    def test_amino_acid_grantham_nonzero(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        grantham_idx = ext.feature_names.index("aa_grantham")
        assert out[0, grantham_idx] > 0

    def test_unparseable_protein_change_gives_zeros(self) -> None:
        df = _mutation_df(protein_change="not_a_change")
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        for name in ("aa_grantham", "aa_blosum62", "aa_hydro_delta"):
            idx = ext.feature_names.index(name)
            assert out[0, idx] == 0.0

    def test_missing_protein_change_gives_zeros(self) -> None:
        df = _mutation_df(protein_change=None)
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        grantham_idx = ext.feature_names.index("aa_grantham")
        assert out[0, grantham_idx] == 0.0

    def test_cosmic_gene_flag(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        cosmic_idx = ext.feature_names.index("gene_cosmic")
        assert out[0, cosmic_idx] == 1.0  # TP53 is in COSMIC
        assert out[1, cosmic_idx] == 1.0  # KRAS is in COSMIC

    def test_non_cosmic_gene(self) -> None:
        df = _mutation_df(gene_symbol="MYGENE99")
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        cosmic_idx = ext.feature_names.index("gene_cosmic")
        assert out[0, cosmic_idx] == 0.0

    def test_chromosome_onehot(self) -> None:
        df = _mutation_df()
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        c17 = ext.feature_names.index("chrom_17")
        c12 = ext.feature_names.index("chrom_12")
        assert out[0, c17] == 1.0
        assert out[1, c12] == 1.0
        assert out[0, c12] == 0.0

    def test_transform_before_fit_raises(self) -> None:
        ext = MutationFeatureExtractor()
        with pytest.raises(RuntimeError, match="fit"):
            ext.transform(_mutation_df())

    def test_empty_df(self) -> None:
        df = _mutation_df().head(0)
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape == (0, len(ext.feature_names))

    def test_no_nan_in_output(self) -> None:
        df = _mutation_df()
        df.loc[0, "protein_change"] = None
        df.loc[1, "start_position"] = None
        ext = MutationFeatureExtractor()
        out = ext.fit_transform(df)
        assert not np.isnan(out).any()


# ===== ExpressionFeatureExtractor ===========================================


class TestExpressionFeatureExtractor:
    def test_output_shape(self) -> None:
        df = _expression_df(5, 10)
        ext = ExpressionFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape == (5, 10)

    def test_feature_names(self) -> None:
        df = _expression_df(5, 10)
        ext = ExpressionFeatureExtractor()
        ext.fit(df)
        assert len(ext.feature_names) == 10
        assert all(name.startswith("expr_") for name in ext.feature_names)

    def test_no_nan_in_output(self) -> None:
        df = _expression_df(5, 10)
        df.iloc[0, 1] = np.nan
        df.iloc[2, 3] = np.nan
        ext = ExpressionFeatureExtractor()
        out = ext.fit_transform(df)
        assert not np.isnan(out).any()

    def test_z_score_approx_centered(self) -> None:
        df = _expression_df(100, 5)
        ext = ExpressionFeatureExtractor()
        out = ext.fit_transform(df)
        means = out.mean(axis=0)
        assert np.allclose(means, 0.0, atol=0.5)

    def test_transform_with_missing_columns(self) -> None:
        train = _expression_df(5, 10)
        ext = ExpressionFeatureExtractor()
        ext.fit(train)
        test = train.drop(columns=["expr_GENE0"])
        out = ext.transform(test)
        assert out.shape == (5, 10)
        assert not np.isnan(out).any()

    def test_empty_expr_columns(self) -> None:
        df = pd.DataFrame({"sample_id": ["S0", "S1"]})
        ext = ExpressionFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape == (2, 0)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            ExpressionFeatureExtractor().transform(
                _expression_df(3, 5)
            )


# ===== MethylationFeatureExtractor ==========================================


class TestMethylationFeatureExtractor:
    def test_output_shape(self) -> None:
        df = _methylation_df(5, 10)
        ext = MethylationFeatureExtractor()
        out = ext.fit_transform(df)
        assert out.shape == (5, 10)

    def test_no_nan_in_output(self) -> None:
        df = _methylation_df(5, 10)
        df.iloc[0, 1] = np.nan
        ext = MethylationFeatureExtractor()
        out = ext.fit_transform(df)
        assert not np.isnan(out).any()

    def test_extreme_beta_clipped(self) -> None:
        df = pd.DataFrame({
            "sample_id": ["S0"],
            "meth_G1": [0.0],
            "meth_G2": [1.0],
        })
        ext = MethylationFeatureExtractor()
        out = ext.fit_transform(df)
        assert np.isfinite(out).all()

    def test_feature_names(self) -> None:
        df = _methylation_df(3, 4)
        ext = MethylationFeatureExtractor()
        ext.fit(df)
        assert len(ext.feature_names) == 4
        assert all(n.startswith("meth_") for n in ext.feature_names)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            MethylationFeatureExtractor().transform(
                _methylation_df(3, 4)
            )


# ===== CNVFeatureExtractor ==================================================


class TestCNVFeatureExtractor:
    def test_ordinal_shape(self) -> None:
        df = _cnv_df(5, ["TP53", "KRAS", "EGFR"])
        ext = CNVFeatureExtractor(mode="ordinal")
        out = ext.fit_transform(df)
        assert out.shape == (5, 3)

    def test_onehot_shape(self) -> None:
        df = _cnv_df(5, ["TP53", "KRAS"])
        ext = CNVFeatureExtractor(mode="onehot")
        out = ext.fit_transform(df)
        assert out.shape == (5, 2 * 5)

    def test_onehot_row_sums(self) -> None:
        df = _cnv_df(5, ["TP53"])
        ext = CNVFeatureExtractor(mode="onehot")
        out = ext.fit_transform(df)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_mutation_gene_intersection(self) -> None:
        df = _cnv_df(5, ["TP53", "KRAS", "FAKEGENE"])
        ext = CNVFeatureExtractor(mode="ordinal")
        ext.fit(df, mutation_genes={"TP53", "KRAS"})
        out = ext.transform(df)
        assert out.shape == (5, 2)

    def test_missing_values_imputed_zero(self) -> None:
        df = _cnv_df(3, ["TP53"])
        df.iloc[0, 1] = np.nan
        ext = CNVFeatureExtractor(mode="ordinal")
        out = ext.fit_transform(df)
        assert out[0, 0] == 0.0

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            CNVFeatureExtractor(mode="bad")

    def test_feature_names_ordinal(self) -> None:
        df = _cnv_df(3, ["TP53", "KRAS"])
        ext = CNVFeatureExtractor(mode="ordinal")
        ext.fit(df)
        assert ext.feature_names == ["cnv_KRAS", "cnv_TP53"]

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            CNVFeatureExtractor().transform(_cnv_df(3, ["TP53"]))


# ===== ClinicalFeatureExtractor =============================================


class TestClinicalFeatureExtractor:
    def test_output_shape(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        n_types = df["cancer_type"].nunique()
        assert out.shape == (5, 3 + n_types)

    def test_feature_names(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        ext.fit(df)
        assert "age_norm" in ext.feature_names
        assert "sex_binary" in ext.feature_names
        assert "stage_ordinal" in ext.feature_names
        assert any(n.startswith("ctype_") for n in ext.feature_names)

    def test_age_normalised(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        age_idx = ext.feature_names.index("age_norm")
        assert 0.0 <= out[:, age_idx].min()
        assert out[:, age_idx].max() <= 1.0

    def test_sex_binary(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        sex_idx = ext.feature_names.index("sex_binary")
        assert out[0, sex_idx] == 1.0   # Male
        assert out[1, sex_idx] == 0.0   # Female

    def test_stage_ordinal(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        stage_idx = ext.feature_names.index("stage_ordinal")
        assert out[2, stage_idx] == pytest.approx(1.0 / 4.0)   # Stage I
        assert out[4, stage_idx] == pytest.approx(4.0 / 4.0)   # Stage IV

    def test_missing_age_imputed(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        age_idx = ext.feature_names.index("age_norm")
        assert not np.isnan(out[3, age_idx])

    def test_missing_sex_imputed(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        sex_idx = ext.feature_names.index("sex_binary")
        assert not np.isnan(out[3, sex_idx])

    def test_unseen_cancer_type_zeros(self) -> None:
        train = _clinical_df()
        ext = ClinicalFeatureExtractor()
        ext.fit(train)
        test = pd.DataFrame({
            "sample_id": ["X1"],
            "age": [55],
            "sex": ["Male"],
            "cancer_type": ["NEVER_SEEN"],
            "stage": ["Stage II"],
        })
        out = ext.transform(test)
        ctype_start = ext.feature_names.index("ctype_BRCA")
        assert out[0, ctype_start:].sum() == 0.0

    def test_no_nan_in_output(self) -> None:
        df = _clinical_df()
        ext = ClinicalFeatureExtractor()
        out = ext.fit_transform(df)
        assert not np.isnan(out).any()

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            ClinicalFeatureExtractor().transform(_clinical_df())


class TestParseStage:
    def test_roman_stages(self) -> None:
        assert _parse_stage("Stage I") == 1.0
        assert _parse_stage("Stage IV") == 4.0

    def test_substage(self) -> None:
        assert _parse_stage("Stage IIA") == 2.0
        assert _parse_stage("Stage IIIB") == 3.0

    def test_numeric(self) -> None:
        assert _parse_stage("2") == 2.0

    def test_missing(self) -> None:
        assert np.isnan(_parse_stage(None))
        assert np.isnan(_parse_stage(""))


# ===== FeaturePipeline ======================================================


class TestFeaturePipeline:
    def test_fit_transform_returns_dict(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        features = pipe.fit_transform(df)
        assert isinstance(features, dict)
        assert set(features.keys()) == {
            "mutation", "expression", "methylation", "cnv", "clinical",
        }

    def test_all_shapes_have_correct_n_samples(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        features = pipe.fit_transform(df)
        for key, arr in features.items():
            assert arr.shape[0] == len(df), f"{key} has wrong n_samples"

    def test_combined_matrix(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        features = pipe.fit_transform(df)
        combined = pipe.get_combined_matrix(features)
        total_features = sum(a.shape[1] for a in features.values())
        assert combined.shape == (len(df), total_features)

    def test_feature_names_dict(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        pipe.fit(df)
        names = pipe.get_feature_names()
        for key in ("mutation", "expression", "methylation", "cnv", "clinical"):
            assert key in names

    def test_no_nan_in_any_output(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        features = pipe.fit_transform(df)
        for key, arr in features.items():
            assert not np.isnan(arr).any(), f"NaN found in {key}"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        df = _merged_df()
        pipe = FeaturePipeline()
        features_before = pipe.fit_transform(df)

        save_path = tmp_path / "pipeline.pkl"
        pipe.save(save_path)
        loaded = FeaturePipeline.load(save_path)
        features_after = loaded.transform(df)

        for key in features_before:
            np.testing.assert_array_almost_equal(
                features_before[key], features_after[key],
            )

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            FeaturePipeline().transform(_merged_df())

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="unfitted"):
            FeaturePipeline().save(tmp_path / "nope.pkl")

    def test_cnv_onehot_mode(self) -> None:
        df = _merged_df()
        pipe = FeaturePipeline(cnv_mode="onehot")
        features = pipe.fit_transform(df)
        assert features["cnv"].shape[0] == len(df)
