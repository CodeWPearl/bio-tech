"""Unit tests for :mod:`src.data.clinvar_loader`.

These tests use small synthetic DataFrames and a mocked HTTP layer so they run
fast and never touch the network or the full ClinVar dataset (see ``CLAUDE.md``:
"Test with small synthetic data — never require full dataset downloads").
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import requests

from src.data import cbioportal_client as cbio
from src.data import clinvar_loader as cl


# --- Fixtures ----------------------------------------------------------------


def _row(**overrides: object) -> dict[str, object]:
    """Build a single valid synthetic ClinVar row, applying any overrides.

    The defaults describe a row that passes every inclusion filter; individual
    tests override single fields to exercise specific rejection paths.
    """
    base: dict[str, object] = {
        "VariationID": "1",
        "Name": "NM_000546.6(TP53):c.215C>G",
        "GeneSymbol": "TP53",
        "GeneID": "7157",
        "ClinicalSignificance": "Pathogenic",
        "ClinSigSimple": "1",
        "ReviewStatus": "criteria provided, single submitter",
        "NumberSubmitters": 3,
        "Type": "single nucleotide variant",
        "Chromosome": "17",
        "Start": "7676154",
        "Stop": "7676154",
        "Assembly": "GRCh38",
        "PhenotypeList": "Li-Fraumeni syndrome",
        "Origin": "germline",
    }
    base.update(overrides)
    return base


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A small mixed DataFrame: valid rows plus several that must be filtered."""
    rows = [
        _row(VariationID="1", ClinicalSignificance="Pathogenic"),
        _row(VariationID="2", GeneSymbol="BRCA1", ClinicalSignificance="Likely benign"),
        _row(VariationID="3", GeneSymbol="BRCA2", ClinicalSignificance="Benign",
             Origin="somatic"),
        # --- rows that must be dropped ---
        _row(VariationID="4", Assembly="GRCh37"),                       # wrong build
        _row(VariationID="5", ClinicalSignificance="Uncertain significance"),
        _row(VariationID="6", ClinicalSignificance="Conflicting interpretations"),
        _row(VariationID="7", ReviewStatus="no assertion criteria provided"),
        _row(VariationID="8", GeneSymbol="-"),                          # no gene
        _row(VariationID="9", GeneSymbol=""),                           # empty gene
        _row(VariationID="10", Type="copy number loss"),               # large SV
        _row(VariationID="11", Origin="unknown"),                       # bad origin
    ]
    return pd.DataFrame(rows)


# --- Download ----------------------------------------------------------------


def test_download_uses_mocked_http(tmp_path: Path) -> None:
    """The download streams mocked content to disk without real network I/O."""
    payload = b"col1\tcol2\nval1\tval2\n"
    response = mock.MagicMock()
    response.headers = {"Content-Length": str(len(payload))}
    response.iter_content.return_value = [payload]
    response.raise_for_status.return_value = None
    session = mock.MagicMock()
    session.get.return_value = response

    dest = tmp_path / "raw" / "variant_summary.txt.gz"
    result = cl.download_clinvar(dest=dest, session=session)

    assert result == dest
    assert dest.read_bytes() == payload
    session.get.assert_called_once()


def test_download_skips_when_size_matches(tmp_path: Path) -> None:
    """An existing file whose size matches Content-Length is not re-downloaded."""
    payload = b"already-here"
    dest = tmp_path / "variant_summary.txt.gz"
    dest.write_bytes(payload)

    response = mock.MagicMock()
    response.headers = {"Content-Length": str(len(payload))}
    response.raise_for_status.return_value = None
    session = mock.MagicMock()
    session.get.return_value = response

    cl.download_clinvar(dest=dest, session=session)

    # Skipped: streaming body was never consumed and the file is untouched.
    response.iter_content.assert_not_called()
    assert dest.read_bytes() == payload


def test_download_resumes_partial_file(tmp_path: Path) -> None:
    """An incomplete file is resumed via a Range request, not restarted."""
    dest = tmp_path / "variant_summary.txt.gz"
    dest.write_bytes(b"AAAA")  # 4 of 10 bytes already on disk

    probe = mock.MagicMock()
    probe.headers = {"Content-Length": "10"}
    probe.raise_for_status.return_value = None
    resume = mock.MagicMock()
    resume.headers = {"Content-Length": "6"}
    resume.iter_content.return_value = [b"BBBBBB"]
    resume.raise_for_status.return_value = None
    session = mock.MagicMock()
    session.get.side_effect = [probe, resume]

    cl.download_clinvar(dest=dest, session=session)

    # First call probes size; second requests the remaining bytes via Range.
    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["headers"] == {"Range": "bytes=4-"}
    assert dest.read_bytes() == b"AAAABBBBBB"


def test_download_retries_on_connection_drop(tmp_path: Path) -> None:
    """A dropped connection mid-stream is retried and resumed to completion."""
    from requests.exceptions import ChunkedEncodingError

    def first_stream():
        yield b"AAAA"
        raise ChunkedEncodingError("connection reset")

    probe = mock.MagicMock()
    probe.headers = {"Content-Length": "8"}
    probe.iter_content.side_effect = lambda chunk_size: first_stream()
    probe.raise_for_status.return_value = None
    retry = mock.MagicMock()
    retry.headers = {"Content-Length": "4"}
    retry.iter_content.return_value = [b"BBBB"]
    retry.raise_for_status.return_value = None
    session = mock.MagicMock()
    session.get.side_effect = [probe, retry]

    dest = tmp_path / "variant_summary.txt.gz"
    with mock.patch("time.sleep"):
        cl.download_clinvar(dest=dest, session=session)

    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["headers"] == {"Range": "bytes=4-"}
    assert dest.read_bytes() == b"AAAABBBB"


def test_download_force_redownloads(tmp_path: Path) -> None:
    """``force=True`` re-downloads even when a matching file exists."""
    payload = b"new-bytes"
    dest = tmp_path / "variant_summary.txt.gz"
    dest.write_bytes(payload)

    response = mock.MagicMock()
    response.headers = {"Content-Length": str(len(payload))}
    response.iter_content.return_value = [payload]
    response.raise_for_status.return_value = None
    session = mock.MagicMock()
    session.get.return_value = response

    cl.download_clinvar(dest=dest, session=session, force=True)
    response.iter_content.assert_called_once()


# --- Filtering ---------------------------------------------------------------


def test_filter_keeps_only_valid_rows(synthetic_df: pd.DataFrame) -> None:
    """Only the three valid rows survive all inclusion filters."""
    filtered = cl.filter_variants(synthetic_df)
    assert set(filtered["VariationID"]) == {"1", "2", "3"}


def test_filter_rejects_wrong_assembly(synthetic_df: pd.DataFrame) -> None:
    """Non-GRCh38 rows are removed (never mix genome builds)."""
    filtered = cl.filter_variants(synthetic_df)
    assert (filtered["Assembly"] == "GRCh38").all()


def test_filter_rejects_uncertain_and_conflicting() -> None:
    """Uncertain and conflicting significances are dropped."""
    df = pd.DataFrame(
        [
            _row(ClinicalSignificance="Uncertain significance"),
            _row(ClinicalSignificance="Conflicting interpretations"),
        ]
    )
    assert cl.filter_variants(df).empty


def test_filter_requires_at_least_one_star() -> None:
    """Zero-star review statuses are excluded."""
    df = pd.DataFrame([_row(ReviewStatus="no assertion criteria provided")])
    assert cl.filter_variants(df).empty


def test_filter_accepts_germline_and_somatic_origins() -> None:
    """Both germline and somatic origins (and combinations) are accepted."""
    df = pd.DataFrame(
        [
            _row(VariationID="a", Origin="germline"),
            _row(VariationID="b", Origin="somatic"),
            _row(VariationID="c", Origin="germline;somatic"),
        ]
    )
    assert len(cl.filter_variants(df)) == 3


def test_filter_keeps_only_small_variant_types() -> None:
    """All allowed small-scale types pass; large structural variants do not."""
    allowed = pd.DataFrame(
        [_row(VariationID=str(i), Type=t) for i, t in enumerate(cl.ALLOWED_TYPES)]
    )
    assert len(cl.filter_variants(allowed)) == len(cl.ALLOWED_TYPES)

    rejected = pd.DataFrame([_row(Type="copy number gain")])
    assert cl.filter_variants(rejected).empty


# --- Label mapping -----------------------------------------------------------


def test_label_mapping_values() -> None:
    """Each significance maps to its documented integer label."""
    df = pd.DataFrame(
        [
            _row(VariationID="1", ClinicalSignificance="Pathogenic"),
            _row(VariationID="2", ClinicalSignificance="Likely pathogenic"),
            _row(VariationID="3", ClinicalSignificance="Benign"),
            _row(VariationID="4", ClinicalSignificance="Likely benign"),
        ]
    )
    labelled = cl.add_labels(df)
    assert labelled["label"].tolist() == [0, 1, 2, 3]


def test_label_mapping_rejects_unknown_significance() -> None:
    """An unmappable significance raises a clear error."""
    df = pd.DataFrame([_row(ClinicalSignificance="drug response")])
    with pytest.raises(ValueError, match="Unmapped"):
        cl.add_labels(df)


# --- Deduplication -----------------------------------------------------------


def test_deduplication_keeps_highest_submitter_count() -> None:
    """Among duplicate loci, the record with most submitters is retained."""
    df = pd.DataFrame(
        [
            _row(VariationID="low", NumberSubmitters=1),
            _row(VariationID="high", NumberSubmitters=9),
            _row(VariationID="mid", NumberSubmitters=4),
        ]
    )
    deduped = cl.deduplicate_variants(df)
    assert len(deduped) == 1
    assert deduped.iloc[0]["VariationID"] == "high"
    assert deduped.iloc[0]["NumberSubmitters"] == 9


def test_deduplication_distinguishes_loci() -> None:
    """Variants at different loci are not collapsed together."""
    df = pd.DataFrame(
        [
            _row(VariationID="1", Start="100", Stop="100"),
            _row(VariationID="2", Start="200", Stop="200"),
            _row(VariationID="3", GeneSymbol="BRCA1", Start="100", Stop="100"),
        ]
    )
    assert len(cl.deduplicate_variants(df)) == 3


# --- Edge cases --------------------------------------------------------------


def test_empty_gene_symbol_is_dropped() -> None:
    """Empty and ``"-"`` gene symbols are excluded."""
    df = pd.DataFrame(
        [
            _row(VariationID="1", GeneSymbol=""),
            _row(VariationID="2", GeneSymbol="-"),
            _row(VariationID="3", GeneSymbol="EGFR"),
        ]
    )
    filtered = cl.filter_variants(df)
    assert filtered["VariationID"].tolist() == ["3"]


def test_missing_columns_raise_keyerror() -> None:
    """Filtering a frame missing required columns fails loudly."""
    df = pd.DataFrame({"VariationID": ["1"], "GeneSymbol": ["TP53"]})
    with pytest.raises(KeyError, match="missing required ClinVar columns"):
        cl.filter_variants(df)


def test_non_numeric_submitter_count_is_coerced() -> None:
    """Blank/garbage submitter counts coerce to 0 rather than crashing."""
    df = pd.DataFrame(
        [
            _row(VariationID="blank", NumberSubmitters=""),
            _row(VariationID="real", NumberSubmitters=5),
        ]
    )
    deduped = cl.deduplicate_variants(cl.filter_variants(df))
    assert deduped.iloc[0]["VariationID"] == "real"


def test_full_pipeline_round_trip(tmp_path: Path, synthetic_df: pd.DataFrame) -> None:
    """End-to-end processing of a TSV produces a labelled, deduplicated Parquet."""
    raw = tmp_path / "variant_summary.txt.gz"
    synthetic_df.to_csv(raw, sep="\t", index=False, compression="gzip")

    out = tmp_path / "clinvar_processed.parquet"
    result = cl.process_clinvar(raw_path=raw, out_path=out)

    assert out.is_file()
    reloaded = pd.read_parquet(out)
    assert "label" in reloaded.columns
    assert len(reloaded) == len(result) == 3
    assert set(reloaded["label"]).issubset(set(cl.CLASS_NAMES))


# =============================================================================
# cBioPortal client (src.data.cbioportal_client)
# =============================================================================
#
# Every test mocks the HTTP layer (no real network, per CLAUDE.md). The client's
# ``session.get`` is replaced with a MagicMock so responses are fully synthetic.


def _response(json_data: object = None, status_code: int = 200) -> mock.MagicMock:
    """Build a fake ``requests.Response`` returning ``json_data``.

    A ``status_code >= 400`` makes ``raise_for_status`` raise an
    :class:`requests.HTTPError` carrying the response (so the client's 404
    handling can inspect it), mirroring real ``requests`` behaviour.
    """
    response = mock.MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None
    return response


def _client_with_get(side_effect: object) -> cbio.CBioPortalClient:
    """Return a client whose ``session.get`` is mocked with ``side_effect``.

    ``side_effect`` may be a single response (used for every call) or a list of
    responses (consumed in order).
    """
    client = cbio.CBioPortalClient()
    if isinstance(side_effect, list):
        client.session.get = mock.MagicMock(side_effect=side_effect)
    else:
        client.session.get = mock.MagicMock(return_value=side_effect)
    return client


def _mutation_record(**overrides: object) -> dict[str, object]:
    """A single DETAILED-projection mutation record from the API."""
    base: dict[str, object] = {
        "sampleId": "TCGA-AA-0001-01",
        "patientId": "TCGA-AA-0001",
        "entrezGeneId": 7157,
        "gene": {"hugoGeneSymbol": "TP53", "entrezGeneId": 7157},
        "mutationType": "Missense_Mutation",
        "variantType": "SNP",
        "proteinChange": "R175H",
        "chr": "17",
        "startPosition": 7676154,
        "endPosition": 7676154,
        "referenceAllele": "C",
        "variantAllele": "T",
    }
    base.update(overrides)
    return base


# --- Session configuration ---------------------------------------------------


def test_client_configures_retry_adapter() -> None:
    """The mounted adapter carries 3 retries with backoff_factor 0.5."""
    client = cbio.CBioPortalClient()
    retries = client.session.get_adapter("https://www.cbioportal.org/api").max_retries
    assert retries.total == 3
    assert retries.backoff_factor == 0.5
    assert client.session.headers["User-Agent"] == cbio.USER_AGENT


def test_throttle_sleeps_to_respect_rate() -> None:
    """A request issued too soon after the previous one is delayed."""
    client = cbio.CBioPortalClient(max_requests_per_second=10)
    with mock.patch(
        "src.data.cbioportal_client.time.monotonic", side_effect=[100.0, 100.0]
    ), mock.patch("src.data.cbioportal_client.time.sleep") as sleep:
        client._last_request_time = 100.0
        client._throttle()
    sleep.assert_called_once()
    assert sleep.call_args.args[0] == pytest.approx(0.1, abs=1e-9)


# --- Pagination --------------------------------------------------------------


def test_pagination_walks_pages_until_short_page() -> None:
    """Pages are fetched until one returns fewer than ``page_size`` records."""
    pages = [
        _response([{"i": 1}, {"i": 2}]),  # full page
        _response([{"i": 3}, {"i": 4}]),  # full page
        _response([{"i": 5}]),            # short page -> stop
    ]
    client = _client_with_get(pages)
    records = client._get_paginated("/some/path", {"projection": "DETAILED"}, page_size=2)

    assert [r["i"] for r in records] == [1, 2, 3, 4, 5]
    assert client.session.get.call_count == 3
    # pageNumber increments 0, 1, 2 across the three calls.
    page_numbers = [
        call.kwargs["params"]["pageNumber"] for call in client.session.get.call_args_list
    ]
    assert page_numbers == [0, 1, 2]


def test_get_mutations_builds_typed_dataframe() -> None:
    """Mutation records map onto the documented columns, single page."""
    records = [
        _mutation_record(),
        _mutation_record(
            sampleId="TCGA-AA-0002-01",
            gene={"hugoGeneSymbol": "KRAS", "entrezGeneId": 3845},
            entrezGeneId=3845,
            mutationType="Nonsense_Mutation",
        ),
    ]
    client = _client_with_get(_response(records))
    df = client.get_mutations("brca_tcga")

    assert list(df.columns) == list(cbio.MUTATION_COLUMNS)
    assert df.loc[0, "gene_symbol"] == "TP53"
    assert df.loc[0, "entrez_gene_id"] == 7157
    # cBioPortal stores MAF Variant_Classification in mutationType.
    assert df.loc[0, "variant_classification"] == "Missense_Mutation"
    assert df.loc[1, "gene_symbol"] == "KRAS"
    # The sample-list id and DETAILED projection are passed through.
    params = client.session.get.call_args_list[0].kwargs["params"]
    assert params["sampleListId"] == "brca_tcga_all"
    assert params["projection"] == "DETAILED"


# --- Graceful 404 handling ---------------------------------------------------


def test_get_mutations_handles_missing_profile() -> None:
    """A 404 on the mutation profile yields an empty, correctly-typed frame."""
    client = _client_with_get(_response(status_code=404))
    df = client.get_mutations("nonexistent_study")
    assert df.empty
    assert list(df.columns) == list(cbio.MUTATION_COLUMNS)


def test_get_expression_tries_all_suffixes_then_gives_up() -> None:
    """When every candidate profile 404s, an empty DataFrame is returned."""
    client = _client_with_get(_response(status_code=404))
    df = client.get_expression("study_without_rnaseq")
    assert df.empty
    # One request per expression suffix tried.
    assert client.session.get.call_count == len(cbio.EXPRESSION_SUFFIXES)


def test_get_expression_falls_back_to_second_suffix() -> None:
    """A 404 on the first suffix falls through to the next, which succeeds."""
    records = [
        {"sampleId": "S1", "gene": {"hugoGeneSymbol": "EGFR"}, "value": 5.5},
        {"sampleId": "S2", "gene": {"hugoGeneSymbol": "EGFR"}, "value": 1.2},
    ]
    client = _client_with_get([_response(status_code=404), _response(records)])
    df = client.get_expression("brca_tcga")

    assert client.session.get.call_count == 2
    # Second suffix in EXPRESSION_SUFFIXES was the one that resolved.
    used_profile = client.session.get.call_args_list[1].args[0]
    assert used_profile.endswith(f"brca_tcga{cbio.EXPRESSION_SUFFIXES[1]}/molecular-data")
    assert set(df["EGFR"]) == {5.5, 1.2}


def test_non_404_http_error_propagates() -> None:
    """A 500 (after retries) is not swallowed by the 404 handler."""
    client = _client_with_get(_response(status_code=500))
    with pytest.raises(requests.HTTPError):
        client.get_mutations("brca_tcga")


# --- DataFrame pivoting ------------------------------------------------------


def test_pivot_molecular_to_wide_matrix() -> None:
    """Long molecular-data records pivot to a samples × genes matrix."""
    records = [
        {"sampleId": "S1", "gene": {"hugoGeneSymbol": "TP53"}, "value": 1.0},
        {"sampleId": "S1", "gene": {"hugoGeneSymbol": "EGFR"}, "value": 2.0},
        {"sampleId": "S2", "gene": {"hugoGeneSymbol": "TP53"}, "value": 3.0},
        {"sampleId": "S2", "gene": {"hugoGeneSymbol": "EGFR"}, "value": 4.0},
    ]
    wide = cbio.CBioPortalClient._pivot_molecular(records)

    assert list(wide.columns) == ["sample_id", "EGFR", "TP53"]
    assert wide.shape == (2, 3)
    row = wide.set_index("sample_id").loc["S1"]
    assert row["TP53"] == 1.0 and row["EGFR"] == 2.0


def test_pivot_molecular_falls_back_to_entrez_id() -> None:
    """Records lacking a gene symbol key on the Entrez id instead."""
    records = [{"sampleId": "S1", "entrezGeneId": 7157, "value": 9.0}]
    wide = cbio.CBioPortalClient._pivot_molecular(records)
    assert "7157" in wide.columns
    assert wide.loc[0, "7157"] == 9.0


def test_pivot_molecular_empty_input() -> None:
    """No records yields an empty DataFrame, not an error."""
    assert cbio.CBioPortalClient._pivot_molecular([]).empty


def test_get_cnv_returns_discrete_values() -> None:
    """GISTIC2 discrete values survive the pivot as numeric columns."""
    records = [
        {"sampleId": "S1", "gene": {"hugoGeneSymbol": "MYC"}, "value": -2},
        {"sampleId": "S2", "gene": {"hugoGeneSymbol": "MYC"}, "value": 1},
    ]
    client = _client_with_get(_response(records))
    df = client.get_cnv("brca_tcga")
    assert set(df["MYC"]) == {-2, 1}


# --- Clinical ----------------------------------------------------------------


def test_build_clinical_df_pivots_and_joins() -> None:
    """Sample- and patient-level attributes are pivoted, joined and renamed."""
    sample_records = [
        {"sampleId": "S1", "patientId": "P1", "clinicalAttributeId": "CANCER_TYPE",
         "value": "Breast"},
        {"sampleId": "S1", "patientId": "P1", "clinicalAttributeId": "GRADE",
         "value": "G2"},
    ]
    patient_records = [
        {"patientId": "P1", "clinicalAttributeId": "SEX", "value": "Female"},
        {"patientId": "P1", "clinicalAttributeId": "AGE", "value": "61"},
        {"patientId": "P1", "clinicalAttributeId": "OS_MONTHS", "value": "24.5"},
        {"patientId": "P1", "clinicalAttributeId": "OS_STATUS", "value": "1:DECEASED"},
    ]
    df = cbio.CBioPortalClient._build_clinical_df(sample_records, patient_records)

    assert list(df.columns) == list(cbio.CLINICAL_COLUMNS)
    row = df.iloc[0]
    assert row["sample_id"] == "S1" and row["patient_id"] == "P1"
    assert row["cancer_type"] == "Breast" and row["grade"] == "G2"
    assert row["sex"] == "Female" and row["os_status"] == "1:DECEASED"
    # Numeric columns are coerced.
    assert row["age"] == 61.0 and row["os_months"] == 24.5


def test_build_clinical_df_empty() -> None:
    """No clinical records yields an empty, correctly-typed frame."""
    df = cbio.CBioPortalClient._build_clinical_df([], [])
    assert df.empty
    assert list(df.columns) == list(cbio.CLINICAL_COLUMNS)


def test_get_clinical_end_to_end() -> None:
    """get_clinical fetches both slices and standardises them."""
    sample_records = [
        {"sampleId": "S1", "patientId": "P1", "clinicalAttributeId": "CANCER_TYPE",
         "value": "Breast"},
    ]
    patient_records = [
        {"patientId": "P1", "clinicalAttributeId": "AGE", "value": "61"},
    ]
    client = _client_with_get([_response(sample_records), _response(patient_records)])
    df = client.get_clinical("brca_tcga")
    assert df.iloc[0]["cancer_type"] == "Breast"
    assert df.iloc[0]["age"] == 61.0


# --- Orchestration -----------------------------------------------------------


def test_download_study_writes_parquet(tmp_path: Path) -> None:
    """download_study saves a parquet per non-empty modality and returns frames."""
    client = cbio.CBioPortalClient()
    client.get_mutations = mock.MagicMock(  # type: ignore[method-assign]
        return_value=pd.DataFrame({"gene_symbol": ["TP53"]})
    )
    client.get_expression = mock.MagicMock(  # type: ignore[method-assign]
        return_value=pd.DataFrame({"sample_id": ["S1"], "TP53": [1.0]})
    )
    empty = pd.DataFrame()
    client.get_methylation = mock.MagicMock(return_value=empty)  # type: ignore[method-assign]
    client.get_cnv = mock.MagicMock(return_value=empty)  # type: ignore[method-assign]
    client.get_clinical = mock.MagicMock(  # type: ignore[method-assign]
        return_value=pd.DataFrame({"sample_id": ["S1"]})
    )

    frames = client.download_study("brca_tcga", tmp_path)

    study_dir = tmp_path / "brca_tcga"
    assert (study_dir / "mutations.parquet").is_file()
    assert (study_dir / "expression.parquet").is_file()
    assert (study_dir / "clinical.parquet").is_file()
    # Empty modalities are not written.
    assert not (study_dir / "methylation.parquet").exists()
    assert not (study_dir / "cnv.parquet").exists()
    assert set(frames) == {"mutations", "expression", "methylation", "cnv", "clinical"}


def test_download_study_survives_modality_failure(tmp_path: Path) -> None:
    """A modality that raises is logged and recorded as an empty frame."""
    client = cbio.CBioPortalClient()
    client.get_mutations = mock.MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("boom")
    )
    for name in ("get_expression", "get_methylation", "get_cnv", "get_clinical"):
        setattr(client, name, mock.MagicMock(return_value=pd.DataFrame()))

    frames = client.download_study("brca_tcga", tmp_path)
    assert frames["mutations"].empty  # failure became an empty frame, no crash


def test_download_all_studies_writes_manifest(tmp_path: Path) -> None:
    """A manifest records per-modality shapes and survives per-study failure."""
    client = cbio.CBioPortalClient()

    def fake_download(study_id: str, output_dir: Path) -> dict[str, pd.DataFrame]:
        if study_id == "bad_study":
            raise RuntimeError("network down")
        return {
            "mutations": pd.DataFrame({"gene_symbol": ["TP53", "KRAS"]}),
            "expression": pd.DataFrame(),
        }

    client.download_study = mock.MagicMock(side_effect=fake_download)  # type: ignore[method-assign]
    manifest = client.download_all_studies(["good_study", "bad_study"], tmp_path)

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["good_study"]["mutations"] == {"rows": 2, "cols": 1}
    assert "error" in on_disk["bad_study"]
    assert manifest == on_disk
