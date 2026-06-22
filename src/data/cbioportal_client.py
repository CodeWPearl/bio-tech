"""cBioPortal REST API client for multi-omics TCGA data.

This module fetches the molecular and clinical data that pairs with the ClinVar
pathogenicity labels (see :mod:`src.data.clinvar_loader`). It talks to the public
cBioPortal REST API (https://www.cbioportal.org/api), which needs no
authentication for public studies (see ``CLAUDE.md`` — "ALL FREE, NO
AUTHENTICATION WALLS").

The single public entry point is :class:`CBioPortalClient`, which exposes one
method per data modality plus orchestration helpers:

* :meth:`~CBioPortalClient.get_mutations` — somatic mutation calls (paginated).
* :meth:`~CBioPortalClient.get_expression` — RNA-seq expression matrix.
* :meth:`~CBioPortalClient.get_methylation` — methylation beta-value matrix.
* :meth:`~CBioPortalClient.get_cnv` — GISTIC2 discrete copy-number matrix.
* :meth:`~CBioPortalClient.get_clinical` — patient/sample clinical attributes.
* :meth:`~CBioPortalClient.download_study` — fetch all five and save Parquet.
* :meth:`~CBioPortalClient.download_all_studies` — loop with a JSON manifest.

Design notes:

* The client self-throttles to at most ``max_requests_per_second`` requests and
  mounts a :class:`urllib3.util.retry.Retry` adapter for transient server errors
  with exponential backoff.
* Molecular data is fetched per profile with a server-side ``sampleListId`` so the
  whole study is returned in one request (the equivalent of cBioPortal's
  ``fetchAllMolecularDataInMolecularProfile`` operation) rather than gene-by-gene.
* Not every study exposes every modality, and studies use different profile-id
  suffixes. Each ``get_*`` method tries a list of known suffixes and, when a
  profile is absent, catches the HTTP 404 and returns an empty DataFrame instead
  of crashing (per the task's API notes).

Per project standards we log via :mod:`logging` rather than ``print()`` and put
type hints on every signature.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

#: Default base URL of the public cBioPortal REST API.
DEFAULT_BASE_URL: str = "https://www.cbioportal.org/api"

#: User-Agent sent with every request so the API can attribute traffic.
USER_AGENT: str = "cancer-mutation-pathogenicity/0.1 (cBioPortal Python client)"

#: Maximum records the API returns per page; used as the pagination page size.
MAX_PAGE_SIZE: int = 10_000

#: Profile-id suffixes tried (in order) for each molecular modality. Studies vary
#: in which assays they ran, so the first existing profile wins.
EXPRESSION_SUFFIXES: tuple[str, ...] = ("_rna_seq_v2_mrna", "_rna_seq_mrna", "_mrna")
METHYLATION_SUFFIXES: tuple[str, ...] = ("_methylation_hm450", "_methylation_hm27")
CNV_SUFFIXES: tuple[str, ...] = ("_gistic", "_cna", "_linear_CNA")

#: Ordered output columns for the mutation table.
MUTATION_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "patient_id",
    "gene_symbol",
    "entrez_gene_id",
    "mutation_type",
    "variant_type",
    "protein_change",
    "chromosome",
    "start_position",
    "end_position",
    "reference_allele",
    "variant_allele",
    "variant_classification",
)

#: Standardised clinical columns mapped to the cBioPortal attribute ids that may
#: supply them (first match wins; ids differ across studies).
CLINICAL_ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "cancer_type": ("CANCER_TYPE", "CANCER_TYPE_DETAILED", "TUMOR_TYPE"),
    "age": ("AGE", "AGE_AT_DIAGNOSIS"),
    "sex": ("SEX", "GENDER"),
    "stage": ("AJCC_PATHOLOGIC_TUMOR_STAGE", "TUMOR_STAGE", "PATH_STAGE", "STAGE"),
    "grade": ("GRADE", "TUMOR_GRADE", "NEOPLASM_HISTOLOGIC_GRADE"),
    "os_months": ("OS_MONTHS",),
    "os_status": ("OS_STATUS",),
}

#: Final column order for the clinical table.
CLINICAL_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "patient_id",
    *CLINICAL_ATTRIBUTE_ALIASES.keys(),
)

#: Clinical columns coerced to numeric in the output.
NUMERIC_CLINICAL_COLUMNS: frozenset[str] = frozenset({"age", "os_months"})


def _is_status(exc: requests.HTTPError, status: int) -> bool:
    """Return ``True`` if ``exc`` carries an HTTP response with the given status.

    Args:
        exc: The :class:`requests.HTTPError` raised by ``raise_for_status``.
        status: The HTTP status code to test for (e.g. ``404``).

    Returns:
        ``True`` when the exception's response status matches ``status``.
    """
    return exc.response is not None and exc.response.status_code == status


class CBioPortalClient:
    """A rate-limited, retrying client for the cBioPortal REST API.

    The client owns a :class:`requests.Session` configured with retry/backoff and
    a User-Agent header, and enforces a minimum interval between requests so it
    never exceeds the configured request rate.

    Attributes:
        base_url: API root with no trailing slash.
        session: The underlying configured :class:`requests.Session`.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        max_requests_per_second: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: int = 60,
    ) -> None:
        """Initialise the client and its retrying HTTP session.

        Args:
            base_url: Root URL of the cBioPortal API.
            max_requests_per_second: Upper bound on outgoing request rate; the
                client sleeps as needed to stay at or below it.
            max_retries: Number of automatic retries for transient server errors.
            backoff_factor: Exponential backoff factor passed to urllib3's
                :class:`~urllib3.util.retry.Retry`.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._min_interval = 1.0 / max_requests_per_second if max_requests_per_second else 0.0
        self._last_request_time = 0.0

        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "application/json"}
        )

    # --- Low-level HTTP ------------------------------------------------------

    def _throttle(self) -> None:
        """Sleep just long enough to honour the configured request rate."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _get_json(self, path: str, params: dict[str, object] | None = None) -> object:
        """Perform a throttled GET and return the parsed JSON body.

        Args:
            path: API path beginning with ``/`` (appended to :attr:`base_url`).
            params: Optional query-string parameters.

        Returns:
            The decoded JSON payload (typically a ``list`` of records).

        Raises:
            requests.HTTPError: If the server returns an unsuccessful status.
        """
        self._throttle()
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _get_paginated(
        self,
        path: str,
        params: dict[str, object],
        page_size: int = MAX_PAGE_SIZE,
    ) -> list[dict[str, object]]:
        """Fetch every page of a list endpoint and concatenate the records.

        The API caps each response at ``page_size`` records; this requests
        successive ``pageNumber`` values until a short (final) page is returned.

        Args:
            path: API path of the list endpoint.
            params: Base query parameters (page controls are added per request).
            page_size: Records requested per page.

        Returns:
            The concatenated list of record dictionaries across all pages.
        """
        records: list[dict[str, object]] = []
        page_number = 0
        while True:
            page_params = {**params, "pageSize": page_size, "pageNumber": page_number}
            batch = self._get_json(path, page_params)
            if not isinstance(batch, list) or not batch:
                break
            records.extend(batch)
            if len(batch) < page_size:
                break
            page_number += 1
        return records

    # --- Mutations -----------------------------------------------------------

    def get_mutations(self, study_id: str) -> pd.DataFrame:
        """Fetch all somatic mutations for ``study_id`` as a tidy DataFrame.

        Args:
            study_id: cBioPortal study identifier (e.g.
                ``brca_tcga_pan_can_atlas_2018``).

        Returns:
            One row per mutation call with the columns in :data:`MUTATION_COLUMNS`.
            An empty DataFrame (with those columns) is returned when the study has
            no mutation profile.
        """
        profile_id = f"{study_id}_mutations"
        path = f"/molecular-profiles/{profile_id}/mutations"
        params = {"sampleListId": f"{study_id}_all", "projection": "DETAILED"}
        try:
            records = self._get_paginated(path, params)
        except requests.HTTPError as exc:
            if _is_status(exc, 404):
                logger.warning("No mutation profile for study %s; skipping", study_id)
                return pd.DataFrame(columns=list(MUTATION_COLUMNS))
            raise
        logger.info("Fetched %d mutation records for %s", len(records), study_id)
        return self._build_mutations_df(records)

    @staticmethod
    def _build_mutations_df(records: list[dict[str, object]]) -> pd.DataFrame:
        """Flatten raw mutation records into the :data:`MUTATION_COLUMNS` schema.

        Args:
            records: Mutation objects as returned by the API (DETAILED projection,
                with a nested ``gene`` object).

        Returns:
            A DataFrame with exactly :data:`MUTATION_COLUMNS`.
        """
        if not records:
            return pd.DataFrame(columns=list(MUTATION_COLUMNS))

        rows: list[dict[str, object]] = []
        for record in records:
            gene = record.get("gene") or {}
            gene = gene if isinstance(gene, dict) else {}
            # cBioPortal stores the MAF "Variant_Classification" in mutationType.
            mutation_type = record.get("mutationType")
            rows.append(
                {
                    "sample_id": record.get("sampleId"),
                    "patient_id": record.get("patientId"),
                    "gene_symbol": gene.get("hugoGeneSymbol"),
                    "entrez_gene_id": record.get("entrezGeneId")
                    or gene.get("entrezGeneId"),
                    "mutation_type": mutation_type,
                    "variant_type": record.get("variantType"),
                    "protein_change": record.get("proteinChange"),
                    "chromosome": record.get("chr"),
                    "start_position": record.get("startPosition"),
                    "end_position": record.get("endPosition"),
                    "reference_allele": record.get("referenceAllele"),
                    "variant_allele": record.get("variantAllele"),
                    "variant_classification": mutation_type,
                }
            )
        return pd.DataFrame(rows, columns=list(MUTATION_COLUMNS))

    # --- Molecular matrices (expression / methylation / CNV) -----------------

    def get_expression(self, study_id: str) -> pd.DataFrame:
        """Fetch the RNA-seq expression matrix (samples × genes) for ``study_id``.

        Args:
            study_id: cBioPortal study identifier.

        Returns:
            Wide DataFrame: one row per sample, one column per gene, RSEM
            normalized expression values. Empty if no expression profile exists.
        """
        return self._get_molecular_matrix(study_id, EXPRESSION_SUFFIXES, "expression")

    def get_methylation(self, study_id: str) -> pd.DataFrame:
        """Fetch the methylation beta-value matrix (samples × genes).

        Args:
            study_id: cBioPortal study identifier.

        Returns:
            Wide DataFrame of methylation beta values, or empty if absent.
        """
        return self._get_molecular_matrix(study_id, METHYLATION_SUFFIXES, "methylation")

    def get_cnv(self, study_id: str) -> pd.DataFrame:
        """Fetch the GISTIC2 discrete copy-number matrix (samples × genes).

        Args:
            study_id: cBioPortal study identifier.

        Returns:
            Wide DataFrame of GISTIC2 values (-2, -1, 0, 1, 2), or empty if absent.
        """
        return self._get_molecular_matrix(study_id, CNV_SUFFIXES, "cnv")

    def _get_molecular_matrix(
        self, study_id: str, suffixes: tuple[str, ...], label: str
    ) -> pd.DataFrame:
        """Resolve a molecular profile by suffix and return its pivoted matrix.

        Each candidate suffix is tried in turn; a 404 means that profile does not
        exist for the study, so the next suffix is attempted. The first profile
        that returns data is fetched whole (one request, server-side sample list)
        and pivoted to wide form.

        Args:
            study_id: cBioPortal study identifier.
            suffixes: Candidate profile-id suffixes to append to ``study_id``.
            label: Human-readable modality name used in log messages.

        Returns:
            Wide samples × genes DataFrame, or an empty DataFrame if no candidate
            profile exists.
        """
        params = {"sampleListId": f"{study_id}_all", "projection": "DETAILED"}
        for suffix in suffixes:
            profile_id = f"{study_id}{suffix}"
            path = f"/molecular-profiles/{profile_id}/molecular-data"
            try:
                records = self._get_json(path, params)
            except requests.HTTPError as exc:
                if _is_status(exc, 404):
                    logger.debug("Profile %s not found; trying next suffix", profile_id)
                    continue
                raise
            records = records if isinstance(records, list) else []
            logger.info(
                "Fetched %d %s records from profile %s", len(records), label, profile_id
            )
            return self._pivot_molecular(records)

        logger.warning(
            "No %s profile for study %s (tried suffixes %s)",
            label,
            study_id,
            list(suffixes),
        )
        return pd.DataFrame()

    @staticmethod
    def _pivot_molecular(records: list[dict[str, object]]) -> pd.DataFrame:
        """Pivot long molecular-data records into a samples × genes matrix.

        Args:
            records: Molecular-data objects, each with ``sampleId``, ``value`` and
                either a nested ``gene.hugoGeneSymbol`` or an ``entrezGeneId``.

        Returns:
            A DataFrame with a ``sample_id`` column followed by one numeric column
            per gene. Empty input yields an empty DataFrame.
        """
        if not records:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for record in records:
            gene = record.get("gene")
            symbol = gene.get("hugoGeneSymbol") if isinstance(gene, dict) else None
            rows.append(
                {
                    "sample_id": record.get("sampleId"),
                    "gene": symbol or str(record.get("entrezGeneId")),
                    "value": record.get("value"),
                }
            )

        long = pd.DataFrame(rows)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        wide = long.pivot_table(
            index="sample_id", columns="gene", values="value", aggfunc="first"
        )
        wide.columns.name = None
        return wide.reset_index()

    # --- Clinical ------------------------------------------------------------

    def get_clinical(self, study_id: str) -> pd.DataFrame:
        """Fetch and standardise sample- and patient-level clinical attributes.

        Sample- and patient-level clinical data live in separate slices of the
        API; both are fetched, pivoted to wide form, joined on ``patientId`` and
        reduced to the standardised columns in :data:`CLINICAL_COLUMNS`.

        Args:
            study_id: cBioPortal study identifier.

        Returns:
            One row per sample with the columns in :data:`CLINICAL_COLUMNS`. Empty
            (with those columns) if the study exposes no clinical data.
        """
        sample_records = self._get_clinical_data(study_id, "SAMPLE")
        patient_records = self._get_clinical_data(study_id, "PATIENT")
        return self._build_clinical_df(sample_records, patient_records)

    def _get_clinical_data(
        self, study_id: str, clinical_data_type: str
    ) -> list[dict[str, object]]:
        """Fetch one clinical-data slice (``SAMPLE`` or ``PATIENT``) for a study.

        Args:
            study_id: cBioPortal study identifier.
            clinical_data_type: Either ``"SAMPLE"`` or ``"PATIENT"``.

        Returns:
            Long-format clinical-data records, or ``[]`` if the slice is absent.
        """
        path = f"/studies/{study_id}/clinical-data"
        params = {"clinicalDataType": clinical_data_type, "projection": "DETAILED"}
        try:
            return self._get_paginated(path, params)
        except requests.HTTPError as exc:
            if _is_status(exc, 404):
                logger.warning(
                    "No %s clinical data for study %s; skipping",
                    clinical_data_type,
                    study_id,
                )
                return []
            raise

    @classmethod
    def _build_clinical_df(
        cls,
        sample_records: list[dict[str, object]],
        patient_records: list[dict[str, object]],
    ) -> pd.DataFrame:
        """Pivot, join and standardise clinical records into one tidy table.

        Args:
            sample_records: Long-format SAMPLE-level clinical records.
            patient_records: Long-format PATIENT-level clinical records.

        Returns:
            A DataFrame with exactly :data:`CLINICAL_COLUMNS`.
        """
        if not sample_records and not patient_records:
            return pd.DataFrame(columns=list(CLINICAL_COLUMNS))

        sample_wide = cls._pivot_clinical(sample_records, ["sampleId", "patientId"])
        patient_wide = cls._pivot_clinical(patient_records, ["patientId"])

        if not sample_wide.empty:
            merged = sample_wide
            if not patient_wide.empty and "patientId" in sample_wide.columns:
                merged = sample_wide.merge(
                    patient_wide, on="patientId", how="left", suffixes=("", "_patient")
                )
        else:
            merged = patient_wide

        out = pd.DataFrame()
        out["sample_id"] = (
            merged["sampleId"] if "sampleId" in merged.columns else pd.NA
        )
        out["patient_id"] = (
            merged["patientId"] if "patientId" in merged.columns else pd.NA
        )
        for column, aliases in CLINICAL_ATTRIBUTE_ALIASES.items():
            out[column] = cls._first_available(merged, aliases)

        for column in NUMERIC_CLINICAL_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        return out[list(CLINICAL_COLUMNS)]

    @staticmethod
    def _pivot_clinical(
        records: list[dict[str, object]], index_keys: list[str]
    ) -> pd.DataFrame:
        """Pivot long clinical records (attribute-per-row) into wide form.

        Args:
            records: Clinical-data records with ``clinicalAttributeId``/``value``.
            index_keys: Identifier columns to use as the pivot index (kept as
                columns in the result).

        Returns:
            A wide DataFrame: one row per ``index_keys`` combination, one column
            per clinical attribute id. Empty input yields an empty DataFrame.
        """
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records)
        wide = frame.pivot_table(
            index=index_keys,
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        wide.columns.name = None
        return wide.reset_index()

    @staticmethod
    def _first_available(frame: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
        """Return the first present alias column, else an all-NA column.

        Args:
            frame: The wide clinical DataFrame to search.
            aliases: Candidate column names in priority order.

        Returns:
            The matching column as a Series, or an all-NA Series aligned to
            ``frame`` if no alias is present.
        """
        for alias in aliases:
            if alias in frame.columns:
                return frame[alias]
        return pd.Series(pd.NA, index=frame.index, dtype="object")

    # --- Orchestration -------------------------------------------------------

    def download_study(
        self, study_id: str, output_dir: str | Path
    ) -> dict[str, pd.DataFrame]:
        """Download all five modalities for one study and save them as Parquet.

        Each modality is fetched independently; a failure in one (e.g. a transient
        error after retries) is logged and recorded as an empty DataFrame so the
        remaining modalities still download.

        Args:
            study_id: cBioPortal study identifier.
            output_dir: Root directory; files are written to
                ``output_dir/{study_id}/{modality}.parquet``.

        Returns:
            A mapping of modality name to its DataFrame.
        """
        study_dir = Path(output_dir) / study_id
        study_dir.mkdir(parents=True, exist_ok=True)

        fetchers = {
            "mutations": self.get_mutations,
            "expression": self.get_expression,
            "methylation": self.get_methylation,
            "cnv": self.get_cnv,
            "clinical": self.get_clinical,
        }

        frames: dict[str, pd.DataFrame] = {}
        for name, fetch in fetchers.items():
            try:
                frame = fetch(study_id)
            except Exception as exc:  # noqa: BLE001 - keep other modalities going
                logger.error("Failed to fetch %s for %s: %s", name, study_id, exc)
                frame = pd.DataFrame()
            frames[name] = frame

            if frame is None or frame.empty:
                logger.warning("No %s data for %s; nothing written", name, study_id)
                continue
            destination = study_dir / f"{name}.parquet"
            frame.to_parquet(destination, index=False)
            logger.info(
                "Saved %s for %s: %d rows x %d cols -> %s",
                name,
                study_id,
                frame.shape[0],
                frame.shape[1],
                destination,
            )
        return frames

    def download_all_studies(
        self, study_ids: list[str], output_dir: str | Path
    ) -> dict[str, object]:
        """Download several studies, logging failures and writing a manifest.

        Args:
            study_ids: cBioPortal study identifiers to download.
            output_dir: Root directory for per-study folders and ``manifest.json``.

        Returns:
            The manifest mapping each study id to either per-modality shapes or an
            ``{"error": ...}`` entry when the whole study failed.
        """
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, object] = {}
        for study_id in study_ids:
            try:
                frames = self.download_study(study_id, root)
            except Exception as exc:  # noqa: BLE001 - continue to the next study
                logger.error("Failed to download study %s: %s", study_id, exc)
                manifest[study_id] = {"error": str(exc)}
                continue
            manifest[study_id] = {
                name: {"rows": int(frame.shape[0]), "cols": int(frame.shape[1])}
                for name, frame in frames.items()
            }

        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Wrote download manifest to %s", manifest_path)
        return manifest
