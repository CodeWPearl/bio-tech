"""Feature extraction service for the prediction API."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from api.schemas import PredictionRequest
from src.models.full_model import MODALITY_NAMES

logger = logging.getLogger(__name__)

KNOWN_CANCER_DRIVERS: set[str] = {
    "TP53", "KRAS", "PIK3CA", "PTEN", "APC", "BRCA1", "BRCA2", "EGFR",
    "BRAF", "RB1", "MYC", "ERBB2", "CDH1", "CDKN2A", "SMAD4", "VHL",
    "NF1", "NF2", "RET", "MET", "ALK", "KIT", "PDGFRA", "FGFR2",
    "FGFR3", "IDH1", "IDH2", "NRAS", "HRAS", "ABL1", "JAK2", "NOTCH1",
    "FBXW7", "CTNNB1", "MAP2K1", "ARID1A", "SETD2", "KDM6A", "KMT2D",
    "CREBBP", "EP300", "SMARCA4", "STAG2", "ATM", "ATR", "CHEK2",
    "PALB2", "RAD51C", "RAD51D", "MLH1", "MSH2", "MSH6", "PMS2",
}

MUTATION_TYPE_ENCODING: dict[str, int] = {
    "Missense_Mutation": 0,
    "Nonsense_Mutation": 1,
    "Frame_Shift_Del": 2,
    "Frame_Shift_Ins": 3,
    "Splice_Site": 4,
    "In_Frame_Del": 5,
    "In_Frame_Ins": 6,
    "Silent": 7,
    "Nonstop_Mutation": 8,
    "Translation_Start_Site": 9,
}

CHROMOSOME_ENCODING: dict[str, int] = {
    str(i): i for i in range(1, 23)
} | {"X": 23, "Y": 24, "MT": 25}

NUCLEOTIDE_ENCODING: dict[str, list[float]] = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "N": [0.25, 0.25, 0.25, 0.25],
    "-": [0.0, 0.0, 0.0, 0.0],
}


class FeatureService:
    """Extracts feature tensors from API prediction requests.

    Args:
        config: Project configuration with model input dimensions.
        pipeline_path: Optional path to a saved feature pipeline pickle.
    """

    def __init__(
        self,
        config: Any,
        pipeline_path: str | Path | None = None,
    ) -> None:
        model_cfg = config.model
        self.dims = {
            "mutation": model_cfg.get("mutation_input_dim", 42),
            "expression": model_cfg.get("expression_input_dim", 2000),
            "methylation": model_cfg.get("methylation_input_dim", 2000),
            "cnv": model_cfg.get("cnv_input_dim", 200),
            "clinical": model_cfg.get("clinical_input_dim", 32),
        }
        self.pipeline: Any | None = None
        self.known_genes: set[str] = set(KNOWN_CANCER_DRIVERS)

        if pipeline_path is not None:
            pp = Path(pipeline_path)
            if pp.is_file():
                with pp.open("rb") as fh:
                    self.pipeline = pickle.load(fh)
                if hasattr(self.pipeline, "known_genes"):
                    self.known_genes = set(self.pipeline.known_genes)
                logger.info("Feature pipeline loaded from %s", pp)

    def is_known_gene(self, gene_symbol: str) -> bool:
        """Check if a gene symbol is in the known gene list.

        Args:
            gene_symbol: HUGO gene symbol.

        Returns:
            True if the gene is known.
        """
        return gene_symbol.upper() in self.known_genes

    def extract_features(
        self, request: PredictionRequest,
    ) -> dict[str, torch.Tensor]:
        """Convert a prediction request into model-ready feature tensors.

        Args:
            request: Validated prediction request.

        Returns:
            Dict of tensors matching model input format with modality_mask.
        """
        if self.pipeline is not None and hasattr(self.pipeline, "transform"):
            return self._extract_with_pipeline(request)
        return self._extract_manual(request)

    def _extract_manual(
        self, request: PredictionRequest,
    ) -> dict[str, torch.Tensor]:
        """Build feature tensors from request fields without a saved pipeline."""
        batch: dict[str, torch.Tensor] = {}
        mask: list[bool] = []

        mutation_features = self._encode_mutation_features(request)
        batch["mutation"] = torch.tensor(
            mutation_features, dtype=torch.float32,
        ).unsqueeze(0)
        mask.append(True)

        for modality in ["expression", "methylation", "cnv", "clinical"]:
            batch[modality] = torch.zeros(
                1, self.dims[modality], dtype=torch.float32,
            )
            mask.append(False)

        batch["modality_mask"] = torch.tensor([mask], dtype=torch.bool)
        return batch

    def _encode_mutation_features(
        self, request: PredictionRequest,
    ) -> np.ndarray:
        """Encode mutation-level features from the request.

        Args:
            request: Prediction request with variant details.

        Returns:
            Float array of length mutation_input_dim.
        """
        dim = self.dims["mutation"]
        features = np.zeros(dim, dtype=np.float32)

        mt = request.mutation_type
        if mt in MUTATION_TYPE_ENCODING:
            idx = MUTATION_TYPE_ENCODING[mt]
            if idx < dim:
                features[idx] = 1.0

        n_types = len(MUTATION_TYPE_ENCODING)
        chrom = request.chromosome
        if chrom in CHROMOSOME_ENCODING:
            chrom_idx = n_types + CHROMOSOME_ENCODING[chrom]
            if chrom_idx < dim:
                features[chrom_idx] = 1.0

        pos_idx = n_types + 26
        if pos_idx < dim:
            features[pos_idx] = np.log1p(request.start_position) / 25.0

        ref_start = pos_idx + 1
        ref_enc = NUCLEOTIDE_ENCODING.get(
            request.reference_allele[0], [0.0, 0.0, 0.0, 0.0],
        )
        for i, val in enumerate(ref_enc):
            if ref_start + i < dim:
                features[ref_start + i] = val

        var_start = ref_start + 4
        var_enc = NUCLEOTIDE_ENCODING.get(
            request.variant_allele[0], [0.0, 0.0, 0.0, 0.0],
        )
        for i, val in enumerate(var_enc):
            if var_start + i < dim:
                features[var_start + i] = val

        return features

    def _extract_with_pipeline(
        self, request: PredictionRequest,
    ) -> dict[str, torch.Tensor]:
        """Extract features using a saved sklearn-like pipeline."""
        raw = {
            "gene_symbol": request.gene_symbol,
            "mutation_type": request.mutation_type,
            "chromosome": request.chromosome,
            "start_position": request.start_position,
            "reference_allele": request.reference_allele,
            "variant_allele": request.variant_allele,
            "protein_change": request.protein_change or "",
            "cancer_type": request.cancer_type or "",
        }
        transformed = self.pipeline.transform(raw)

        batch: dict[str, torch.Tensor] = {}
        mask: list[bool] = []
        offset = 0

        for name in MODALITY_NAMES:
            dim = self.dims[name]
            if offset + dim <= len(transformed):
                arr = transformed[offset: offset + dim]
                batch[name] = torch.tensor(
                    arr, dtype=torch.float32,
                ).unsqueeze(0)
                mask.append(True)
                offset += dim
            else:
                batch[name] = torch.zeros(
                    1, dim, dtype=torch.float32,
                )
                mask.append(False)

        batch["modality_mask"] = torch.tensor([mask], dtype=torch.bool)
        return batch

    def get_biological_context(
        self, gene_symbol: str, mutation_type: str,
    ) -> dict[str, Any]:
        """Look up biological context for a gene/mutation.

        Args:
            gene_symbol: HUGO gene symbol.
            mutation_type: Mutation type string.

        Returns:
            Dict with gene annotation info.
        """
        gene_upper = gene_symbol.upper()
        is_driver = gene_upper in KNOWN_CANCER_DRIVERS

        cosmic_info = None
        if is_driver:
            cosmic_info = f"{gene_upper} is listed in the COSMIC Cancer Gene Census"

        type_descriptions = {
            "Missense_Mutation": "Single nucleotide change causing amino acid substitution",
            "Nonsense_Mutation": "Single nucleotide change introducing a premature stop codon",
            "Frame_Shift_Del": "Deletion causing a reading frame shift",
            "Frame_Shift_Ins": "Insertion causing a reading frame shift",
            "Splice_Site": "Mutation affecting mRNA splicing",
            "In_Frame_Del": "Deletion that maintains the reading frame",
            "In_Frame_Ins": "Insertion that maintains the reading frame",
            "Silent": "Synonymous mutation with no amino acid change",
            "Nonstop_Mutation": "Mutation removing the stop codon",
            "Translation_Start_Site": "Mutation at the translation initiation site",
        }

        return {
            "gene_symbol": gene_upper,
            "is_known_cancer_driver": is_driver,
            "cosmic_census_info": cosmic_info,
            "clinvar_entries": 0,
            "variant_type_description": type_descriptions.get(
                mutation_type, f"Variant of type: {mutation_type}",
            ),
        }
