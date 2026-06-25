"""Assembled pathogenicity prediction model.

Wires modality encoders, fusion module, and classification head into a single
end-to-end :class:`PathogenicityPredictor`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.base import BaseModel
from src.models.classifier import ClassificationHead
from src.models.encoders.clinical_encoder import ClinicalEncoder
from src.models.encoders.cnv_encoder import CNVFCEncoder
from src.models.encoders.expression_encoder import DenseAutoencoder
from src.models.encoders.methylation_encoder import MethylationDenseAutoencoder
from src.models.encoders.mutation_encoder import MutationEncoder
from src.models.fusion.attention_fusion import AttentionFusion
from src.models.fusion.cross_attention import CrossAttentionFusion
from src.models.fusion.early_fusion import EarlyFusion
from src.models.fusion.late_fusion import LateFusion
from src.models.fusion.transformer_fusion import TransformerFusion
from src.utils.config import Config

logger = logging.getLogger(__name__)

MODALITY_NAMES: list[str] = [
    "mutation", "expression", "methylation", "cnv", "clinical",
]


class PathogenicityPredictor(BaseModel):
    """End-to-end multi-omics pathogenicity predictor.

    Args:
        config: Project configuration object whose ``model`` section contains
            encoder dimensions, fusion type, and classifier settings.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        model_cfg = config.model
        self.fusion_type: str = model_cfg.fusion_type
        self.num_classes: int = model_cfg.num_classes
        self.fusion_dim: int = model_cfg.fusion_dim
        dropout: float = model_cfg.dropout

        disabled = model_cfg.get("disabled_modalities", None)
        self.disabled_modalities: set[str] = set(disabled) if disabled else set()

        # --- per-modality encoders -------------------------------------------
        self.encoders = nn.ModuleDict()
        self._modality_dims: dict[str, int] = {}

        mutation_input: int = model_cfg.get("mutation_input_dim", 42)
        self.encoders["mutation"] = MutationEncoder(
            input_dim=mutation_input,
            embed_dim=model_cfg.mutation_embed_dim,
            dropout=dropout,
        )
        self._modality_dims["mutation"] = model_cfg.mutation_embed_dim

        expression_input: int = model_cfg.get("expression_input_dim", 2000)
        self.encoders["expression"] = DenseAutoencoder(
            input_dim=expression_input,
            embed_dim=model_cfg.expression_embed_dim,
            dropout=dropout,
        )
        self._modality_dims["expression"] = model_cfg.expression_embed_dim

        methylation_input: int = model_cfg.get("methylation_input_dim", 2000)
        self.encoders["methylation"] = MethylationDenseAutoencoder(
            input_dim=methylation_input,
            embed_dim=model_cfg.methylation_embed_dim,
            dropout=dropout,
        )
        self._modality_dims["methylation"] = model_cfg.methylation_embed_dim

        cnv_input: int = model_cfg.get("cnv_input_dim", 200)
        self.encoders["cnv"] = CNVFCEncoder(
            input_dim=cnv_input,
            embed_dim=model_cfg.cnv_embed_dim,
            dropout=dropout,
        )
        self._modality_dims["cnv"] = model_cfg.cnv_embed_dim

        clinical_input: int = model_cfg.get("clinical_input_dim", 32)
        clinical_embed: int = model_cfg.get("clinical_embed_dim", 32)
        self.encoders["clinical"] = ClinicalEncoder(
            input_dim=clinical_input,
            embed_dim=clinical_embed,
            dropout=dropout,
        )
        self._modality_dims["clinical"] = clinical_embed

        # --- fusion module ---------------------------------------------------
        self._build_fusion(dropout)

        # --- classification head (skipped for late fusion) -------------------
        if self.fusion_type == "late":
            self.classifier: ClassificationHead | None = None
            concat_dim = sum(self._modality_dims[n] for n in MODALITY_NAMES)
            self.embedding_projection: nn.Sequential | None = nn.Sequential(
                nn.Linear(concat_dim, self.fusion_dim),
                nn.ReLU(),
            )
        else:
            self.classifier = ClassificationHead(
                fusion_dim=self.fusion_dim,
                num_classes=self.num_classes,
            )
            self.embedding_projection = None

        # --- load pretrained encoder weights (if configured) ----------------
        self._load_pretrained_encoders(config)

    def _load_pretrained_encoders(self, config: Config) -> None:
        """Load pretrained encoder weights if paths are set in config.

        Args:
            config: Project configuration; checks ``config.pretrain``
                for checkpoint paths.
        """
        pretrain_cfg = config.get("pretrain", None)
        if pretrain_cfg is None:
            return

        if isinstance(pretrain_cfg, dict):
            pretrain_cfg = Config(pretrain_cfg)

        freeze = bool(getattr(pretrain_cfg, "freeze_epochs", 0) > 0)

        expr_ae_path = getattr(pretrain_cfg, "expression_ae_path", None)
        if expr_ae_path and Path(expr_ae_path).is_file():
            encoder = self.encoders["expression"]
            if hasattr(encoder, "load_pretrained_weights"):
                encoder.load_pretrained_weights(expr_ae_path, freeze=freeze)
                logger.info("Loaded pretrained expression AE weights")

        meth_ae_path = getattr(pretrain_cfg, "methylation_ae_path", None)
        if meth_ae_path and Path(meth_ae_path).is_file():
            encoder = self.encoders["methylation"]
            if hasattr(encoder, "load_pretrained_weights"):
                encoder.load_pretrained_weights(meth_ae_path, freeze=freeze)
                logger.info("Loaded pretrained methylation AE weights")

    def _build_fusion(self, dropout: float) -> None:
        """Instantiate the fusion module based on ``self.fusion_type``.

        Args:
            dropout: Dropout rate for fusion modules that accept it.
        """
        if self.fusion_type == "early":
            self.fusion: nn.Module = EarlyFusion(
                modality_dims=self._modality_dims,
                fusion_dim=self.fusion_dim,
                dropout=dropout,
            )
        elif self.fusion_type == "late":
            self.fusion = LateFusion(
                modality_dims=self._modality_dims,
                num_classes=self.num_classes,
                fusion_dim=self.fusion_dim,
                dropout=dropout,
            )
        elif self.fusion_type == "attention":
            self.fusion = AttentionFusion(
                modality_dims=self._modality_dims,
                fusion_dim=self.fusion_dim,
            )
        elif self.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                modality_dims=self._modality_dims,
                fusion_dim=self.fusion_dim,
            )
        elif self.fusion_type == "transformer":
            self.fusion = TransformerFusion(
                modality_dims=self._modality_dims,
                fusion_dim=self.fusion_dim,
            )
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def _encode_modalities(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run each modality through its encoder.

        Disabled modalities (from config) are replaced with zero vectors
        so the fusion module receives a consistent number of embeddings.

        Args:
            batch: Mapping containing modality feature tensors.

        Returns:
            Mapping from modality name to embedding tensors.
        """
        embeddings: dict[str, torch.Tensor] = {}
        for name in MODALITY_NAMES:
            if name not in batch:
                continue
            encoder = self.encoders[name]
            if hasattr(encoder, "encode"):
                emb = encoder.encode(batch[name])
            else:
                emb = encoder(batch[name])

            if name in self.disabled_modalities:
                emb = torch.zeros_like(emb)

            embeddings[name] = emb
        return embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Not applicable — use :meth:`forward` with a batch dict.

        Args:
            x: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Use forward(batch) instead of encode(x) for the full model."
        )

    def forward(
        self, batch: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        """Full forward pass: encode, fuse, classify.

        Args:
            batch: Dict with modality feature tensors and a
                ``modality_mask`` of shape ``(batch, num_modalities)``.

        Returns:
            Dict with ``logits``, ``probabilities``, ``predicted_class``,
            ``fused_embedding``, ``modality_embeddings``, and
            ``attention_weights``.
        """
        modality_embeddings = self._encode_modalities(batch)
        modality_mask = batch["modality_mask"].clone()

        for i, name in enumerate(MODALITY_NAMES):
            if name in self.disabled_modalities:
                modality_mask[:, i] = False

        attention_weights: torch.Tensor | None = None

        if self.fusion_type == "late":
            fusion_out = self.fusion(modality_embeddings, modality_mask)
            logits = fusion_out["fused"]
            parts: list[torch.Tensor] = []
            for i, name in enumerate(MODALITY_NAMES):
                emb = modality_embeddings[name]
                mask_val = modality_mask[:, i].unsqueeze(1).float()
                parts.append(emb * mask_val)
            fused_embedding = self.embedding_projection(
                torch.cat(parts, dim=1),
            )
        else:
            fused_embedding = self.fusion(modality_embeddings, modality_mask)
            logits = self.classifier(fused_embedding)
            if hasattr(self.fusion, "attention_weights"):
                attention_weights = self.fusion.attention_weights

        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=-1)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "predicted_class": predicted_class,
            "fused_embedding": fused_embedding,
            "modality_embeddings": modality_embeddings,
            "attention_weights": attention_weights,
        }

    def get_output_dim(self) -> int:
        """Return the number of output classes."""
        return self.num_classes

    @classmethod
    def from_config(cls, config: Config) -> PathogenicityPredictor:
        """Instantiate a predictor from a configuration object.

        Args:
            config: Project configuration with a ``model`` section.

        Returns:
            Initialized :class:`PathogenicityPredictor`.
        """
        return cls(config)

    def summary(self) -> str:
        """Return a parameter-count summary per component.

        Returns:
            Multi-line summary string (also logged at INFO level).
        """
        lines: list[str] = [
            "PathogenicityPredictor Summary",
            "=" * 40,
        ]
        total = 0

        for name in MODALITY_NAMES:
            count = sum(
                p.numel()
                for p in self.encoders[name].parameters()
                if p.requires_grad
            )
            lines.append(f"  Encoder [{name}]: {count:,} params")
            total += count

        fusion_count = sum(
            p.numel() for p in self.fusion.parameters() if p.requires_grad
        )
        lines.append(f"  Fusion [{self.fusion_type}]: {fusion_count:,} params")
        total += fusion_count

        if self.classifier is not None:
            cls_count = sum(
                p.numel()
                for p in self.classifier.parameters()
                if p.requires_grad
            )
            lines.append(f"  Classifier: {cls_count:,} params")
            total += cls_count

        if self.embedding_projection is not None:
            proj_count = sum(
                p.numel()
                for p in self.embedding_projection.parameters()
                if p.requires_grad
            )
            lines.append(f"  Embedding projection: {proj_count:,} params")
            total += proj_count

        lines.append("-" * 40)
        lines.append(f"  TOTAL: {total:,} trainable params")

        text = "\n".join(lines)
        logger.info(text)
        return text
