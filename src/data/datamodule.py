"""PyTorch Lightning DataModule for the pathogenicity prediction pipeline.

:class:`PathogenicityDataModule` orchestrates the full data lifecycle:
downloading raw data (if absent), running the merge + feature-extraction
pipeline, caching processed features to disk, and serving
:class:`~torch.utils.data.DataLoader` instances with class-balanced sampling
for training.

All behaviour is driven by the project YAML config (see
``configs/default.yaml``), so no hardcoded paths or magic numbers leak in.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import MultiOmicsDataset, collate_fn
from src.utils.config import Config

logger = logging.getLogger(__name__)

# On-disk cache file written after the first feature-extraction run.
_CACHE_FILENAME: str = "feature_cache.pkl"


class PathogenicityDataModule(pl.LightningDataModule):
    """Lightning DataModule for multi-omics pathogenicity classification.

    Args:
        config: Project configuration (a :class:`~src.utils.config.Config`
            instance or an ``omegaconf.DictConfig``-compatible object with
            ``data``, ``training``, and ``model`` sections).
    """

    def __init__(self, config: Config | Any) -> None:
        super().__init__()
        self.config = config
        self.batch_size: int = int(config.training.batch_size)
        self.num_workers: int = int(config.training.num_workers)

        self.train_dataset: MultiOmicsDataset | None = None
        self.val_dataset: MultiOmicsDataset | None = None
        self.test_dataset: MultiOmicsDataset | None = None

        self._class_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Download raw data if not already present on disk.

        Called once per node (not per GPU), so it is safe to perform I/O
        here.  The actual download is delegated to the existing download
        script; this method only checks whether the processed ClinVar file
        and at least one cBioPortal study directory exist.
        """
        project_root = Path.cwd()
        clinvar_path = project_root / "data" / "processed" / "clinvar_processed.parquet"
        raw_dir = project_root / "data" / "raw"

        if clinvar_path.is_file() and raw_dir.is_dir():
            logger.info("Raw data already present — skipping download")
            return

        logger.info("Raw data missing — triggering download pipeline")
        try:
            from src.data.clinvar_loader import download_clinvar, process_clinvar
            from src.data.cbioportal_client import CBioPortalClient

            if not clinvar_path.is_file():
                raw_clinvar = project_root / "data" / "raw" / "variant_summary.txt.gz"
                download_clinvar(dest=raw_clinvar)
                process_clinvar(raw_path=raw_clinvar, out_path=clinvar_path)

            client = CBioPortalClient()
            studies = list(self.config.data.studies)
            client.download_all_studies(studies, raw_dir)
        except Exception:
            logger.exception("Download pipeline failed — see traceback above")
            raise

    def setup(self, stage: str | None = None) -> None:
        """Load splits, extract features, and build Dataset objects.

        Results are cached to ``data/processed/feature_cache.pkl`` after
        the first run so subsequent calls skip the expensive extraction.

        Args:
            stage: ``"fit"``, ``"validate"``, ``"test"``, or ``None`` (all).
        """
        project_root = Path.cwd()
        cache_path = project_root / "data" / "processed" / _CACHE_FILENAME

        if cache_path.is_file():
            logger.info("Loading cached features from %s", cache_path)
            cached = self._load_cache(cache_path)
        else:
            logger.info("No feature cache — running extraction pipeline")
            cached = self._extract_and_cache(project_root, cache_path)

        self._log_statistics(cached)

        if stage in ("fit", None):
            self.train_dataset = self._build_dataset(cached, "train")
            self.val_dataset = self._build_dataset(cached, "val")
            train_labels = cached["train"]["labels"]
            self._class_weights = self._compute_class_weights(train_labels)

        if stage in ("validate", None) and self.val_dataset is None:
            self.val_dataset = self._build_dataset(cached, "val")

        if stage in ("test", None):
            self.test_dataset = self._build_dataset(cached, "test")

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader with class-weighted random sampling."""
        assert self.train_dataset is not None, "Call setup('fit') first"
        assert self._class_weights is not None

        sampler = self._build_weighted_sampler(
            self.train_dataset.labels.numpy(), self._class_weights
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader without shuffling."""
        assert self.val_dataset is not None, "Call setup('fit') or setup('validate') first"
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader without shuffling."""
        assert self.test_dataset is not None, "Call setup('test') first"
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_and_cache(
        self, project_root: Path, cache_path: Path
    ) -> dict[str, dict[str, Any]]:
        """Run the merge → feature pipeline and persist results.

        Args:
            project_root: Root of the project directory.
            cache_path: Where to write the pickle cache.

        Returns:
            Nested dict ``{split_name: {modality: array, ...}}``.
        """
        import pandas as pd

        from src.data.data_merger import (
            DataSplitter,
            load_cbioportal_modalities,
            DataMerger,
        )
        from src.features import FeaturePipeline

        clinvar_path = project_root / "data" / "processed" / "clinvar_processed.parquet"
        raw_dir = project_root / "data" / "raw"
        splits_dir = project_root / "data" / "splits"

        studies = list(self.config.data.studies)
        test_size = float(self.config.data.test_size)
        val_size = float(self.config.data.val_size)
        random_seed = int(self.config.data.random_seed)

        # --- merge ---
        clinvar_df = pd.read_parquet(clinvar_path)
        modalities = load_cbioportal_modalities(raw_dir, studies)

        merger = DataMerger()
        merged = merger.merge_clinvar_with_mutations(
            clinvar_df, modalities["mutations"]
        )
        full = merger.attach_omics_features(
            merged,
            modalities.get("expression"),
            modalities.get("methylation"),
            modalities.get("cnv"),
            modalities.get("clinical"),
        )

        # --- split ---
        splitter = DataSplitter(output_dir=splits_dir)
        splits = splitter.split_by_gene(
            full, test_size, val_size, random_seed, save=True
        )

        # --- features ---
        pipeline = FeaturePipeline()
        pipeline.fit(splits["train"])

        cached: dict[str, dict[str, Any]] = {}
        for name, split_df in splits.items():
            features = pipeline.transform(split_df)
            labels = split_df["label"].values.astype(np.int64)

            mask = np.column_stack([
                split_df.get("has_expression", pd.Series(False, index=split_df.index)).values,
                split_df.get("has_methylation", pd.Series(False, index=split_df.index)).values,
                split_df.get("has_cnv", pd.Series(False, index=split_df.index)).values,
            ]).astype(bool)

            cached[name] = {
                "mutation": features["mutation"],
                "expression": features["expression"],
                "methylation": features["methylation"],
                "cnv": features["cnv"],
                "clinical": features["clinical"],
                "labels": labels,
                "modality_mask": mask,
            }

        pipeline.save(project_root / "data" / "processed" / "feature_pipeline.pkl")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(cached, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Wrote feature cache to %s", cache_path)

        return cached

    @staticmethod
    def _load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
        """Deserialise the feature cache from disk.

        Args:
            cache_path: Path to the pickle file.

        Returns:
            The cached feature dict.
        """
        with cache_path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301

    @staticmethod
    def _build_dataset(
        cached: dict[str, dict[str, Any]], split: str
    ) -> MultiOmicsDataset:
        """Construct a :class:`MultiOmicsDataset` from cached arrays.

        Args:
            cached: Full cache dict (all splits).
            split: Name of the split (``"train"``, ``"val"``, ``"test"``).

        Returns:
            The dataset for the requested split.
        """
        data = cached[split]
        expr = data["expression"] if data["expression"].shape[1] > 0 else None
        meth = data["methylation"] if data["methylation"].shape[1] > 0 else None
        cnv = data["cnv"] if data["cnv"].shape[1] > 0 else None

        return MultiOmicsDataset(
            mutation_features=data["mutation"],
            expression_features=expr,
            methylation_features=meth,
            cnv_features=cnv,
            clinical_features=data["clinical"],
            labels=data["labels"],
            modality_mask=data["modality_mask"],
        )

    @staticmethod
    def _compute_class_weights(labels: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency weights for class-balanced sampling.

        Args:
            labels: Integer label array from the training split.

        Returns:
            Array of per-class weights (higher for rarer classes).
        """
        classes, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts.astype(np.float64)
        weights /= weights.sum()
        logger.info(
            "Class weights: %s",
            {int(c): round(float(w), 6) for c, w in zip(classes, weights)},
        )
        return weights

    @staticmethod
    def _build_weighted_sampler(
        labels: np.ndarray, class_weights: np.ndarray
    ) -> WeightedRandomSampler:
        """Build a :class:`WeightedRandomSampler` from per-class weights.

        Each sample's sampling weight equals the weight of its class, so
        rare classes are drawn more frequently.

        Args:
            labels: Integer label array.
            class_weights: Per-class weights from :meth:`_compute_class_weights`.

        Returns:
            A sampler that can be passed to :class:`DataLoader`.
        """
        sample_weights = torch.as_tensor(
            class_weights[labels], dtype=torch.float64
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

    @staticmethod
    def _log_statistics(cached: dict[str, dict[str, Any]]) -> None:
        """Log dataset statistics for each split.

        Args:
            cached: The full feature cache.
        """
        logger.info("=== Dataset statistics ===")
        for split_name, data in cached.items():
            n_samples = len(data["labels"])
            classes, counts = np.unique(data["labels"], return_counts=True)
            dist = {int(c): int(n) for c, n in zip(classes, counts)}
            mask = data["modality_mask"]
            n_expr = int(mask[:, 0].sum()) if mask.shape[1] > 0 else 0
            n_meth = int(mask[:, 1].sum()) if mask.shape[1] > 1 else 0
            n_cnv = int(mask[:, 2].sum()) if mask.shape[1] > 2 else 0
            logger.info(
                "%-5s | samples=%d | labels=%s | "
                "expression=%d | methylation=%d | cnv=%d",
                split_name,
                n_samples,
                dist,
                n_expr,
                n_meth,
                n_cnv,
            )
