"""Pre-train autoencoder and VAE encoders on unlabeled omics data.

Learns meaningful latent representations from the full expression and
methylation matrices (ALL samples, not just ClinVar-labeled ones) before
the main supervised training begins.  Pre-trained encoder weights are
saved to ``results/checkpoints/`` and can be loaded by the
:class:`~src.models.full_model.PathogenicityPredictor` during supervised
training.

Four models are pre-trained:

1. DenseAutoencoder on expression data  → expression_ae_pretrained.pt
2. DenseAutoencoder on methylation data → methylation_ae_pretrained.pt
3. VAE on expression data               → expression_vae_pretrained.pt
4. VAE on methylation data              → methylation_vae_pretrained.pt

Usage::

    python scripts/pretrain_autoencoders.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple reconstruction dataset (input == target)
# ---------------------------------------------------------------------------

class OmicsReconstructionDataset(Dataset):
    """Dataset for autoencoder pre-training where input equals target.

    Args:
        data: 2-D NumPy array of shape ``(n_samples, n_features)``.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.as_tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ---------------------------------------------------------------------------
# VAE loss (reconstruction + KL divergence)
# ---------------------------------------------------------------------------

def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss: MSE reconstruction + beta-weighted KL divergence.

    Args:
        recon: Reconstructed tensor.
        target: Original input tensor.
        mu: Mean of the latent distribution.
        logvar: Log-variance of the latent distribution.
        beta: Weight for the KL divergence term.

    Returns:
        Tuple of ``(total_loss, recon_loss, kl_loss)``.
    """
    recon_loss = nn.functional.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ---------------------------------------------------------------------------
# Pre-training loop
# ---------------------------------------------------------------------------

def pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 0.001,
    is_vae: bool = False,
    beta: float = 0.5,
    patience: int = 10,
    device: torch.device | None = None,
    mlflow_experiment: str | None = None,
    run_name: str = "pretrain",
) -> dict[str, list[float]]:
    """Train an autoencoder or VAE on a reconstruction task.

    Args:
        model: The autoencoder or VAE model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        is_vae: If ``True``, use VAE loss with KL divergence.
        beta: KL weight for VAE loss.
        patience: Early stopping patience.
        device: Device to train on.
        mlflow_experiment: MLflow experiment name for logging.
        run_name: MLflow run name.

    Returns:
        Dict with ``train_loss`` and ``val_loss`` history lists.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow_run = None
    if mlflow_experiment:
        try:
            import mlflow

            mlflow.set_experiment(mlflow_experiment)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "epochs": epochs,
                "lr": lr,
                "is_vae": is_vae,
                "beta": beta,
                "patience": patience,
            })
        except ImportError:
            logger.warning("mlflow not installed — skipping experiment logging")

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    wait = 0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)

            if is_vae:
                loss, _, _ = vae_loss(
                    out["reconstruction"], batch,
                    out["mu"], out["logvar"],
                    beta=beta,
                )
            else:
                loss = nn.functional.mse_loss(out["reconstruction"], batch)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        history["train_loss"].append(avg_train)

        # --- validate ---
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)

                if is_vae:
                    loss, _, _ = vae_loss(
                        out["reconstruction"], batch,
                        out["mu"], out["logvar"],
                        beta=beta,
                    )
                else:
                    loss = nn.functional.mse_loss(out["reconstruction"], batch)

                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses))
        history["val_loss"].append(avg_val)

        if mlflow_run:
            try:
                import mlflow

                mlflow.log_metrics(
                    {"train_loss": avg_train, "val_loss": avg_val},
                    step=epoch,
                )
            except Exception:
                pass

        logger.info(
            "Epoch %3d/%d — train_loss=%.6f  val_loss=%.6f",
            epoch + 1, epochs, avg_train, avg_val,
        )

        # --- early stopping ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1, patience,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if mlflow_run:
        try:
            import mlflow

            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.end_run()
        except Exception:
            pass

    return history


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_omics_matrix(data_dir: Path, modality: str) -> np.ndarray | None:
    """Load an omics matrix from processed data files.

    Searches for common file patterns in the data directory.

    Args:
        data_dir: Root data directory.
        modality: ``"expression"`` or ``"methylation"``.

    Returns:
        2-D NumPy array or ``None`` if no file found.
    """
    import pickle

    cache_path = data_dir / "processed" / "feature_cache.pkl"
    if cache_path.is_file():
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)  # noqa: S301
        arrays: list[np.ndarray] = []
        for split_data in cached.values():
            arr = split_data.get(modality)
            if arr is not None and arr.shape[1] > 0:
                arrays.append(arr)
        if arrays:
            combined = np.concatenate(arrays, axis=0)
            logger.info(
                "Loaded %s matrix from cache: %s", modality, combined.shape,
            )
            return combined

    for pattern in [
        f"{modality}_matrix.npy",
        f"{modality}_features.npy",
        f"{modality}.npy",
    ]:
        npy_path = data_dir / "processed" / pattern
        if npy_path.is_file():
            matrix = np.load(npy_path)
            logger.info("Loaded %s from %s: %s", modality, npy_path, matrix.shape)
            return matrix

    logger.warning("No %s matrix found in %s", modality, data_dir)
    return None


def create_data_loaders(
    data: np.ndarray,
    batch_size: int = 128,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Split data and create train/val DataLoaders.

    Args:
        data: 2-D array of omics features.
        batch_size: Batch size.
        val_fraction: Fraction of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    dataset = OmicsReconstructionDataset(data)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pretraining(config_path: str | Path) -> None:
    """Execute the full pre-training pipeline.

    Args:
        config_path: Path to the project config YAML.
    """
    from src.utils.config import load_config

    config = load_config(config_path)
    project_root = Path.cwd()
    data_dir = project_root / "data"
    checkpoint_dir = project_root / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Pre-training hyperparameters from config (with defaults)
    pretrain_cfg = config.get("pretrain", None)
    if pretrain_cfg is None:
        pretrain_epochs = 100
        pretrain_lr = 0.001
        pretrain_batch_size = 128
        beta = 0.5
    else:
        if isinstance(pretrain_cfg, dict):
            from src.utils.config import Config as _Cfg

            pretrain_cfg = _Cfg(pretrain_cfg)
        pretrain_epochs = int(getattr(pretrain_cfg, "pretrain_epochs", 100))
        pretrain_lr = float(getattr(pretrain_cfg, "pretrain_lr", 0.001))
        pretrain_batch_size = int(getattr(pretrain_cfg, "pretrain_batch_size", 128))
        beta = float(getattr(pretrain_cfg, "beta", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pre-training on device: %s", device)

    expression_dim = int(config.model.get("expression_input_dim", 2000))
    expression_embed = int(config.model.expression_embed_dim)
    methylation_dim = int(config.model.get("methylation_input_dim", 2000))
    methylation_embed = int(config.model.methylation_embed_dim)
    dropout = float(config.model.dropout)

    mlflow_experiment = "pretrain_autoencoders"

    # --- load omics data ----------------------------------------------------
    expression_data = load_omics_matrix(data_dir, "expression")
    methylation_data = load_omics_matrix(data_dir, "methylation")

    from src.models.encoders.expression_encoder import (
        DenseAutoencoder,
        VariationalAutoencoder,
    )
    from src.models.encoders.methylation_encoder import (
        MethylationDenseAutoencoder,
        MethylationVAE,
    )

    # --- 1. Expression AE ---------------------------------------------------
    if expression_data is not None and expression_data.shape[1] > 0:
        actual_expr_dim = expression_data.shape[1]
        logger.info("Pre-training Expression AE (input_dim=%d)", actual_expr_dim)

        train_loader, val_loader = create_data_loaders(
            expression_data, batch_size=pretrain_batch_size,
        )

        expr_ae = DenseAutoencoder(
            input_dim=actual_expr_dim,
            embed_dim=expression_embed,
            dropout=dropout,
        )
        pretrain_model(
            expr_ae, train_loader, val_loader,
            epochs=pretrain_epochs, lr=pretrain_lr, is_vae=False,
            device=device, mlflow_experiment=mlflow_experiment,
            run_name="expression_ae",
        )
        save_path = checkpoint_dir / "expression_ae_pretrained.pt"
        torch.save(expr_ae.state_dict(), save_path)
        logger.info("Saved expression AE to %s", save_path)

        # --- 3. Expression VAE -----------------------------------------------
        logger.info("Pre-training Expression VAE (input_dim=%d)", actual_expr_dim)
        expr_vae = VariationalAutoencoder(
            input_dim=actual_expr_dim,
            embed_dim=expression_embed,
            dropout=dropout,
        )
        pretrain_model(
            expr_vae, train_loader, val_loader,
            epochs=pretrain_epochs, lr=pretrain_lr, is_vae=True,
            beta=beta, device=device,
            mlflow_experiment=mlflow_experiment,
            run_name="expression_vae",
        )
        save_path = checkpoint_dir / "expression_vae_pretrained.pt"
        torch.save(expr_vae.state_dict(), save_path)
        logger.info("Saved expression VAE to %s", save_path)
    else:
        logger.warning("No expression data available — skipping expression pre-training")

    # --- 2. Methylation AE ---------------------------------------------------
    if methylation_data is not None and methylation_data.shape[1] > 0:
        actual_meth_dim = methylation_data.shape[1]
        logger.info("Pre-training Methylation AE (input_dim=%d)", actual_meth_dim)

        train_loader, val_loader = create_data_loaders(
            methylation_data, batch_size=pretrain_batch_size,
        )

        meth_ae = MethylationDenseAutoencoder(
            input_dim=actual_meth_dim,
            embed_dim=methylation_embed,
            dropout=dropout,
        )
        pretrain_model(
            meth_ae, train_loader, val_loader,
            epochs=pretrain_epochs, lr=pretrain_lr, is_vae=False,
            device=device, mlflow_experiment=mlflow_experiment,
            run_name="methylation_ae",
        )
        save_path = checkpoint_dir / "methylation_ae_pretrained.pt"
        torch.save(meth_ae.state_dict(), save_path)
        logger.info("Saved methylation AE to %s", save_path)

        # --- 4. Methylation VAE -----------------------------------------------
        logger.info("Pre-training Methylation VAE (input_dim=%d)", actual_meth_dim)
        meth_vae = MethylationVAE(
            input_dim=actual_meth_dim,
            embed_dim=methylation_embed,
            dropout=dropout,
        )
        pretrain_model(
            meth_vae, train_loader, val_loader,
            epochs=pretrain_epochs, lr=pretrain_lr, is_vae=True,
            beta=beta, device=device,
            mlflow_experiment=mlflow_experiment,
            run_name="methylation_vae",
        )
        save_path = checkpoint_dir / "methylation_vae_pretrained.pt"
        torch.save(meth_vae.state_dict(), save_path)
        logger.info("Saved methylation VAE to %s", save_path)
    else:
        logger.warning("No methylation data available — skipping methylation pre-training")

    logger.info("Pre-training pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the pre-training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Pre-train autoencoder and VAE encoders on unlabeled omics data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the project config YAML.",
    )
    args = parser.parse_args()

    run_pretraining(args.config)


if __name__ == "__main__":
    main()
