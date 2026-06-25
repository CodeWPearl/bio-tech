"""Methylation feature encoders.

Mirrors the expression encoder architectures but with independent batch
normalisation statistics, so methylation-specific distributional properties
are preserved.

* :class:`MethylationDenseAutoencoder` — deterministic autoencoder.
* :class:`MethylationVAE` — variational autoencoder.
* :class:`MethylationTransformerEncoder` — Transformer over gene-group tokens.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch import nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class MethylationDenseAutoencoder(BaseModel):
    """Deterministic autoencoder for methylation data.

    Args:
        input_dim: Number of methylation gene features (default 2000).
        embed_dim: Bottleneck / embedding dimensionality (default 128).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(512, input_dim),
        )

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the bottleneck embedding.

        Args:
            x: ``(batch, input_dim)`` methylation tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the embedding.

        Args:
            z: ``(batch, embed_dim)`` latent tensor.

        Returns:
            ``(batch, input_dim)`` reconstruction.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full autoencoder forward pass.

        Args:
            x: ``(batch, input_dim)`` methylation tensor.

        Returns:
            Dict with keys ``embedding`` and ``reconstruction``.
        """
        z = self.encode(x)
        recon = self.decode(z)
        return {"embedding": z, "reconstruction": recon}

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        freeze: bool = False,
    ) -> None:
        """Load pretrained encoder weights from a checkpoint.

        Only the encoder half is loaded; decoder weights are ignored.
        Optionally freezes encoder parameters.

        Args:
            checkpoint_path: Path to the saved ``.pt`` checkpoint.
            freeze: If ``True``, set ``requires_grad=False`` on all
                encoder parameters.
        """
        checkpoint_path = Path(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_state = {
            k: v for k, v in state_dict.items() if k.startswith("encoder.")
        }
        if encoder_state:
            self.load_state_dict(encoder_state, strict=False)
            logger.info(
                "Loaded %d pretrained encoder keys from %s",
                len(encoder_state), checkpoint_path,
            )
        else:
            logger.warning("No encoder keys found in %s", checkpoint_path)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze encoder parameters")


class MethylationVAE(BaseModel):
    """Variational autoencoder for methylation data.

    Args:
        input_dim: Number of methylation gene features (default 2000).
        embed_dim: Latent dimensionality (default 128).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim

        self.encoder_body = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_mu = nn.Linear(512, embed_dim)
        self.fc_logvar = nn.Linear(512, embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(max(dropout - 0.1, 0.0)),
            nn.Linear(512, input_dim),
        )

    def get_output_dim(self) -> int:
        """Return the latent dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the latent sample via reparameterization.

        Args:
            x: ``(batch, input_dim)`` methylation tensor.

        Returns:
            ``(batch, embed_dim)`` sampled latent vector.
        """
        z, _, _ = self._encode_full(x)
        return z

    def _encode_full(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(z, mu, logvar)``."""
        h = self.encoder_body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def _reparameterize(
        mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct the input from the latent sample.

        Args:
            z: ``(batch, embed_dim)`` latent tensor.

        Returns:
            ``(batch, input_dim)`` reconstruction.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full VAE forward pass.

        Args:
            x: ``(batch, input_dim)`` methylation tensor.

        Returns:
            Dict with keys ``embedding``, ``mu``, ``logvar``,
            ``reconstruction``.
        """
        z, mu, logvar = self._encode_full(x)
        recon = self.decode(z)
        return {
            "embedding": z,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": recon,
        }

    def load_pretrained_weights(
        self,
        checkpoint_path: str | Path,
        freeze: bool = False,
    ) -> None:
        """Load pretrained encoder weights from a checkpoint.

        Loads ``encoder_body``, ``fc_mu``, and ``fc_logvar`` weights;
        decoder weights are ignored. Optionally freezes encoder
        parameters.

        Args:
            checkpoint_path: Path to the saved ``.pt`` checkpoint.
            freeze: If ``True``, set ``requires_grad=False`` on all
                encoder parameters.
        """
        checkpoint_path = Path(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        encoder_prefixes = ("encoder_body.", "fc_mu.", "fc_logvar.")
        encoder_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in encoder_prefixes)
        }
        if encoder_state:
            self.load_state_dict(encoder_state, strict=False)
            logger.info(
                "Loaded %d pretrained VAE encoder keys from %s",
                len(encoder_state), checkpoint_path,
            )
        else:
            logger.warning("No VAE encoder keys found in %s", checkpoint_path)

        if freeze:
            for param in self.encoder_body.parameters():
                param.requires_grad = False
            for param in self.fc_mu.parameters():
                param.requires_grad = False
            for param in self.fc_logvar.parameters():
                param.requires_grad = False
            logger.info("Froze VAE encoder parameters")


class MethylationTransformerEncoder(BaseModel):
    """Transformer encoder over methylation gene groups.

    Same architecture as :class:`ExpressionTransformerEncoder` but with
    independent parameters and batch normalisation statistics.

    Args:
        input_dim: Number of methylation gene features (default 2000).
        embed_dim: Output embedding dimensionality (default 128).
        group_size: Genes per chunk (default 50).
        n_heads: Attention heads (default 8).
        n_layers: Transformer layers (default 4).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 128,
        group_size: int = 50,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self.group_size = group_size
        self.n_groups = math.ceil(input_dim / group_size)
        self._input_dim = input_dim

        self.group_projection = nn.Linear(group_size, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_groups + 1, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    def get_output_dim(self) -> int:
        """Return the embedding dimensionality."""
        return self._embed_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce an embedding from the methylation vector.

        Args:
            x: ``(batch, input_dim)`` methylation tensor.

        Returns:
            ``(batch, embed_dim)`` embedding.
        """
        batch_size = x.size(0)

        padded = x
        remainder = self._input_dim % self.group_size
        if remainder != 0:
            pad_size = self.group_size - remainder
            padded = torch.nn.functional.pad(x, (0, pad_size))

        chunks = padded.view(batch_size, self.n_groups, self.group_size)
        tokens = self.group_projection(chunks)

        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)

        seq = seq + self.pos_embed[:, : seq.size(1), :]

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0, :])

        return cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)
