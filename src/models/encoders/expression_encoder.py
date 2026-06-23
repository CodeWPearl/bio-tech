"""RNA-seq expression encoders.

Three architectures for learning compact representations from high-dimensional
gene-expression vectors (~2000 genes):

* :class:`DenseAutoencoder` — deterministic autoencoder whose bottleneck
  serves as the embedding during pathogenicity prediction.
* :class:`VariationalAutoencoder` — VAE that returns ``(z, mu, logvar)`` so
  the KL divergence term can be added to the training loss.
* :class:`ExpressionTransformerEncoder` — chunks the gene vector into groups,
  projects each to a token, and applies a multi-layer Transformer.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import nn

from src.models.base import BaseModel

logger = logging.getLogger(__name__)


class DenseAutoencoder(BaseModel):
    """Deterministic autoencoder for expression data.

    Args:
        input_dim: Number of gene features (default 2000).
        embed_dim: Bottleneck / embedding dimensionality (default 256).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 256,
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
            x: ``(batch, input_dim)`` expression tensor.

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
            x: ``(batch, input_dim)`` expression tensor.

        Returns:
            Dict with keys ``embedding`` and ``reconstruction``.
        """
        z = self.encode(x)
        recon = self.decode(z)
        return {"embedding": z, "reconstruction": recon}


class VariationalAutoencoder(BaseModel):
    """Variational autoencoder for expression data.

    Args:
        input_dim: Number of gene features (default 2000).
        embed_dim: Latent dimensionality (default 256).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 256,
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
            x: ``(batch, input_dim)`` expression tensor.

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
            x: ``(batch, input_dim)`` expression tensor.

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


class ExpressionTransformerEncoder(BaseModel):
    """Transformer encoder over gene-expression groups.

    Chunks the 2000-gene vector into 40 groups of 50, projects each to the
    model dimension, and applies a multi-layer Transformer with a ``[CLS]``
    token.

    Args:
        input_dim: Number of gene features (default 2000).
        embed_dim: Output embedding dimensionality (default 256).
        group_size: Genes per chunk (default 50).
        n_heads: Attention heads (default 8).
        n_layers: Transformer layers (default 4).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2000,
        embed_dim: int = 256,
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
        """Produce an embedding from the expression vector.

        Args:
            x: ``(batch, input_dim)`` expression tensor.

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
        tokens = self.group_projection(chunks)  # (B, n_groups, embed_dim)

        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, tokens], dim=1)  # (B, n_groups+1, embed_dim)

        seq = seq + self.pos_embed[:, : seq.size(1), :]

        out = self.transformer(seq)
        cls_out = self.norm(out[:, 0, :])

        return cls_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`encode`."""
        return self.encode(x)
