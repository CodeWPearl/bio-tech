"""Abstract base class for all models in the pathogenicity prediction framework.

Provides a common interface (:meth:`encode`, :meth:`forward`) and utility
methods (:meth:`count_parameters`, :meth:`get_device`, :meth:`get_output_dim`)
that every encoder and predictor must implement or inherit.
"""

from __future__ import annotations

import abc
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module, abc.ABC):
    """Abstract base for all project models.

    Subclasses must implement :meth:`encode` and :meth:`forward`.
    """

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce a latent embedding from raw input.

        Args:
            x: Input tensor whose shape depends on the concrete encoder.

        Returns:
            Embedding tensor of shape ``(batch, embed_dim)``.
        """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Full forward pass (may include classification head or decoder).

        Args:
            x: Input tensor.

        Returns:
            Output tensor or dict of tensors.
        """

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of model parameters.

        Args:
            trainable_only: If ``True``, count only parameters with
                ``requires_grad=True``.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_device(self) -> torch.device:
        """Return the device of the first parameter, or CPU if empty."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_output_dim(self) -> int:
        """Return the dimensionality of the encoder's embedding output.

        Subclasses should override this when the output dimension is
        configurable.

        Raises:
            NotImplementedError: If the subclass does not override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_output_dim()"
        )
