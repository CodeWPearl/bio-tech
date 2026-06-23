"""Modality-specific encoders for the pathogenicity prediction framework."""

from src.models.encoders.cnv_encoder import CNVAttentionEncoder, CNVFCEncoder
from src.models.encoders.expression_encoder import (
    DenseAutoencoder,
    ExpressionTransformerEncoder,
    VariationalAutoencoder,
)
from src.models.encoders.methylation_encoder import (
    MethylationDenseAutoencoder,
    MethylationTransformerEncoder,
    MethylationVAE,
)
from src.models.encoders.mutation_encoder import (
    MutationEncoder,
    MutationTransformerEncoder,
)

__all__ = [
    "CNVAttentionEncoder",
    "CNVFCEncoder",
    "DenseAutoencoder",
    "ExpressionTransformerEncoder",
    "MethylationDenseAutoencoder",
    "MethylationTransformerEncoder",
    "MethylationVAE",
    "MutationEncoder",
    "MutationTransformerEncoder",
    "VariationalAutoencoder",
]
