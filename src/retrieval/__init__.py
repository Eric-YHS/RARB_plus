"""Retrieval utilities (fingerprints, similarity, indexing).

This package is used by RARB to:
- compute retrieval indices for datasets
- perform online retrieval for inference (e.g., predict.py)
"""

from .weighted_fingerprint import (
    StructuralWeightedMorganConfig,
    compute_structural_weighted_morgan_bit_weights,
    weighted_containment_similarity,
    weighted_containment_similarity_fp,
)
