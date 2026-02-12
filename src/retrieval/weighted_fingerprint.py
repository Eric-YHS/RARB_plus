from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS


@dataclass(frozen=True)
class StructuralWeightedMorganConfig:
    radius: int = 2
    fp_size: int = 4096
    center_method: str = "brics"  # "brics" | "none"
    center_weight: float = 2.0
    center_max_distance: int = 2
    center_decay: float = 0.5


def _find_center_atoms(mol: Chem.Mol, method: str) -> Set[int]:
    if method == "none":
        return set()
    if method != "brics":
        raise ValueError(f"Unknown center_method: {method}")

    centers: Set[int] = set()
    try:
        for (a1, a2), _labels in BRICS.FindBRICSBonds(mol):
            centers.add(int(a1))
            centers.add(int(a2))
    except Exception:
        # Be conservative: if BRICS fails, fall back to uniform weights.
        return set()
    return centers


def _compute_atom_weights(
    mol: Chem.Mol,
    center_atoms: Set[int],
    *,
    center_weight: float,
    center_max_distance: int,
    center_decay: float,
) -> np.ndarray:
    num_atoms = int(mol.GetNumAtoms())
    weights = np.ones(num_atoms, dtype=np.float32)
    if not center_atoms:
        return weights

    # Shortest-path distance matrix (num_atoms x num_atoms).
    # RDKit returns float64; cast later.
    dist = Chem.GetDistanceMatrix(mol)
    center_idx = np.array(sorted(center_atoms), dtype=np.int64)
    if center_idx.size == 0:
        return weights

    # For each atom, distance to closest center.
    # dist[:, center_idx] shape: (n, n_centers)
    dmin = dist[:, center_idx].min(axis=1)

    # Apply a decayed bump around centers.
    # w = 1 + center_weight * decay^d for d <= center_max_distance
    within = dmin <= float(center_max_distance)
    weights[within] = weights[within] + center_weight * np.power(center_decay, dmin[within]).astype(np.float32)
    return weights


def compute_structural_weighted_morgan_bit_weights(
    mol: Chem.Mol,
    cfg: StructuralWeightedMorganConfig,
) -> Dict[int, float]:
    """Compute per-bit weights for Morgan fingerprint bits for a molecule.

    Weights are induced from per-atom weights, which are in turn derived from
    (possibly multiple) "reaction center" proxies (e.g., BRICS cut sites).
    """
    if mol is None:
        return {}

    center_atoms = _find_center_atoms(mol, method=cfg.center_method)
    atom_weights = _compute_atom_weights(
        mol,
        center_atoms,
        center_weight=cfg.center_weight,
        center_max_distance=cfg.center_max_distance,
        center_decay=cfg.center_decay,
    )

    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    _ = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        int(cfg.radius),
        nBits=int(cfg.fp_size),
        bitInfo=bit_info,
        useChirality=True,
    )

    bit_weights: Dict[int, float] = {}
    for bit, occs in bit_info.items():
        w = 1.0
        for atom_idx, _rad in occs:
            w = max(w, float(atom_weights[int(atom_idx)]))
        bit_weights[int(bit)] = float(w)
    return bit_weights


def weighted_containment_similarity(
    query_bit_weights: Dict[int, float],
    candidate_on_bits: Iterable[int],
) -> float:
    """Weighted containment similarity between query and candidate fingerprints.

    This is a weighted variant of |Q ∩ C| / |Q| where bits can have non-uniform weights.
    Importantly, bits that exist only in the candidate do NOT penalize the score, which
    matches the retrosynthesis retrieval setting where reactants can have extra leaving
    groups absent from the product.
    """
    if not query_bit_weights:
        return 0.0

    candidate_bits = set(int(b) for b in candidate_on_bits)
    total = float(sum(query_bit_weights.values()))
    if total <= 0:
        return 0.0

    inter = 0.0
    for bit, w in query_bit_weights.items():
        if bit in candidate_bits:
            inter += float(w)
    return float(inter / total)


def weighted_containment_similarity_fp(
    query_bit_weights: Dict[int, float],
    candidate_fp,
) -> float:
    """Same as weighted_containment_similarity but avoids Python set creation.

    candidate_fp is expected to be an RDKit ExplicitBitVect-like object supporting GetBit(bit).
    """
    if not query_bit_weights:
        return 0.0
    total = float(sum(query_bit_weights.values()))
    if total <= 0:
        return 0.0

    inter = 0.0
    for bit, w in query_bit_weights.items():
        if candidate_fp.GetBit(int(bit)):
            inter += float(w)
    return float(inter / total)
