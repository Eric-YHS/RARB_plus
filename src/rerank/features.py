from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    return mol


def _morgan_fp(mol: Chem.Mol, radius: int, fp_size: int):
    return AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(fp_size), useChirality=True)


def _num_rings(mol: Chem.Mol) -> int:
    try:
        return int(mol.GetRingInfo().NumRings())
    except Exception:
        return 0


def featurize_product_reactant(
    product_smiles: str,
    reactant_smiles: str,
    *,
    radius: int = 2,
    fp_size: int = 2048,
    extra_numeric: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    """Featurize a (product, reactant) pair for reranking.

    Returns:
        np.ndarray of shape (d,) or None if either SMILES is invalid.
    """
    pmol = _safe_mol_from_smiles(product_smiles)
    rmol = _safe_mol_from_smiles(reactant_smiles)
    if pmol is None or rmol is None:
        return None

    pfp = _morgan_fp(pmol, radius=radius, fp_size=fp_size)
    rfp = _morgan_fp(rmol, radius=radius, fp_size=fp_size)

    tanimoto = float(DataStructs.TanimotoSimilarity(pfp, rfp))

    # Component-level similarities (reactants can contain multiple components separated by '.')
    comp_sims: List[float] = []
    for comp in str(reactant_smiles).split("."):
        cmol = _safe_mol_from_smiles(comp)
        if cmol is None:
            continue
        cfp = _morgan_fp(cmol, radius=radius, fp_size=fp_size)
        comp_sims.append(float(DataStructs.TanimotoSimilarity(pfp, cfp)))
    if len(comp_sims) == 0:
        comp_max = 0.0
        comp_mean = 0.0
    else:
        comp_max = float(np.max(comp_sims))
        comp_mean = float(np.mean(comp_sims))

    p_heavy = float(pmol.GetNumHeavyAtoms())
    r_heavy = float(rmol.GetNumHeavyAtoms())
    heavy_diff = float(r_heavy - p_heavy)

    p_rings = float(_num_rings(pmol))
    r_rings = float(_num_rings(rmol))
    ring_diff = float(r_rings - p_rings)

    n_components = float(len([c for c in str(reactant_smiles).split(".") if c.strip() != ""]))

    base_feats = np.array(
        [
            tanimoto,
            comp_max,
            comp_mean,
            p_heavy,
            r_heavy,
            heavy_diff,
            p_rings,
            r_rings,
            ring_diff,
            n_components,
        ],
        dtype=np.float32,
    )

    if extra_numeric:
        extra_vals = np.array([float(extra_numeric[k]) for k in sorted(extra_numeric.keys())], dtype=np.float32)
        return np.concatenate([base_feats, extra_vals], axis=0)
    return base_feats


@dataclass(frozen=True)
class RerankFeaturizer:
    radius: int = 2
    fp_size: int = 2048
    extra_numeric_keys: Tuple[str, ...] = ()

    @property
    def feature_dim(self) -> int:
        return 10 + len(self.extra_numeric_keys)

    def __call__(self, product_smiles: str, reactant_smiles: str, row: Optional[Dict] = None) -> Optional[np.ndarray]:
        extra: Dict[str, float] = {}
        if row is not None:
            for k in self.extra_numeric_keys:
                try:
                    extra[k] = float(row[k])
                except Exception:
                    extra[k] = 0.0
        return featurize_product_reactant(
            product_smiles,
            reactant_smiles,
            radius=self.radius,
            fp_size=self.fp_size,
            extra_numeric=extra if self.extra_numeric_keys else None,
        )


def add_confidence_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-product confidence = count(pred) / group_size(product)."""
    if "product" not in df.columns or "pred" not in df.columns:
        raise ValueError("`add_confidence_feature` expects `product` and `pred` columns.")

    counts = df.groupby(["product", "pred"]).size().reset_index(name="count")
    group_size = df.groupby(["product"]).size().reset_index(name="group_size")

    counts_dict = {(p, r): int(c) for p, r, c in zip(counts["product"], counts["pred"], counts["count"])}
    size_dict = {p: int(s) for p, s in zip(group_size["product"], group_size["group_size"])}

    df = df.copy()
    df["count"] = df.apply(lambda x: counts_dict.get((x["product"], x["pred"]), 0), axis=1)
    df["group_size"] = df["product"].apply(lambda p: size_dict.get(p, 1))
    df["confidence"] = df["count"] / df["group_size"]
    return df
