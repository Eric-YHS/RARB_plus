import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from tqdm import tqdm

from src.retrieval import (
    StructuralWeightedMorganConfig,
    compute_structural_weighted_morgan_bit_weights,
    weighted_containment_similarity_fp,
)


def _split_rxn(rxn_smiles: str) -> Tuple[str, str]:
    reactants_smi, _, product_smi = rxn_smiles.split(">")
    return reactants_smi.strip(), product_smi.strip()


def _build_candidate_fps_morgan(rxn_smiles_list: List[str], *, radius: int, fp_size: int):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(fp_size))
    fps = []
    orig_idx = []
    for j, rxn in enumerate(tqdm(rxn_smiles_list, desc="Building candidate fingerprints (Morgan)")):
        reactants_smi, _product_smi = _split_rxn(rxn)
        mol = Chem.MolFromSmiles(reactants_smi)
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        orig_idx.append(j)
    orig_to_pos = {j: pos for pos, j in enumerate(orig_idx)}
    return gen, fps, orig_idx, orig_to_pos


def _build_candidate_fps_sw_morgan(rxn_smiles_list: List[str], *, radius: int, fp_size: int):
    fps = []
    orig_idx = []
    for j, rxn in enumerate(tqdm(rxn_smiles_list, desc="Building candidate fingerprints (SW-Morgan)")):
        reactants_smi, _product_smi = _split_rxn(rxn)
        mol = Chem.MolFromSmiles(reactants_smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(radius), nBits=int(fp_size), useChirality=True)
        fps.append(fp)
        orig_idx.append(j)
    orig_to_pos = {j: pos for pos, j in enumerate(orig_idx)}
    return fps, orig_idx, orig_to_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--retrieval_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--retrieval_type", type=str, default="morgan", choices=["morgan", "sw_morgan"])
    parser.add_argument("--data_type", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_size", type=int, default=10)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=4096)

    # Structural weighting config (only for sw_morgan)
    parser.add_argument("--center_method", type=str, default="brics", choices=["brics", "none"])
    parser.add_argument("--center_weight", type=float, default=2.0)
    parser.add_argument("--center_max_distance", type=int, default=2)
    parser.add_argument("--center_decay", type=float, default=0.5)

    args = parser.parse_args()

    input_df = pd.read_csv(args.input_file, index_col=False)
    retri_df = pd.read_csv(args.retrieval_file, index_col=False)

    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_{args.retrieval_type}.csv"

    rxn_col = "reactants>reagents>production"
    if rxn_col not in input_df.columns or rxn_col not in retri_df.columns:
        raise ValueError(f"Missing column '{rxn_col}' in input/retrieval files.")

    retri_rxns = retri_df[rxn_col].astype(str).tolist()
    input_rxns = input_df[rxn_col].astype(str).tolist()

    max_k = int(args.max_size)

    index_list = [None] * len(input_df)
    sim_list = [None] * len(input_df)

    skip_self = (args.data_type == "train") and (os.path.abspath(args.input_file) == os.path.abspath(args.retrieval_file))

    if args.retrieval_type == "morgan":
        gen, cand_fps, cand_orig_idx, cand_orig_to_pos = _build_candidate_fps_morgan(
            retri_rxns, radius=args.radius, fp_size=args.fp_size
        )
        if len(cand_fps) == 0:
            raise RuntimeError("No valid candidate fingerprints were built.")

        for i, rxn in enumerate(tqdm(input_rxns, desc="Retrieving (Morgan)")):
            _reactants_smi, product_smi = _split_rxn(rxn)
            qmol = Chem.MolFromSmiles(product_smi)
            if qmol is None:
                index_list[i] = ""
                sim_list[i] = ""
                continue

            qfp = gen.GetFingerprint(qmol)
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(qfp, cand_fps), dtype=np.float32)
            if skip_self and i in cand_orig_to_pos:
                sims[cand_orig_to_pos[i]] = -1.0

            if max_k >= sims.size:
                top_pos = np.argsort(-sims)
            else:
                top_pos = np.argpartition(sims, -max_k)[-max_k:]
                top_pos = top_pos[np.argsort(-sims[top_pos])]

            top_orig = [cand_orig_idx[p] for p in top_pos]
            top_sim = [float(sims[p]) for p in top_pos]

            index_list[i] = ",".join(map(str, top_orig))
            sim_list[i] = ",".join(map(lambda x: f"{x:.6f}", top_sim))

    elif args.retrieval_type == "sw_morgan":
        cand_fps, cand_orig_idx, cand_orig_to_pos = _build_candidate_fps_sw_morgan(
            retri_rxns, radius=args.radius, fp_size=args.fp_size
        )
        if len(cand_fps) == 0:
            raise RuntimeError("No valid candidate fingerprints were built.")

        sw_cfg = StructuralWeightedMorganConfig(
            radius=int(args.radius),
            fp_size=int(args.fp_size),
            center_method=str(args.center_method),
            center_weight=float(args.center_weight),
            center_max_distance=int(args.center_max_distance),
            center_decay=float(args.center_decay),
        )

        for i, rxn in enumerate(tqdm(input_rxns, desc="Retrieving (SW-Morgan)")):
            _reactants_smi, product_smi = _split_rxn(rxn)
            qmol = Chem.MolFromSmiles(product_smi)
            if qmol is None:
                index_list[i] = ""
                sim_list[i] = ""
                continue

            q_bit_weights = compute_structural_weighted_morgan_bit_weights(qmol, sw_cfg)

            sims = np.empty(len(cand_fps), dtype=np.float32)
            for pos, cfp in enumerate(cand_fps):
                sims[pos] = weighted_containment_similarity_fp(q_bit_weights, cfp)

            if skip_self and i in cand_orig_to_pos:
                sims[cand_orig_to_pos[i]] = -1.0

            if max_k >= sims.size:
                top_pos = np.argsort(-sims)
            else:
                top_pos = np.argpartition(sims, -max_k)[-max_k:]
                top_pos = top_pos[np.argsort(-sims[top_pos])]

            top_orig = [cand_orig_idx[p] for p in top_pos]
            top_sim = [float(sims[p]) for p in top_pos]

            index_list[i] = ",".join(map(str, top_orig))
            sim_list[i] = ",".join(map(lambda x: f"{x:.6f}", top_sim))

    else:
        raise ValueError(f"Unknown retrieval_type: {args.retrieval_type}")

    out_df = input_df.copy()
    out_df["retrieval_index"] = index_list
    out_df["retrieval_similarity"] = sim_list

    out_df.to_csv(args.output_file, index=False)
    print(f"Saved: {args.output_file}")


if __name__ == "__main__":
    main()

