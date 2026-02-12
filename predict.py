import argparse
import numpy as np
import pandas as pd
import torch

from src.utils import disable_rdkit_logging, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDatasetInfos, RetroBridgeDataset
from src.retrieval import (
    StructuralWeightedMorganConfig,
    compute_structural_weighted_morgan_bit_weights,
    weighted_containment_similarity_fp,
)
from src.rerank import MLPReactionReranker, RerankFeaturizer

from torch_geometric.data import Data
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from pdb import set_trace


def assign_trivial_atom_mapping_numbers(molecule):
    order = {}
    for atom in molecule.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(idx)
        order[idx] = idx
    return molecule, order


def retrieve_topk_reactants(
    product_smi: str,
    retrieval_csv: str,
    retrieval_type: str,
    max_size: int,
    fp_radius: int,
    fp_size: int,
    sw_cfg: StructuralWeightedMorganConfig,
):
    df = pd.read_csv(retrieval_csv, index_col=False)
    rxn_col = "reactants>reagents>production"
    if rxn_col not in df.columns:
        raise ValueError(f"Missing column '{rxn_col}' in {retrieval_csv}")

    reactant_smiles = []
    for rxn in df[rxn_col].astype(str).tolist():
        r, _, _p = rxn.split(">")
        reactant_smiles.append(r.strip())

    qmol = Chem.MolFromSmiles(product_smi)
    if qmol is None:
        raise ValueError("Invalid product SMILES for retrieval.")

    if retrieval_type == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(fp_radius), fpSize=int(fp_size))
        qfp = gen.GetFingerprint(qmol)
        cand_fps = []
        valid_pos = []
        for i, smi in enumerate(reactant_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            cand_fps.append(gen.GetFingerprint(mol))
            valid_pos.append(i)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(qfp, cand_fps), dtype=np.float32)
        pos_arr = np.asarray(valid_pos, dtype=np.int64)

    elif retrieval_type == "sw_morgan":
        q_bit_w = compute_structural_weighted_morgan_bit_weights(qmol, sw_cfg)
        sims_list = []
        valid_pos = []
        for i, smi in enumerate(reactant_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, int(sw_cfg.radius), nBits=int(sw_cfg.fp_size), useChirality=True
            )
            sims_list.append(weighted_containment_similarity_fp(q_bit_w, fp))
            valid_pos.append(i)
        sims = np.asarray(sims_list, dtype=np.float32)
        pos_arr = np.asarray(valid_pos, dtype=np.int64)

    else:
        raise ValueError(f"Unknown retrieval_type: {retrieval_type}")

    k = min(int(max_size), int(sims.size))
    if k <= 0:
        return [], []
    if k >= sims.size:
        top_local = np.argsort(-sims)
    else:
        top_local = np.argpartition(sims, -k)[-k:]
        top_local = top_local[np.argsort(-sims[top_local])]

    top_idx = pos_arr[top_local].tolist()
    top_sim = sims[top_local].astype(np.float32).tolist()
    return top_idx, top_sim


def main(
    smiles,
    checkpoint,
    n_samples,
    n_steps,
    seed,
    device,
    retrieval_csv=None,
    encoded_reactants_pt=None,
    retrieval_type="morgan",
    retrieval_max_size=10,
    fp_radius=2,
    fp_size=4096,
    center_method="brics",
    center_weight=2.0,
    center_max_distance=2,
    center_decay=0.5,
    reranker_ckpt=None,
):
    set_deterministic(seed)

    # Loading the model
    model = MarkovBridge.load_from_checkpoint(checkpoint, map_location=device).to(device)
    model.T = n_steps

    if getattr(model, "retrieval_k", 0) > 0 and (not hasattr(model, "encoded_reactants") or model.encoded_reactants is None):
        if encoded_reactants_pt is None:
            raise ValueError(
                "Checkpoint requires retrieval augmentation (retrieval_k > 0) but `encoded_reactants` is missing.\n"
                "Provide --encoded_reactants_pt to load the embedding tensor."
            )
        model.encoded_reactants = torch.load(encoded_reactants_pt, map_location="cpu")

    # Preparing input
    pmol, mapping = assign_trivial_atom_mapping_numbers(Chem.MolFromSmiles(smiles))
    r_num_nodes = pmol.GetNumAtoms() + RetroBridgeDatasetInfos.max_n_dummy_nodes
    p_x, p_edge_index, p_edge_attr = RetroBridgeDataset.compute_graph(
        pmol, mapping, r_num_nodes, RetroBridgeDataset.types, RetroBridgeDataset.bonds
    )
    p_x = p_x.to(device)
    p_edge_index = p_edge_index.to(device)
    p_edge_attr = p_edge_attr.to(device)

    retrieval_list = None
    retrieval_sim = None
    if getattr(model, "retrieval_k", 0) > 0:
        if retrieval_csv is None:
            raise ValueError("retrieval_k > 0: please provide --retrieval_csv for online retrieval.")

        sw_cfg = StructuralWeightedMorganConfig(
            radius=int(fp_radius),
            fp_size=int(fp_size),
            center_method=str(center_method),
            center_weight=float(center_weight),
            center_max_distance=int(center_max_distance),
            center_decay=float(center_decay),
        )
        top_n = max(int(retrieval_max_size), int(model.retrieval_k) + 2)
        idxs, sims = retrieve_topk_reactants(
            product_smi=smiles,
            retrieval_csv=retrieval_csv,
            retrieval_type=retrieval_type,
            max_size=top_n,
            fp_radius=int(fp_radius),
            fp_size=int(fp_size),
            sw_cfg=sw_cfg,
        )
        retrieval_list = torch.tensor([idxs], dtype=torch.long, device=device)
        retrieval_sim = torch.tensor([sims], dtype=torch.float32, device=device)

    dataset, batch = [], []
    idx_offset = 0
    for i in range(n_samples):
        data_kwargs = {
            "idx": i,
            "p_x": p_x,
            "p_edge_index": p_edge_index.clone(),
            "p_edge_attr": p_edge_attr,
            "p_smiles": smiles,
            "rxn_class": torch.tensor([-1], dtype=torch.long),
        }
        if retrieval_list is not None:
            data_kwargs["retrieval_list"] = retrieval_list
            data_kwargs["retrieval_sim"] = retrieval_sim
        data = Data(**data_kwargs)
        data.p_edge_index += idx_offset
        dataset.append(data)
        batch.append(torch.ones_like(data.p_x[:, 0]).to(torch.long) * i)
        idx_offset += len(data.p_x)

    data, _ = RetroBridgeDataset.collate(dataset)
    data.batch = torch.concat(batch)

    # Sampling
    _, _, _, _, molecule_list, _, _, _ = model.sample_chain(
        data, batch_size=n_samples, keep_chain=0, number_chain_steps_to_save=1, save_true_reactants=False
    )
    pred_smiles = []
    rdmols = []
    for mol in molecule_list:
        rdmol, _ = build_molecule(mol[0], mol[1], model.dataset_info.atom_decoder, return_n_dummy_atoms=True)
        smi = Chem.MolToSmiles(rdmol)
        rdmols.append(rdmol)
        pred_smiles.append(smi)

    if reranker_ckpt is not None:
        ckpt = torch.load(reranker_ckpt, map_location="cpu")
        feat_cfg = ckpt["featurizer"]
        featurizer = RerankFeaturizer(
            radius=int(feat_cfg["radius"]),
            fp_size=int(feat_cfg["fp_size"]),
            extra_numeric_keys=tuple(feat_cfg.get("extra_numeric_keys", ())),
        )
        model_cfg = ckpt.get("model", {})
        reranker = MLPReactionReranker(
            input_dim=featurizer.feature_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        reranker.load_state_dict(ckpt["state_dict"], strict=True)
        reranker = reranker.to(device).eval()

        feats = []
        for smi in pred_smiles:
            f = featurizer(smiles, smi, row=None)
            feats.append(np.zeros((featurizer.feature_dim,), dtype=np.float32) if f is None else f.astype(np.float32))
        X = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = reranker(X).detach().cpu().numpy()
        order = np.argsort(-logits)
        pred_smiles = [pred_smiles[i] for i in order]
        rdmols = [rdmols[i] for i in order]

    for smi in pred_smiles:
        print(smi)

    return rdmols


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', type=str, required=True)
    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=False, default=1)
    parser.add_argument('--n_steps', action='store', type=int, required=False, default=500)
    parser.add_argument('--seed', action='store', type=int, required=False, default=42)
    parser.add_argument('--device', action='store', type=str, required=False, default='cuda:0')

    # Retrieval (for checkpoints trained with retrieval_k > 0)
    parser.add_argument('--retrieval_csv', type=str, required=False, default=None)
    parser.add_argument('--encoded_reactants_pt', type=str, required=False, default=None)
    parser.add_argument('--retrieval_type', type=str, required=False, default='morgan', choices=['morgan', 'sw_morgan'])
    parser.add_argument('--retrieval_max_size', type=int, required=False, default=10)
    parser.add_argument('--fp_radius', type=int, required=False, default=2)
    parser.add_argument('--fp_size', type=int, required=False, default=4096)
    parser.add_argument('--center_method', type=str, required=False, default='brics', choices=['brics', 'none'])
    parser.add_argument('--center_weight', type=float, required=False, default=2.0)
    parser.add_argument('--center_max_distance', type=int, required=False, default=2)
    parser.add_argument('--center_decay', type=float, required=False, default=0.5)

    # Learned reranking
    parser.add_argument('--reranker_ckpt', type=str, required=False, default=None)
    args = parser.parse_args()
    main(
        smiles=args.smiles,
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
        retrieval_csv=args.retrieval_csv,
        encoded_reactants_pt=args.encoded_reactants_pt,
        retrieval_type=args.retrieval_type,
        retrieval_max_size=args.retrieval_max_size,
        fp_radius=args.fp_radius,
        fp_size=args.fp_size,
        center_method=args.center_method,
        center_weight=args.center_weight,
        center_max_distance=args.center_max_distance,
        center_decay=args.center_decay,
        reranker_ckpt=args.reranker_ckpt,
    )
