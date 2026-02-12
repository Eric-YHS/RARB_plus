import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.rerank import MLPReactionReranker, RerankFeaturizer
from src.rerank.features import add_confidence_feature


def _load_reranker(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feat_cfg = ckpt["featurizer"]
    featurizer = RerankFeaturizer(
        radius=int(feat_cfg["radius"]),
        fp_size=int(feat_cfg["fp_size"]),
        extra_numeric_keys=tuple(feat_cfg.get("extra_numeric_keys", ())),
    )
    model_cfg = ckpt.get("model", {})
    model = MLPReactionReranker(
        input_dim=featurizer.feature_dim,
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, featurizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--reranker_ckpt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--add_confidence", action="store_true", default=False)
    args = parser.parse_args()

    if args.output_csv is None:
        base, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{base}.reranked{ext}"

    df = pd.read_csv(args.input_csv, index_col=False)
    required_cols = {"product", "pred"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    if args.add_confidence and "confidence" not in df.columns:
        df = add_confidence_feature(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, featurizer = _load_reranker(args.reranker_ckpt, device=device)

    feats: List[np.ndarray] = []
    valid_mask: List[bool] = []
    rows = df.to_dict(orient="records")
    for row in tqdm(rows, desc="Featurizing"):
        feat = featurizer(row["product"], row["pred"], row=row)
        if feat is None:
            valid_mask.append(False)
            feats.append(np.zeros((featurizer.feature_dim,), dtype=np.float32))
        else:
            valid_mask.append(True)
            feats.append(feat.astype(np.float32))

    X = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)

    logits = torch.empty((X.size(0),), dtype=torch.float32)
    with torch.no_grad():
        for start in tqdm(range(0, X.size(0), args.batch_size), desc="Scoring"):
            xb = X[start : start + args.batch_size].to(device)
            logits[start : start + xb.size(0)] = model(xb).cpu()

    probs = torch.sigmoid(logits)

    df_out = df.copy()
    df_out["rerank_valid"] = valid_mask
    df_out["rerank_logit"] = logits.numpy()
    df_out["rerank_score"] = probs.numpy()
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
