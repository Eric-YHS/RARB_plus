import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.rerank import MLPReactionReranker, RerankFeaturizer
from src.rerank.features import add_confidence_feature


def canonicalize_smiles(smi: str):
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, nargs="+", required=True)
    parser.add_argument("--output_ckpt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--extra_numeric_keys", type=str, nargs="*", default=[])
    parser.add_argument("--add_confidence", action="store_true", default=False)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dfs = [pd.read_csv(p, index_col=False) for p in args.input_csv]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    required_cols = {"product", "pred", "true"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    if args.add_confidence and "confidence" not in df.columns:
        df = add_confidence_feature(df)

    featurizer = RerankFeaturizer(
        radius=args.radius,
        fp_size=args.fp_size,
        extra_numeric_keys=tuple(args.extra_numeric_keys),
    )

    xs: List[np.ndarray] = []
    ys: List[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing"):
        product = row["product"]
        pred = row["pred"]
        true = row["true"]

        canon_pred = canonicalize_smiles(pred)
        canon_true = canonicalize_smiles(true)
        label = 1.0 if (canon_pred is not None and canon_true is not None and canon_pred == canon_true) else 0.0

        feat = featurizer(product, pred, row=row.to_dict())
        if feat is None:
            continue
        xs.append(feat)
        ys.append(label)

    if len(xs) == 0:
        raise RuntimeError("No valid training pairs after featurization.")

    X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
    y = torch.tensor(np.asarray(ys, dtype=np.float32), dtype=torch.float32)

    # Train/val split
    n = X.size(0)
    idx = torch.randperm(n)
    n_val = int(n * args.val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = MLPReactionReranker(input_dim=featurizer.feature_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Handle imbalance
    pos = float(y_train.sum().item())
    neg = float((y_train.numel() - y_train.sum()).item())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} pos_weight={pos_weight.item():.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "featurizer": asdict(featurizer),
                    "model": {"hidden_dim": args.hidden_dim, "dropout": args.dropout},
                    "best_val_loss": best_val,
                },
                args.output_ckpt,
            )

    print(f"Saved reranker checkpoint: {args.output_ckpt}")


if __name__ == "__main__":
    main()
