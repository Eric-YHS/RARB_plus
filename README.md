# RARB_plus: Retrieval-Augmented Graph Generation for Retrosynthesis

This repository is a research sandbox built on top of **RetroBridge** and includes additional components for:

- **Multi-center structural-weighted fingerprint retrieval** (e.g., BRICS-based multi-center weighting) and exporting both `retrieval_index` and `retrieval_similarity`.
- **Condition injection from retrieved candidates**, including token-level metadata (rank and optionally similarity) to improve generation.
- **Learned selection & reranking** over multiple generated candidates (replacing heuristic ranking).
- **Multi-task optimization on the Graph Transformer**, with an extra classification head predicting the `class` column of USPTO-50k.

<img src="overview.jpg">

## Data
Download the USPTO-50k data and place it under `data/uspto50k/raw/` (as expected by the default configs).

Data link: https://drive.google.com/file/d/13FP-RBetjKZ1T-6gzD_PosLMIcaNLfF7/view?usp=sharing

When you generate new retrieval-indexed CSVs, make sure the filenames match what the dataloader expects (e.g., overwrite/rename to `uspto50k_train.csv`, `uspto50k_val.csv`, `uspto50k_test.csv` or the corresponding `*_unirxnfp.csv` variants when `retrieval_dataset: 50k` is enabled).

## Environment
| Software | Version |
|---|---|
| Python | 3.9 |
| CUDA | 11.6 |

```shell
conda create --name rarb python=3.9 rdkit=2023.09.5 -c conda-forge -y
conda activate rarb
pip install -r requirements.txt
```

## Data format
The training/evaluation CSVs are expected to contain at least:

- `reactants>reagents>production` (reaction SMILES)

For retrieval-augmented training/sampling, the CSV should also contain:

- `retrieval_index` (comma-separated indices into the retrieval pool)
- `retrieval_similarity` (comma-separated floats aligned with `retrieval_index`)

For multi-task training (reaction class prediction), the CSV should also contain:

- `class` (USPTO-50k reaction class label)

Note: if you have previously processed datasets, delete the corresponding `processed_*` folders to re-run preprocessing and include the new fields (`retrieval_sim`, `rxn_class`).

## Build retrieval index (offline)
This repo provides a unified script to build `retrieval_index` + `retrieval_similarity`:

```shell
python scripts/build_retrieval_index.py \
  --input_file data/uspto50k/raw/uspto50k_train.csv \
  --retrieval_file data/uspto50k/raw/uspto50k_train.csv \
  --output_file data/uspto50k/raw/uspto50k_train_sw_morgan.csv \
  --data_type train \
  --retrieval_type sw_morgan \
  --max_size 10
```

Supported retrieval types:

- `morgan`: standard Morgan + Tanimoto similarity
- `sw_morgan`: structural-weighted Morgan + **weighted containment similarity** (designed for retrosynthesis where reactants may contain extra groups)

## Training
Baseline RARB training:

```shell
python train.py --config configs/retrobridge.yaml --model RetroBridge
```

Multi-task training (GraphTransformer + USPTO-50k `class` head):

```shell
python train.py --config configs/retrobridge_multitask.yaml --model RetroBridge
```

Key multi-task / retrieval settings (in YAML):

- `retrieval_token_meta_dim`: set to `2` to inject `(rank, similarity)`; set to `1` for rank-only.
- `rxn_class_num`: number of classes (e.g., `10` for USPTO-50k)
- `rxn_class_loss_weight`: weight of the classification loss
- `rxn_class_offset`: set to `1` if your labels are `1..N`, otherwise `0`

## Sampling
```shell
python sample.py \
  --config configs/retrobridge.yaml \
  --checkpoint checkpoints/RARB.ckpt \
  --samples samples \
  --model RetroBridge \
  --mode test \
  --n_samples 100 \
  --n_steps 500 \
  --sampling_seed 1
```

Optional: attach a learned reranker and export `rerank_logit`/`rerank_score` to the CSV:

```shell
python sample.py \
  --config configs/retrobridge.yaml \
  --checkpoint checkpoints/RARB.ckpt \
  --samples samples \
  --model RetroBridge \
  --mode test \
  --n_samples 100 \
  --n_steps 500 \
  --sampling_seed 1 \
  --reranker_ckpt output/reranker.pt
```

## Learned reranker
Train a reranker from generated samples CSVs:

```shell
python scripts/train_reranker.py \
  --input_csv samples/uspto50k_test/your_samples.csv \
  --output_ckpt output/reranker.pt \
  --epochs 10 \
  --add_confidence
```

Score/rerank an existing samples CSV:

```shell
python scripts/rerank_samples.py \
  --input_csv samples/uspto50k_test/your_samples.csv \
  --reranker_ckpt output/reranker.pt \
  --add_confidence
```

## Single-molecule prediction (with optional online retrieval + reranking)
If your checkpoint was trained with retrieval augmentation (`retrieval_k > 0`), you must provide:

- `--retrieval_csv`: a CSV serving as the retrieval pool
- `--encoded_reactants_pt`: the reactant embedding tensor used by the model

Example:

```shell
python predict.py \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --checkpoint checkpoints/RARB.ckpt \
  --n_samples 10 \
  --n_steps 500 \
  --retrieval_csv data/uspto50k/raw/uspto50k_train.csv \
  --encoded_reactants_pt data/uspto50k/raw/rxn_encoded_react_tensor.pt \
  --retrieval_type sw_morgan \
  --reranker_ckpt output/reranker.pt
```

## Notes
- `openbabel` is optional; if not installed, evaluation will fall back to RDKit-only SMILES validation.
