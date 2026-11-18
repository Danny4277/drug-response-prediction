"""Train and evaluate the VAE + multimodal regressor on GDSC2 drug response.

Reports metrics on the held-out TEST split for two split strategies:
  random     - the default PyTDC split (drugs and cell lines seen in training)
  cold_drug  - no drug in the test split appears in training

Usage:
    python src/run_experiment.py --split random
    python src/run_experiment.py --split cold_drug
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

RDLogger.DisableLog("rdApp.*")

N_BITS = 2048
LATENT_DIM = 128
SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def smiles_to_fp(smiles, gen):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_BITS, dtype=np.float32)
    return np.array(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)


def build_tables(splits):
    """Deduplicate drugs and cell lines across all splits.

    The raw frame repeats a cell line's full 17,737-dim expression vector on
    every row (4.6 GB in float32). Storing one row per unique cell line and
    indexing into it keeps the whole dataset under ~100 MB.
    """
    frames = list(splits.values())
    allrows = pd.concat(frames, ignore_index=True)

    drug_ids = allrows["Drug_ID"].drop_duplicates().tolist()
    drug_smiles = allrows.drop_duplicates("Drug_ID").set_index("Drug_ID")["Drug"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    drug_mat = np.stack([smiles_to_fp(drug_smiles[d], gen) for d in drug_ids])
    drug_idx = {d: i for i, d in enumerate(drug_ids)}

    cell_ids = allrows["Cell Line_ID"].drop_duplicates().tolist()
    cell_rows = allrows.drop_duplicates("Cell Line_ID").set_index("Cell Line_ID")["Cell Line"]
    cell_mat = np.stack([np.asarray(cell_rows[c], dtype=np.float32) for c in cell_ids])
    cell_idx = {c: i for i, c in enumerate(cell_ids)}

    print(f"  unique drugs: {len(drug_ids)}  fingerprint matrix {drug_mat.shape}")
    print(f"  unique cell lines: {len(cell_ids)}  expression matrix {cell_mat.shape}")
    return drug_mat, drug_idx, cell_mat, cell_idx


def index_split(df, drug_idx, cell_idx):
    d = torch.tensor([drug_idx[x] for x in df["Drug_ID"]], dtype=torch.long)
    c = torch.tensor([cell_idx[x] for x in df["Cell Line_ID"]], dtype=torch.long)
    y = torch.tensor(df["Y"].values, dtype=torch.float32)
    return d, c, y


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class GeneExpressionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, latent_dim)
        self.logvar_layer = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=1e-4):
    recon_loss = nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


class RawExpressionRegressor(nn.Module):
    """Baseline: the deeper two-tower MLP on raw (standardized) expression."""

    def __init__(self, input_dim):
        super().__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(N_BITS, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU())
        self.cell_encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU())
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, drug, cell):
        combined = torch.cat(
            (self.drug_encoder(drug), self.cell_encoder(cell)), dim=1)
        return self.regression_head(combined).squeeze(-1)


class VAEDrugResponseRegressor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.drug_encoder = nn.Sequential(
            nn.Linear(N_BITS, 256), nn.ReLU(), nn.Dropout(0.2))
        self.cell_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(0.2))
        self.regression_head = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, drug, cell_latent):
        combined = torch.cat(
            (self.drug_encoder(drug), self.cell_encoder(cell_latent)), dim=1)
        return self.regression_head(combined).squeeze(-1)


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------
def pretrain_vae(cell_train, device, epochs=30):
    """Pretrain on the unique cell lines that appear in the training split."""
    vae = GeneExpressionVAE(cell_train.shape[1]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    loader = DataLoader(TensorDataset(cell_train), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        vae.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            loss, _, _ = vae_loss(recon, batch, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    VAE epoch {epoch+1:3d}: loss={total/len(loader):.4f}")
    return vae


@torch.no_grad()
def encode_all(vae, cell_mat, device, chunk=256):
    vae.eval()
    out = []
    for i in range(0, len(cell_mat), chunk):
        mu, _ = vae.encode(cell_mat[i:i + chunk].to(device))
        out.append(mu.cpu())
    return torch.cat(out)


def run_epoch(model, loader, drug_mat, latents, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for di, ci, y in loader:
            drug = drug_mat[di].to(device)
            cell = latents[ci].to(device)
            y = y.to(device)
            out = model(drug, cell)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item()
            if not train:
                preds.append(out.cpu())
                labels.append(y.cpu())
    if train:
        return total / len(loader), None, None
    return total / len(loader), torch.cat(preds).numpy(), torch.cat(labels).numpy()


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]),
        "Spearman": float(spearmanr(y_true, y_pred)[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "cold_drug"], default="random")
    ap.add_argument("--model", choices=["vae", "raw"], default="vae")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--vae-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    set_seed()
    device = get_device()
    t0 = time.time()
    tag = f"{args.split}_{args.model}"
    print(f"[{tag}] device={device}")

    from tdc.multi_pred import DrugRes
    data = DrugRes(name="GDSC2")
    if args.split == "random":
        splits = data.get_split(method="random", seed=SEED)
    else:
        splits = data.get_split(method="cold_split", column_name="Drug_ID", seed=SEED)
    print("  rows:", {k: len(v) for k, v in splits.items()})

    drug_mat, drug_idx, cell_mat, cell_idx = build_tables(splits)
    tr = index_split(splits["train"], drug_idx, cell_idx)
    va = index_split(splits["valid"], drug_idx, cell_idx)
    te = index_split(splits["test"], drug_idx, cell_idx)

    if args.split == "cold_drug":
        overlap = set(tr[0].tolist()) & set(te[0].tolist())
        print(f"  drug overlap train/test: {len(overlap)} (expected 0)")

    # Standardize gene expression with TRAIN-split statistics only.
    cell_mat = torch.tensor(cell_mat)
    train_cells = torch.unique(tr[1])
    mu = cell_mat[train_cells].mean(0, keepdim=True)
    sd = cell_mat[train_cells].std(0, keepdim=True).clamp_min(1e-6)
    cell_mat = (cell_mat - mu) / sd
    drug_mat = torch.tensor(drug_mat)

    if args.model == "vae":
        print("  pretraining VAE on", len(train_cells), "training cell lines")
        vae = pretrain_vae(cell_mat[train_cells], device, epochs=args.vae_epochs)
        latents = encode_all(vae, cell_mat, device)
        print(f"  latent matrix {tuple(latents.shape)}")
        model = VAEDrugResponseRegressor().to(device)
    else:
        latents = cell_mat
        print(f"  raw expression matrix {tuple(latents.shape)}")
        model = RawExpressionRegressor(cell_mat.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    loaders = {
        "train": DataLoader(TensorDataset(*tr), batch_size=64, shuffle=True),
        "valid": DataLoader(TensorDataset(*va), batch_size=256),
        "test": DataLoader(TensorDataset(*te), batch_size=256),
    }

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(args.epochs):
        trl, _, _ = run_epoch(model, loaders["train"], drug_mat, latents, device, loss_fn, opt)
        vl, vp, vy = run_epoch(model, loaders["valid"], drug_mat, latents, device, loss_fn)
        flag = ""
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        print(f"    epoch {epoch+1:3d}: train={trl:.4f} valid={vl:.4f}{flag}")
        if bad >= args.patience:
            print(f"    early stop (no improvement in {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    _, vp, vy = run_epoch(model, loaders["valid"], drug_mat, latents, device, loss_fn)
    _, tp, ty = run_epoch(model, loaders["test"], drug_mat, latents, device, loss_fn)

    result = {
        "split": args.split,
        "model": args.model,
        "n_train": len(tr[2]), "n_valid": len(va[2]), "n_test": len(te[2]),
        "validation": metrics(vy, vp),
        "test": metrics(ty, tp),
        "runtime_sec": round(time.time() - t0, 1),
    }
    print(json.dumps(result, indent=2))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print("wrote", out / f"{tag}.json")


if __name__ == "__main__":
    main()
