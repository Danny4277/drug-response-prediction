"""Controlled cost comparison: raw-expression vs VAE-latent regressor.

Same split, same batch size, same device, 3 timed epochs each, run
back to back in one process so machine state is comparable.
"""
import json, sys, time, torch, torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from torch.utils.data import DataLoader, TensorDataset
from run_experiment import (SEED, build_tables, encode_all, get_device,
                            index_split, pretrain_vae, set_seed,
                            RawExpressionRegressor, VAEDrugResponseRegressor)

EPOCHS = 3
set_seed(); device = get_device()
from tdc.multi_pred import DrugRes
splits = DrugRes(name="GDSC2").get_split(method="random", seed=SEED)
drug_mat, drug_idx, cell_mat, cell_idx = build_tables(splits)
tr = index_split(splits["train"], drug_idx, cell_idx)
cell_mat = torch.tensor(cell_mat); drug_mat = torch.tensor(drug_mat)
train_cells = torch.unique(tr[1])
mu = cell_mat[train_cells].mean(0, keepdim=True)
sd = cell_mat[train_cells].std(0, keepdim=True).clamp_min(1e-6)
cell_mat = (cell_mat - mu) / sd

loader = DataLoader(TensorDataset(*tr), batch_size=64, shuffle=True)
loss_fn = nn.MSELoss()

def timed(model, feats, label):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    per = []
    for _ in range(EPOCHS):
        t = time.time(); model.train()
        for di, ci, y in loader:
            out = model(drug_mat[di].to(device), feats[ci].to(device))
            loss = loss_fn(out, y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "mps": torch.mps.synchronize()
        per.append(time.time() - t)
    n = sum(p.numel() for p in model.parameters())
    print(f"{label}: {min(per):.1f}s/epoch (best of {EPOCHS}), {n/1e6:.1f}M params")
    return {"label": label, "sec_per_epoch": round(min(per), 1),
            "params_millions": round(n / 1e6, 1), "all_epochs": [round(x,1) for x in per]}

t0 = time.time()
vae = pretrain_vae(cell_mat[train_cells], device, epochs=30)
vae_pre = round(time.time() - t0, 1)
latents = encode_all(vae, cell_mat, device)
print(f"VAE pretraining (one-off, 30 epochs): {vae_pre}s")

res = {"vae_pretrain_sec": vae_pre,
       "runs": [timed(RawExpressionRegressor(cell_mat.shape[1]), cell_mat, "raw-expression regressor"),
                timed(VAEDrugResponseRegressor(), latents, "VAE-latent regressor")]}
(Path(__file__).resolve().parents[1] / "results").mkdir(exist_ok=True)
(Path(__file__).resolve().parents[1] / "results/cost_benchmark.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
