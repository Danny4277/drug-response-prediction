# Drug Response Prediction with Multimodal Deep Learning

Predicting cancer cell-line response to drugs (GDSC2) from **molecular fingerprints** of each drug and **gene-expression profiles** of each cell line, with a **variational autoencoder (VAE)** that compresses 17,737 expression features into a 128-dimensional latent space.

The headline result is not the accuracy number — it is how much that number depends on how the data is split.

## Results

All metrics below are on the **test split**, which was never used for model selection or early stopping. Seeded, with gene expression standardized using training-split statistics only.

### Random split — drugs and cell lines appear in both train and test

| Model | Cell-line input | RMSE ↓ | MAE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---|---|---|---|---|---|
| Raw expression MLP | 17,737 features | **0.980** | **0.740** | **0.870** | **0.934** | **0.904** |
| VAE latent + MLP | 128-d latent | 1.030 | 0.769 | 0.856 | 0.926 | 0.896 |

### Cold-drug split — no test drug is ever seen in training

| Model | Cell-line input | RMSE ↓ | MAE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ |
|---|---|---|---|---|---|---|
| Raw expression MLP | 17,737 features | **2.620** | **1.997** | **0.220** | **0.470** | **0.380** |
| VAE latent + MLP | 128-d latent | 2.683 | 2.039 | 0.182 | 0.429 | 0.336 |

## What these numbers mean

**The random split overstates the model by about 4×.** R² falls from 0.87 to 0.22 when the model is asked about drugs it has never seen. Most of the apparent skill on the random split comes from having seen the same compound against other cell lines during training, not from learning chemistry that transfers. Any claim that this model "predicts drug response" is really a claim about interpolating within known drugs — which is a useful task, but a different one from screening a new compound.

**The VAE's advantage is cost, not accuracy.** Compressing expression 138× costs about 0.014 R² on the random split and buys a far cheaper model:

| | Parameters | Training time |
|---|---|---|
| Raw expression regressor | 38.8 M | 16.2 s/epoch |
| VAE latent regressor | 0.6 M | 2.3 s/epoch (+ 7.3 s one-off VAE pretraining) |

That is **65× fewer parameters and 7.1× faster epochs**, measured back to back in one process on the same device (Apple M-series MPS, batch size 64, best of 3 epochs; see `src/benchmark_cost.py` and `results/cost_benchmark.json`). An earlier version of this project found the VAE *more* accurate; that turned out to be an artifact of the raw-expression model being undertrained, and the gap disappeared once both models got standardized inputs and early stopping. Both numbers are kept here rather than quietly dropping the old one.

<p align="center">
  <img src="assets/vae_pretraining_loss.png" width="45%" alt="VAE pretraining loss">
  <img src="assets/baseline_pred_vs_actual.png" width="40%" alt="Predicted vs actual">
</p>

## Pipeline

1. **Data** — GDSC2 via [PyTDC](https://tdcommons.ai/) (`DrugRes(name="GDSC2")`): 92,703 drug–cell-line pairs over 137 drugs and 805 cell lines.
2. **Drug features** — SMILES → RDKit molecule → 2,048-bit Morgan fingerprint (radius 2); invalid SMILES fall back to a zero vector.
3. **Cell-line features** — 17,737-dimensional gene-expression vector, standardized with training-split mean and standard deviation.
4. **VAE pretraining** — encoder 17,737 → 1,024 → 256 → (μ, log σ²) ∈ ℝ¹²⁸ with a symmetric decoder; loss is reconstruction MSE + β·KL (β = 1e-4). Pretrained only on cell lines present in the training split, so no test-set expression leaks in.
5. **Regression** — a two-tower MLP encodes drug and cell-line features separately, concatenates them, and regresses response under MSE, with early stopping on the validation split (patience 5).

### Memory note

The raw frame repeats a cell line's full expression vector on every row, which is about 4.6 GB in float32. Deduplicating to one row per unique cell line and indexing into that matrix brings the whole dataset under 100 MB and makes the experiments runnable on a laptop.

## Repository layout

```
src/
  run_experiment.py      # training + evaluation for both splits and both models
  benchmark_cost.py      # controlled per-epoch cost comparison
notebooks/
  01_data_pipeline_and_baselines.ipynb   # original exploration: featurization + MLP baselines
  02_vae_pretraining.ipynb               # original VAE pretraining (run after 01, same kernel)
results/                 # metrics written by the scripts
assets/                  # plots
```

The notebooks are the original exploratory work and report validation-split metrics from before standardization and early stopping were added. `src/` supersedes them; the tables above come from `src/run_experiment.py`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/run_experiment.py --split random --model vae
python src/run_experiment.py --split random --model raw
python src/run_experiment.py --split cold_drug --model vae
python src/run_experiment.py --split cold_drug --model raw
```

PyTDC downloads GDSC2 on first run. The scripts use Apple MPS or CUDA when available and fall back to CPU.

## Limitations and next steps

- **No cold-cell-line split yet.** Generalizing to an unseen tumor is a separate axis from generalizing to an unseen drug, and it is the one that matters for a new patient sample.
- **Single seed.** Each number is one run; the spread across seeds has not been measured, so small gaps (such as 0.870 vs 0.856) should not be over-read.
- **Fingerprints are a weak drug representation.** Morgan fingerprints cannot generalize to a scaffold unlike anything in training, which is a likely cause of the cold-drug collapse. A learned molecular encoder (GNN or a pretrained chemical language model) is the obvious next step.
- **No hyperparameter search.** Architecture, learning rate, latent size and β were taken from the original notebooks rather than tuned.
