# PROBES Reproduction

Reproduction code for **PROBES: A Bernstein Simplex Approach to Differentially Private OLS Estimation**.

Each differentially private algorithm lives in its own module under `probes/`. Experiment scripts under `experiments/` reproduce the synthetic and real-world tables from the paper.

## Repository layout

```
probes-reproduction/
├── probes/
│   ├── bernstein.py       # Shared Bernstein basis utilities
│   ├── ols.py             # Non-private OLS baseline
│   ├── probes.py          # PROBES (Algorithm 1): single-shot Bernstein release
│   ├── ada_probes.py      # Ada-PROBES (Algorithm 2): ridge post-processing
│   ├── sprobes.py         # S-PROBES: Steiner-block extension (p ≥ 6)
│   ├── steiner_designs.py # Combinatorial block definitions
│   ├── adassp.py          # AdaSSP baseline (ε,δ)-DP
│   ├── binagg.py          # BinAgg baseline (Gaussian DP)
│   └── diffprivlib_lr.py  # DiffPrivLib baseline (corrected sensitivity)
├── utils/
│   ├── metrics.py         # RMSE / MSE helpers
│   ├── normalization.py   # Unit-cube protocol ([0,1] train-only scaling)
│   ├── synthetic.py       # Synthetic data generators
│   └── datasets.py        # Real-world dataset loaders
├── experiments/
│   ├── run_synthetic_iid.py
│   ├── run_synthetic_correlated.py
│   ├── run_realworld.py
│   ├── run_covertype_ntrain.py
│   └── validate_all_tables.py
├── data/                  # Place downloaded datasets here
└── results/               # CSV outputs from experiments
```

## Algorithms (one file each)

| File | Method | Privacy | Paper reference |
|------|--------|---------|-----------------|
| `probes/probes.py` | **PROBES** | pure ε-DP | Algorithm 1 |
| `probes/ada_probes.py` | **Ada-PROBES** | pure ε-DP (post-processing) | Algorithm 2, Eq. (ridge_solve) |
| `probes/sprobes.py` | **S-PROBES** | pure ε-DP | Algorithm 3 (S-PROBES-3), §6 |
| `probes/adassp.py` | AdaSSP | (ε,δ)-DP | Wang (2018) |
| `probes/binagg.py` | BinAgg | (ε,δ)-DP via μ-GDP | Lin et al. (2025) |
| `probes/diffprivlib_lr.py` | DiffPrivLib | ε-DP (functional mechanism) | Cebere et al. (2026) fix |

## Setup

```bash
cd probes-reproduction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: BinAgg is installed from GitHub (`binagg` in `requirements.txt`). DiffPrivLib is required for the DiffPrivLib baseline.

## Quick sanity check

```python
import numpy as np
from probes.ada_probes import ada_probes
from probes.sprobes import build_sprobes_models
from utils.synthetic import generate_iid_linear

X, y, beta = generate_iid_linear(5000, p=4, seed=0)
b = ada_probes(X, y, epsilon=1.0)
print("Ada-PROBES ||beta - true||:", np.linalg.norm(b - beta))

model = build_sprobes_models()[6]
b6 = model.fit(X[:, :6], y, epsilon=1.0)  # use p=6 features
```

At large ε, private coefficients should match non-private OLS to numerical precision.

## Reproduce experiments

### i.i.d. synthetic (Table `tab:synthetic_iid_combined`)

```bash
python experiments/run_synthetic_iid.py --iters 100 --eps 0.1 1.0
```

Grid: `p ∈ {1,…,24}`, `n ∈ {2000p, 5000p, 10000p}`. Fixed `(X, y)` per `(p, n)` config; MC iterations re-draw only DP noise. Fit on all `n` points and report **coefficient MSE** `mean((β̂ − β*)²)`. Methods: OLS / Ada-PROBES / S-PROBES / AdaSSP / DiffPrivLib / BinAgg.

### Correlated synthetic (Table `tab:synthetic_corr`)

```bash
python experiments/run_synthetic_correlated.py --iters 100 --eps 1.0 10.0
```

Grid: `p ∈ {4,5,6,8,12,15,20,24}`, `n = 10,000p`. Fresh correlated data per MC seed (latent-factor block + i.i.d. block); sequential **80/20 train/test split**; fit on train and report **test RMSE** on held-out `y`. Methods: OLS / Ada-PROBES / S-PROBES / AdaSSP / DiffPrivLib / BinAgg.

### Real-world — all datasets (Table `tab:realworld_combined`)

```bash
python experiments/run_realworld.py --dataset all --iters 100 --eps 0.5 1.0 10.0
```

### Covertype training size (`tab:covertype_ntrain`)

```bash
python experiments/run_covertype_ntrain.py --iters 100
```

### Real-world — California Housing demo

```bash
python experiments/run_realworld.py --iters 100 --eps 0.5 1.0 10.0 --p 1 2 3 4 5 6
```

Uses the **unit-cube protocol**: train-only min–max to `[0,1]`, fit all methods on normalized data, map coefficients back to original scale for test RMSE.

### Full real-world suite

The paper evaluates six datasets (California Housing, NYC Yellow Taxi, Appliances Energy, Superconductivity, Year Prediction MSD, Covertype). Place files under `data/` or set environment variables:

| Dataset | Env variable | Notes |
|---------|--------------|-------|
| Year Prediction MSD | `PROBES_MSD_CSV` | `YearPredictionMSD.txt.bz2` from UCI |
| NYC Yellow Taxi | `PROBES_TAXI_CSV` | Parquet/CSV from TLC |
| Appliances Energy | `PROBES_APPLIANCES_CSV` | `energydata_complete.csv` |
| Superconductivity | `PROBES_SUPERCONDUCTIVITY_CSV` | `train.csv` |
| Covertype | `PROBES_COVTYPE_CSV` | `covtype.data.gz` |

California loads via sklearn with no extra files. Other datasets need `data/` files or `probes_realworld_eval.py` (`PROBES_REALWORLD_EVAL`).

## Key implementation details

1. **Single-shot PROBES release**: one `Lap(1/ε)` draw on the joint vector of size `2·3^p` (not separate releases on y / ȳ cells).
2. **Ada-PROBES regularization**: `ρ = (2/ε)·3^((p-1)/2)`, `λ = min(λ_min(Â), 0)`.
3. **S-PROBES**: `M = (# Steiner blocks) + (# diagonal blocks)`; noise scale `M/ε` per cell; ridge `ρ = max_k 2^(k_d/2)·M/ε`.
4. **Ada-PROBES skipped** when `3^p × 2 > 200,000` cells (tractability limit from experiments).
5. **DiffPrivLib**: diagonal `x²` sensitivity uses `(bounds_X[0][i], bounds_X[1][i])` per Cebere et al. (2026).

For full VLDB artifact instructions (hardware, all tables, checklist), see **[REPRODUCTION.md](REPRODUCTION.md)**.

Validate all tables against paper values:

```bash
VALIDATE_QUICK=1 VALIDATE_ITERS=100 python experiments/validate_all_tables.py
# Full spot grid (~hours):
python experiments/validate_all_tables.py
```


If you use this code, please cite the PROBES paper (VLDB 2026).

## License

Research reproduction code. Dataset licenses apply to downloaded benchmarks.
