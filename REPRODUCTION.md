# PROBES — VLDB 2026 Artifact Reproduction Guide

This document describes how to reproduce the empirical results in  
**PROBES: A Bernstein Simplex Approach to Differentially Private OLS Estimation** (VLDB 2026).

## 1. Artifact overview

| Paper table / figure | Script | Status in this repo |
|----------------------|--------|---------------------|
| `tab:realworld_combined` | `experiments/run_realworld.py` | **Runnable** — all 6 datasets via `utils/datasets.py` |
| `tab:covertype_ntrain` | `experiments/run_covertype_ntrain.py` | **Runnable** |
| `tab:synthetic_iid_combined` | `experiments/run_synthetic_iid.py` | **Runnable** — matches author benchmark CSV (coef MSE); paper table uses test MSE |
| `tab:synthetic_corr` | `experiments/run_synthetic_correlated.py` | **Runnable** — test RMSE, 80/20 split, 100 seeds |
| All tables (spot check) | `experiments/validate_all_tables.py` | Compares repo vs embedded paper values → `results/validation_all_tables.csv` |

## 2. System requirements

### Hardware (paper experiments)

Experiments were run on a **single-machine CPU** setup. Representative configuration:

| Component | Minimum | Recommended (full grid) |
|-----------|---------|------------------------|
| CPU | 4 cores | 8–16 cores (Apple Silicon or x86_64) |
| RAM | 16 GB | 32–64 GB (Covertype `n=240k`, S-PROBES p=24) |
| Disk | 5 GB free | 20 GB (datasets + CSV outputs) |
| GPU | Not required | — |

**Validation machine (this artifact check):** macOS 26.2, Apple Silicon (arm64), Python 3.11.14, NumPy 2.4.6.

Runtime order-of-magnitude (100 MC iterations):

| Experiment | Approx. time |
|------------|--------------|
| California Housing, all p≤6, 3 ε | ~5–15 min |
| i.i.d. synthetic, full grid | ~2–6 hours |
| Correlated synthetic, all p, ε∈{1,10} | ~4–12 hours |
| Full real-world (6 datasets) | ~1–2 days |
| Covertype n_train sweep | ~1–8 hours |

Use `VALIDATE_QUICK=1` for a ~10–30 minute smoke test.

### Software

```bash
cd probes-reproduction
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: NumPy, SciPy, pandas, scikit-learn, diffprivlib, BinAgg (from GitHub).

Optional environment variables for threading:

```bash
export OMP_NUM_THREADS=8
export PROBES_N_THREADS=8
```

## 3. Quick validation (recommended first step)

```bash
# Spot-check all tables vs paper values (~15–30 min)
VALIDATE_QUICK=1 VALIDATE_ITERS=100 python experiments/validate_all_tables.py

# View results
column -t -s, results/validation_all_tables.csv | less
```

Columns: `table`, `config`, `method`, `paper`, `repo`, `rel_err_pct`, `status` (`OK` / `MISMATCH` / `SKIP`).

Tolerance default: **15%** relative error (`VALIDATE_TOL_PCT=15`).

## 4. Table-by-table reproduction

### 4.1 Real-world (`tab:realworld_combined`)

**Protocol:** Top-p features by |corr(x,y)| on train; 80/20 split; unit-cube normalization (train-only min–max to [0,1]); test **RMSE** on original scale; **100 MC seeds**; ε ∈ {0.5, 1, 10}; δ = min(10⁻⁶, 1/n_train²) for AdaSSP/BinAgg.

```bash
python experiments/run_realworld.py --dataset all --iters 100 --eps 0.5 1.0 10.0
```

Output: `results/realworld_all.csv`.

**Datasets** — download and set paths:

| Dataset | Variable | Source |
|---------|----------|--------|
| Year Prediction MSD | `PROBES_MSD_CSV` | [UCI YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) |
| NYC Yellow Taxi | `PROBES_TAXI_CSV` | [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Appliances Energy | `PROBES_APPLIANCES_CSV` | [UCI Appliances energy](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) |
| Superconductivity | `PROBES_SUPERCONDUCTIVITY_CSV` | [UCI Superconductivty Data](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) |
| Covertype | `PROBES_COVTYPE_CSV` | [UCI Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype) |

Loaders: `utils/datasets.py` (requires `probes_realworld_eval.py` in `~/Downloads` or set `PROBES_REALWORLD_EVAL`).

### 4.2 Covertype training size (`tab:covertype_ntrain`)

Subsample Covertype to `n_train ∈ {500, 5k, 50k, 100k}`; ε=10; p ∈ {6, 8, 12, 20, 24}.

```bash
python experiments/run_covertype_ntrain.py --iters 100
```

Output: `results/covertype_ntrain.csv`.

### 4.3 i.i.d. synthetic (`tab:synthetic_iid_combined`)

**Paper protocol:** Uniform X ∈ [0,1]^p, clipped linear y, 80/20 split, **test MSE**, ε=1.0, 100 seeds.

**This repo's benchmark script** follows the authors' `sprobes_benchmark_p6_p8_p12.py`:

- Fixed (X, y) per (p, n); only DP noise varies
- Fit on all n points
- Metric: **coefficient MSE** mean((β̂−β*)²)

```bash
python experiments/run_synthetic_iid.py --iters 100 --eps 0.1 1.0
VALIDATE_QUICK=1 python experiments/validate_all_tables.py
```

Reference: `~/Downloads/results/probes_benchmark_p1_p5_eps_grid.csv` (or ship `data/reference/` in artifact).

**Known alignment:** Ada-PROBES / S-PROBES / AdaSSP within ~5–15% of reference at ε=1; DiffPrivLib/BinAgg highly variable at low ε.

### 4.4 Correlated synthetic (`tab:synthetic_corr`)

**Protocol:** Latent-factor block (min-max) + i.i.d. uniform block; n = 10,000p; fresh data per seed; 80/20 split; **test RMSE**; ε ∈ {1.0, 10.0}; 100 seeds.

```bash
python experiments/run_synthetic_correlated.py --iters 100 --eps 1.0 10.0
VALIDATE_QUICK=1 python experiments/validate_all_tables.py
```

**Known alignment (100 iters):**

| Regime | Match quality |
|--------|---------------|
| ε = 10, all p | Excellent (~0–1% for OLS, Ada-PROBES, S-PROBES) |
| ε = 1, p ≤ 8 | Good (~5–15% for main methods) |
| ε = 1, p ≥ 12 | S-PROBES mean can exceed paper (~20–40%); heavy-tailed MC |
| OLS | Matches paper within ~1% all configs |

## 5. Algorithm modules (one file per method)

| File | Method |
|------|--------|
| `probes/probes.py` | PROBES (Algorithm 1) |
| `probes/ada_probes.py` | Ada-PROBES (Algorithm 2) |
| `probes/sprobes.py` | S-PROBES (Steiner blocks; p ∈ {6,8,12,15,20,24}) |
| `probes/steiner_designs.py` | STS(13), AG(2,4), PG(2,4), MOLS BIBD |
| `probes/adassp.py` | AdaSSP |
| `probes/binagg.py` | BinAgg |
| `probes/diffprivlib_lr.py` | DiffPrivLib (Cebere et al. fix) |

S-PROBES Steiner designs per p:

| p | Design | Blocks M |
|---|--------|----------|
| 6, 8 | Fano / AG(2,3) | 32, 44 |
| 12 | STS(13) | 32 |
| 15 | AG(2,4) | 24 |
| 20 | PG(2,4) | 26 |
| 24 | MOLS BIBD(25,5,1) | 36 |

## 6. Expected outputs

All runners write CSV under `results/`:

```
results/
├── synthetic_iid.csv / _summary.csv
├── synthetic_correlated.csv / _summary.csv
├── realworld_all.csv
├── covertype_ntrain.csv
└── validation_all_tables.csv
```

## 7. Reproduction checklist for reviewers

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `VALIDATE_QUICK=1 python experiments/validate_all_tables.py` completes
- [ ] California Housing: Ada-PROBES within ~15% of paper at ε=1, p≤4
- [ ] Correlated ε=10: S-PROBES within ~5% of paper
- [ ] Covertype n_train: methods within ~15% of paper
- [ ] (Optional) Full grids with `--iters 100` overnight

## 8. Known gaps vs paper

1. **i.i.d. synthetic metric:** repo uses coefficient MSE (author benchmark); paper table reports test MSE.
2. **DiffPrivLib / BinAgg** at low ε: heavy-tailed failures; compare medians or use ≥1000 iters for stable means.
3. **S-PROBES at correlated ε=1, large p:** implementation verified bit-identical to author code; remaining gap is MC variance / seed sensitivity.

## 9. Citation

```bibtex
@inproceedings{probes2026,
  title={PROBES: A Bernstein Simplex Approach to Differentially Private OLS Estimation},
  author={Hariini S, Sasmita and Tandon, Anshoo},
  booktitle={Proceedings of the VLDB Endowment},
  year={2026}
}
```
