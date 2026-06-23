#!/usr/bin/env python3
"""
Reproduce correlated synthetic benchmark (Table: tab:synthetic_corr).

Protocol (paper Section 7.2):
  - Correlated block: Z @ A.T with column min-max to [0,1]; rest i.i.d. Uniform[0,1]
  - n = 10,000 × p; fresh data per MC seed
  - Sequential 80/20 train/test split; fit on train; report test RMSE on held-out y
  - DP noise seed: data_seed + 1

Example:
  python experiments/run_synthetic_correlated.py --iters 100 --eps 0.1 1.0 10.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from probes.ada_probes import ada_probes_cached
from probes.adassp import adassp, prepare_adassp_design
from probes.binagg import binagg_linear
from probes.diffprivlib_lr import diffprivlib_linear
from probes.ols import ols
from probes.probes import ProbesCache
from probes.sprobes import build_sprobes_models
from utils.metrics import predict, rmse
from utils.synthetic import generate_correlated_linear_split

CELL_LIMIT = 200_000
P_VALUES = [4, 5, 6, 8, 12, 15, 20, 24]
P_STEINER = {6, 8, 12, 15, 20, 24}


def _delta(n: int) -> float:
    return min(1e-6, 1.0 / (n ** 2))


def run_one(
    p: int,
    epsilon: float,
    n: int,
    seed: int,
    sprobes_models: dict,
) -> dict[str, float]:
    """Generate one split, fit on train, evaluate test RMSE."""
    X_tr, y_tr, X_te, y_te = generate_correlated_linear_split(n, p, seed=seed)
    np.random.seed(seed + 1)

    X_tr = np.clip(X_tr, 0.0, 1.0)
    y_tr = np.clip(y_tr, 0.0, 1.0)
    X_te = np.clip(X_te, 0.0, 1.0)

    run_probes = (3 ** p) * 2 <= CELL_LIMIT
    probes_cache = ProbesCache(X_tr, y_tr) if run_probes else None
    Z_adassp, BZ_adassp = prepare_adassp_design(X_tr)

    def test_rmse(beta: np.ndarray | None) -> float:
        if beta is None or not np.isfinite(beta).all():
            return float("nan")
        return rmse(y_te, predict(beta, X_te))

    out: dict[str, float] = {"OLS": test_rmse(ols(X_tr, y_tr))}

    if run_probes and probes_cache is not None:
        out["Ada-PROBES"] = test_rmse(ada_probes_cached(probes_cache, epsilon))
    else:
        out["Ada-PROBES"] = float("nan")

    if p in P_STEINER:
        out["S-PROBES"] = test_rmse(sprobes_models[p].fit(X_tr, y_tr, epsilon))
    else:
        out["S-PROBES"] = float("nan")

    out["AdaSSP"] = test_rmse(
        adassp(Z_adassp, y_tr, epsilon, delta=_delta(len(y_tr)), BZ=BZ_adassp, BY=1.0),
    )
    out["DiffPrivLib"] = test_rmse(
        diffprivlib_linear(X_tr, y_tr, epsilon, bounds_X=(0, 1), bounds_y=(0, 1)),
    )
    out["BinAgg"] = test_rmse(
        binagg_linear(X_tr, y_tr, epsilon, delta=_delta(len(y_tr))),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PROBES correlated synthetic benchmark")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--p", type=int, nargs="*", default=P_VALUES)
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Data seed per iter is base_seed + iter; DP noise uses seed + 1",
    )
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "results", "synthetic_correlated.csv"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sprobes_models = build_sprobes_models()
    rows = []
    t0 = time.time()

    for p in args.p:
        n = 10_000 * p
        for eps in args.eps:
            for it in range(args.iters):
                metrics = run_one(
                    p, eps, n, seed=args.base_seed + it, sprobes_models=sprobes_models,
                )
                for method, rmse_val in metrics.items():
                    rows.append({
                        "p": p,
                        "n": n,
                        "epsilon": eps,
                        "iter": it,
                        "method": method,
                        "rmse": rmse_val,
                    })
            print(f"p={p} n={n} eps={eps} done ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["p", "epsilon", "method"])["rmse"].mean().unstack("method")
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {args.out}")
    print(f"Saved {summary_path}")
    print(summary.round(4))


if __name__ == "__main__":
    main()
