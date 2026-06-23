#!/usr/bin/env python3
"""
Reproduce i.i.d. synthetic benchmark (Table: synthetic_iid_combined).

Protocol (matches ``sprobes_benchmark_p6_p8_p12.py``):
  - Generate n = mult * p samples (uniform X, clipped y in [0,1])
  - Fixed (X, y) per (p, n) config; only DP noise varies across MC draws
  - Fit on all n points; report coefficient MSE vs true beta

Example:
  python experiments/run_synthetic_iid.py --iters 100 --eps 0.1 1.0
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
from utils.metrics import coefficient_mse
from utils.synthetic import generate_iid_linear

CELL_LIMIT = 200_000
P_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 15, 20, 24]
P_STEINER = {6, 8, 12, 15, 20, 24}
SAMPLE_MULTIPLIERS = [2000, 5000, 10000]


def _delta(n: int) -> float:
    return min(1e-6, 1.0 / (n ** 2))


def run_one(
    X: np.ndarray,
    y: np.ndarray,
    p: int,
    epsilon: float,
    true_beta: np.ndarray,
    sprobes_models: dict,
    seed: int | None = None,
    probes_cache: ProbesCache | None = None,
    Z_adassp: np.ndarray | None = None,
    BZ_adassp: float | None = None,
) -> dict[str, float]:
    """Fit on full data; evaluate coefficient MSE vs true_beta."""
    if seed is not None:
        np.random.seed(seed)

    X = np.clip(X, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    out: dict[str, float] = {}

    def coef_mse(beta: np.ndarray | None) -> float:
        return coefficient_mse(beta, true_beta)

    out["OLS"] = coef_mse(ols(X, y))

    run_probes = (3 ** p) * 2 <= CELL_LIMIT
    if run_probes and probes_cache is not None:
        out["Ada-PROBES"] = coef_mse(ada_probes_cached(probes_cache, epsilon))
    else:
        out["Ada-PROBES"] = float("nan")

    if p in P_STEINER:
        out["S-PROBES"] = coef_mse(sprobes_models[p].fit(X, y, epsilon))
    else:
        out["S-PROBES"] = float("nan")

    if Z_adassp is None or BZ_adassp is None:
        Z_adassp, BZ_adassp = prepare_adassp_design(X)
    out["AdaSSP"] = coef_mse(
        adassp(Z_adassp, y, epsilon, delta=_delta(len(y)), BZ=BZ_adassp, BY=1.0),
    )
    out["DiffPrivLib"] = coef_mse(
        diffprivlib_linear(X, y, epsilon, bounds_X=(0, 1), bounds_y=(0, 1)),
    )
    out["BinAgg"] = coef_mse(
        binagg_linear(X, y, epsilon, delta=_delta(len(y))),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PROBES i.i.d. synthetic benchmark")
    parser.add_argument("--iters", type=int, default=100, help="MC iterations per config")
    parser.add_argument("--eps", type=float, nargs="+", default=[0.1, 1.0])
    parser.add_argument("--p", type=int, nargs="*", default=P_VALUES)
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Seed for data generation (DP noise uses base_seed + iter + 1)",
    )
    parser.add_argument(
        "--out", default=os.path.join(ROOT, "results", "synthetic_iid.csv"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sprobes_models = build_sprobes_models()
    rows = []
    t0 = time.time()

    for p in args.p:
        for mult in SAMPLE_MULTIPLIERS:
            n = mult * p
            X, y, true_beta = generate_iid_linear(n, p, seed=args.base_seed)
            X = np.clip(X, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)

            run_probes = (3 ** p) * 2 <= CELL_LIMIT
            probes_cache = ProbesCache(X, y) if run_probes else None
            Z_adassp, BZ_adassp = prepare_adassp_design(X)

            for eps in args.eps:
                for it in range(args.iters):
                    metrics = run_one(
                        X, y, p, eps, true_beta, sprobes_models,
                        seed=args.base_seed + it + 1,
                        probes_cache=probes_cache,
                        Z_adassp=Z_adassp,
                        BZ_adassp=BZ_adassp,
                    )
                    for method, mse_val in metrics.items():
                        rows.append({
                            "p": p,
                            "n": n,
                            "mult": mult,
                            "epsilon": eps,
                            "iter": it,
                            "method": method,
                            "mse": mse_val,
                        })
                print(
                    f"p={p} n={n} eps={eps} done ({time.time()-t0:.1f}s)",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["p", "n", "epsilon", "method"])["mse"].mean().unstack("method")
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {args.out}")
    print(f"Saved {summary_path}")
    print(summary.round(6))


if __name__ == "__main__":
    main()
