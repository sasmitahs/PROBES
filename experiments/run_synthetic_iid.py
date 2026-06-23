#!/usr/bin/env python3
"""
Reproduce i.i.d. synthetic benchmark (Table: tab:synthetic_iid_combined).

Default protocol (paper):
  - Uniform X, clipped y; fresh data per MC seed
  - Sequential 80/20 train/test split
  - Fit on train; report test MSE on held-out y
  - ε=1.0, 100 seeds

Legacy protocol (--metric coef):
  - Fixed (X, y) per (p, n); only DP noise varies
  - Fit on all n points; coefficient MSE vs true beta

Example:
  python experiments/run_synthetic_iid.py --iters 100 --eps 1.0
  python experiments/generate_iid_latex_table.py
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

from experiments.synthetic_common import (
    METHODS,
    _cell_seed,
    iid_run_one_coef,
    iid_run_one_test,
)
from probes.probes import ProbesCache
from probes.adassp import prepare_adassp_design
from probes.sprobes import build_sprobes_models
from utils.synthetic import generate_iid_linear

CELL_LIMIT = 200_000
P_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 15, 20, 24]
SAMPLE_MULTIPLIERS = [2000, 5000, 10000]


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
    """Backward-compatible coef-MSE entry point for validate imports."""
    return iid_run_one_coef(
        X, y, p, epsilon, true_beta, sprobes_models,
        seed=seed,
        probes_cache=probes_cache,
        Z_adassp=Z_adassp,
        BZ_adassp=BZ_adassp,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PROBES i.i.d. synthetic benchmark")
    parser.add_argument("--iters", type=int, default=100, help="MC iterations per config")
    parser.add_argument("--eps", type=float, nargs="+", default=[1.0])
    parser.add_argument("--p", type=int, nargs="*", default=P_VALUES)
    parser.add_argument(
        "--metric",
        choices=["test", "coef"],
        default="test",
        help="test = 80/20 test MSE (paper); coef = coefficient MSE (author CSV)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42,
        help="Base seed for MC draws",
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
            for eps in args.eps:
                if args.metric == "coef":
                    X, y, true_beta = generate_iid_linear(n, p, seed=args.base_seed)
                    X = np.clip(X, 0.0, 1.0)
                    y = np.clip(y, 0.0, 1.0)
                    run_probes = (3 ** p) * 2 <= CELL_LIMIT
                    probes_cache = ProbesCache(X, y) if run_probes else None
                    Z_adassp, BZ_adassp = prepare_adassp_design(X)
                    for it in range(args.iters):
                        metrics = iid_run_one_coef(
                            X, y, p, eps, true_beta, sprobes_models,
                            seed=args.base_seed + it + 1,
                            probes_cache=probes_cache,
                            Z_adassp=Z_adassp,
                            BZ_adassp=BZ_adassp,
                        )
                        for method, mse_val in metrics.items():
                            rows.append({
                                "p": p, "n": n, "mult": mult, "epsilon": eps,
                                "iter": it, "method": method, "mse": mse_val,
                            })
                else:
                    for it in range(args.iters):
                        seed = _cell_seed(args.base_seed, p, mult, it)
                        metrics = iid_run_one_test(
                            p, eps, n, seed, sprobes_models=sprobes_models,
                        )
                        for method, mse_val in metrics.items():
                            if method == "OLS":
                                continue
                            rows.append({
                                "p": p, "n": n, "mult": mult, "epsilon": eps,
                                "iter": it, "method": method, "mse": mse_val,
                            })
                print(
                    f"p={p} n={n} eps={eps} metric={args.metric} done "
                    f"({time.time()-t0:.1f}s)",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["p", "n", "mult", "epsilon", "method"])["mse"].mean()
    summary = summary.unstack("method")
    summary_path = args.out.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path)
    print(f"Saved {args.out}")
    print(f"Saved {summary_path}")
    print(summary.round(6))


if __name__ == "__main__":
    main()
