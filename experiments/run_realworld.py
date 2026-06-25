#!/usr/bin/env python3
"""
Run real-world benchmarks (unit-cube protocol, 80/20 split).

Supports California, NYC-Taxi, Appliances, Superconductivity, YearMSD, and
Covertype. Set PROBES_* env vars for local CSV paths (see utils/datasets.py).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.realworld_common import evaluate_config
from probes.sprobes import build_sprobes_models
from utils.datasets import (
    REALWORLD_DATASETS,
    load_dataset,
    p_values_for_dataset,
)

DEFAULT_EPS = [0.5, 1.0, 10.0]


def main() -> None:
    parser = argparse.ArgumentParser(description="PROBES real-world benchmark")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["California"],
        help=f"Dataset name(s) or 'all'. Choices: {REALWORLD_DATASETS}",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, nargs="+", default=DEFAULT_EPS)
    parser.add_argument("--p", type=int, nargs="*", default=None, help="Override p grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "realworld_all.csv"),
    )
    args = parser.parse_args()

    if len(args.dataset) == 1 and args.dataset[0].lower() == "all":
        datasets = REALWORLD_DATASETS
    else:
        datasets = args.dataset

    sprobes_models = build_sprobes_models()
    dfs: list[pd.DataFrame] = []
    t0 = time.time()

    for ds in datasets:
        print(f"\n=== Loading {ds} ===", flush=True)
        try:
            X, y, names = load_dataset(ds)
        except Exception as exc:
            print(f"  SKIP {ds}: {exc}", flush=True)
            continue
        p_grid = args.p if args.p else p_values_for_dataset(ds, X.shape[1])
        print(f"  n={len(y):,} d={X.shape[1]}  p_grid={p_grid}", flush=True)

        for p in p_grid:
            for eps in args.eps:
                df = evaluate_config(
                    X, y, names, p, eps, args.iters, args.seed,
                    sprobes_models, dataset=ds,
                )
                dfs.append(df)
                print(
                    f"  {ds} p={p} eps={eps} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    if not dfs:
        print("No results produced.")
        return

    out = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    for ds in out["dataset"].unique():
        sub = out[out["dataset"] == ds]
        pivot = sub.pivot_table(
            index=["p", "epsilon"], columns="method", values="rmse_mean", aggfunc="mean",
        )
        print(f"\n{ds}:")
        print(pivot.round(4))


if __name__ == "__main__":
    main()
