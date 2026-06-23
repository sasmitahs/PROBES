#!/usr/bin/env python3
"""
Covertype training-size sweep (tab:covertype_ntrain).

Subsample Covertype so ~n_train rows land in the train fold after 80/20 split;
evaluate at epsilon=10 for p in {6, 8, 12, 20, 24}.
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
from utils.datasets import load_covertype

N_TRAIN_GRID = [500, 5_000, 50_000, 100_000]
P_GRID = [6, 8, 12, 20, 24]
EPS = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Covertype n_train sweep (tab:covertype_ntrain)")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, nargs="+", default=N_TRAIN_GRID)
    parser.add_argument("--p", type=int, nargs="+", default=P_GRID)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "results", "covertype_ntrain.csv"),
    )
    args = parser.parse_args()

    sprobes = build_sprobes_models()
    dfs: list[pd.DataFrame] = []
    t0 = time.time()

    for n_train in args.n_train:
        print(f"\n=== Covertype n_train≈{n_train:,} ===", flush=True)
        X, y, names = load_covertype(n_train=n_train, random_state=args.seed)
        print(f"  loaded n={len(y):,} d={X.shape[1]}", flush=True)

        for p in args.p:
            if p > X.shape[1]:
                print(f"  skip p={p} (> d={X.shape[1]})", flush=True)
                continue
            df = evaluate_config(
                X, y, names, p, args.eps, args.iters, args.seed,
                sprobes, dataset="Covertype",
            )
            df["n_train_target"] = n_train
            dfs.append(df)
            print(f"  p={p} eps={args.eps} elapsed={time.time()-t0:.1f}s", flush=True)

    out_df = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")
    pivot = out_df.pivot_table(
        index=["n_train_target", "p"],
        columns="method",
        values="rmse_mean",
        aggfunc="mean",
    )
    print(pivot.round(2))


if __name__ == "__main__":
    main()
