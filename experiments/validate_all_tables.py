#!/usr/bin/env python3
"""
Validate reproduction outputs against paper table values.

Writes results/validation_all_tables.csv and prints a per-table summary.
Use VALIDATE_ITERS=100 (default), VALIDATE_TOL_PCT=15, VALIDATE_QUICK=1 for spot checks.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.paper_tables import (
    CORR_SYN,
    COVERTYPE_NTRAIN,
    IID_SYN,
    REALWORLD,
)
from experiments.realworld_common import realworld_run_config
from experiments.run_synthetic_correlated import run_one as corr_run_one
from experiments.run_synthetic_iid import run_one as iid_run_one
from experiments.synthetic_common import synth_run_config
from probes.sprobes import build_sprobes_models
from utils.datasets import load_dataset


@dataclass
class Row:
    table: str
    config: str
    method: str
    paper: float
    repo: float
    rel_err_pct: float
    status: str  # OK | MISMATCH | SKIP | NA
    note: str = ""


def _rel_err(ref: float, got: float) -> float:
    if not np.isfinite(ref) or not np.isfinite(got):
        return float("nan")
    if ref == 0:
        return abs(got) * 100
    return abs(got - ref) / abs(ref) * 100


def _flag(ref: float, got: float, tol: float, heavy_tailed: bool = False) -> str:
    if not np.isfinite(got):
        return "NA"
    err = _rel_err(ref, got)
    if not np.isfinite(err):
        return "NA"
    if heavy_tailed and ref < 1.0 and got > 10 * ref:
        return "MISMATCH"
    return "OK" if err <= tol else "MISMATCH"


def validate_realworld(n_iters: int, tol: float, quick: bool) -> list[Row]:
    rows: list[Row] = []
    configs = list(REALWORLD.keys())
    if quick:
        configs = [k for k in configs if k[0] == "California"]

    for dataset, p, eps in sorted(configs):
        cfg = f"{dataset} p={p} eps={eps}"
        try:
            load_dataset(dataset)
        except Exception as exc:
            for method, paper_val in REALWORLD[(dataset, p, eps)].items():
                rows.append(Row(
                    "tab:realworld_combined", cfg, method, paper_val, float("nan"),
                    float("nan"), "SKIP", f"dataset load failed: {exc}",
                ))
            continue
        try:
            got = realworld_run_config(dataset, p, eps, n_iters=n_iters, seed=42)
        except Exception as exc:
            for method, paper_val in REALWORLD[(dataset, p, eps)].items():
                rows.append(Row(
                    "tab:realworld_combined", cfg, method, paper_val, float("nan"),
                    float("nan"), "NA", str(exc),
                ))
            continue
        for method, paper_val in REALWORLD[(dataset, p, eps)].items():
            repo_val = got.get(method, float("nan"))
            err = _rel_err(paper_val, repo_val)
            ht = method in {"DiffPrivLib", "BinAgg"} and paper_val > 10
            status = _flag(paper_val, repo_val, tol, heavy_tailed=ht)
            rows.append(Row(
                "tab:realworld_combined", cfg, method, paper_val, repo_val, err, status,
            ))
    return rows


def validate_iid(n_iters: int, tol: float, quick: bool) -> list[Row]:
    rows: list[Row] = []
    configs = list(IID_SYN.keys())
    if quick:
        configs = [(6, 2000), (6, 10000), (4, 2000)]
    for p, mult in configs:
        n = mult * p
        eps = 1.0
        got = synth_run_config(p, n, eps, n_iters=n_iters, base_seed=42)
        cfg = f"p={p} n={n} eps={eps}"
        for method, paper_val in IID_SYN[(p, mult)].items():
            repo_val = got.get(method, float("nan"))
            err = _rel_err(paper_val, repo_val)
            status = _flag(paper_val, repo_val, tol)
            note = "repo uses coef MSE (benchmark CSV); paper table is test MSE"
            rows.append(Row(
                "tab:synthetic_iid_combined", cfg, method, paper_val, repo_val, err, status, note,
            ))
    return rows


def validate_corr(n_iters: int, tol: float, quick: bool) -> list[Row]:
    rows: list[Row] = []
    sprobes = build_sprobes_models()
    configs = list(CORR_SYN.keys())
    if quick:
        configs = [(4, 1.0), (6, 1.0), (6, 10.0), (12, 1.0)]
        n_iters = min(n_iters, 20)

    for p, eps in configs:
        n = 10_000 * p
        store: dict[str, list[float]] = {}
        for it in range(n_iters):
            m = corr_run_one(p, eps, n, seed=42 + it, sprobes_models=sprobes)
            for method, val in m.items():
                store.setdefault(method, []).append(val)
        cfg = f"p={p} n={n} eps={eps}"
        for method, paper_val in CORR_SYN[(p, eps)].items():
            vals = np.array(store.get(method, [float("nan")]), float)
            vals = vals[np.isfinite(vals)]
            repo_val = float(np.mean(vals)) if len(vals) else float("nan")
            ht = method in {"DiffPrivLib", "BinAgg", "S-PROBES"} and eps <= 1.0
            err = _rel_err(paper_val, repo_val)
            status = _flag(paper_val, repo_val, tol, heavy_tailed=ht)
            rows.append(Row(
                "tab:synthetic_corr", cfg, method, paper_val, repo_val, err, status,
            ))
    return rows


def validate_covertype_ntrain(n_iters: int, tol: float, quick: bool) -> list[Row]:
    rows: list[Row] = []
    configs = list(COVERTYPE_NTRAIN.keys())
    if quick:
        configs = [(100_000, 12)]

    for n_train, p in sorted(configs):
        cfg = f"Covertype n_train={n_train} p={p} eps=10"
        try:
            got = realworld_run_config(
                "Covertype", p, 10.0, n_iters=n_iters, seed=42,
                loader_kwargs={"n_train": n_train},
            )
        except Exception as exc:
            for method, paper_val in COVERTYPE_NTRAIN[(n_train, p)].items():
                rows.append(Row(
                    "tab:covertype_ntrain", cfg, method, paper_val, float("nan"),
                    float("nan"), "SKIP", str(exc),
                ))
            continue
        for method, paper_val in COVERTYPE_NTRAIN[(n_train, p)].items():
            repo_val = got.get(method, float("nan"))
            err = _rel_err(paper_val, repo_val)
            status = _flag(paper_val, repo_val, tol)
            rows.append(Row(
                "tab:covertype_ntrain", cfg, method, paper_val, repo_val, err, status,
            ))
    return rows


def main() -> None:
    n_iters = int(os.environ.get("VALIDATE_ITERS", "100"))
    tol = float(os.environ.get("VALIDATE_TOL_PCT", "15"))
    quick = os.environ.get("VALIDATE_QUICK", "0") == "1"
    t0 = time.time()

    print(f"PROBES table validation  iters={n_iters}  tol={tol}%  quick={quick}")
    all_rows: list[Row] = []
    all_rows.extend(validate_realworld(n_iters, tol, quick))
    all_rows.extend(validate_iid(n_iters, tol, quick))
    all_rows.extend(validate_corr(n_iters, tol, quick))
    all_rows.extend(validate_covertype_ntrain(n_iters, tol, quick))

    df = pd.DataFrame([asdict(r) for r in all_rows])
    out = os.path.join(ROOT, "results", "validation_all_tables.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved {out}  ({time.time()-t0:.1f}s)\n")
    for table in df["table"].unique():
        sub = df[df["table"] == table]
        ok = (sub["status"] == "OK").sum()
        mis = (sub["status"] == "MISMATCH").sum()
        skip = (sub["status"] == "SKIP").sum()
        print(f"=== {table} ===  OK={ok}  MISMATCH={mis}  SKIP={skip}")
        show = sub[sub["status"].isin(["OK", "MISMATCH"])].head(20 if quick else 100)
        for _, r in show.iterrows():
            print(
                f"  [{r['status']:8s}] {r['config']:30s} {r['method']:12s}  "
                f"paper={r['paper']:.6g}  repo={r['repo']:.6g}  err={r['rel_err_pct']:.1f}%"
            )
        if len(sub) > len(show):
            print(f"  ... {len(sub)-len(show)} more rows in CSV")
        print()


if __name__ == "__main__":
    main()
