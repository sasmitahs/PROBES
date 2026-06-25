"""Shared i.i.d. synthetic benchmark logic (coef MSE and test MSE protocols)."""

from __future__ import annotations

import numpy as np

from probes.ada_probes import ada_probes_cached
from probes.adassp import adassp, prepare_adassp_design
from probes.binagg import binagg_linear
from probes.diffprivlib_lr import diffprivlib_linear
from probes.ols import ols
from probes.probes import ProbesCache
from probes.sprobes import build_sprobes_models
from utils.metrics import coefficient_mse, mse, predict
from utils.synthetic import generate_iid_linear, split_train_test

CELL_LIMIT = 200_000
P_STEINER = {6, 8, 12, 15, 20, 24}
METHODS = ["Ada-PROBES", "S-PROBES", "AdaSSP", "DiffPrivLib", "BinAgg"]


def _delta(n: int) -> float:
    return min(1e-6, 1.0 / (n ** 2))


def _cell_seed(base_seed: int, p: int, mult: int, it: int) -> int:
    """Deterministic seed for one (p, mult, iter) cell; stays in NumPy range."""
    raw = base_seed + p * 1_000_003 + mult * 10_007 + it * 97
    return int(raw % (2**32 - 1))


def iid_run_one_coef(
    X: np.ndarray,
    y: np.ndarray,
    p: int,
    epsilon: float,
    true_beta: np.ndarray,
    sprobes_models: dict | None = None,
    seed: int | None = None,
    probes_cache: ProbesCache | None = None,
    Z_adassp: np.ndarray | None = None,
    BZ_adassp: float | None = None,
) -> dict[str, float]:
    """Fit on full data; evaluate coefficient MSE vs true_beta."""
    if sprobes_models is None:
        sprobes_models = build_sprobes_models()
    if seed is not None:
        np.random.seed(seed)

    X = np.clip(X, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    out: dict[str, float] = {}

    def coef_mse(beta: np.ndarray | None) -> float:
        return coefficient_mse(beta, true_beta)

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
    out["BinAgg"] = coef_mse(binagg_linear(X, y, epsilon, delta=_delta(len(y))))
    return out


def iid_run_one_test(
    p: int,
    epsilon: float,
    n: int,
    seed: int,
    sprobes_models: dict | None = None,
) -> dict[str, float]:
    """Fresh data per seed, 80/20 split, fit on train, test MSE on held-out y."""
    if sprobes_models is None:
        sprobes_models = build_sprobes_models()

    X, y, _ = generate_iid_linear(n, p, seed=seed)
    X = np.clip(X, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    X_tr, y_tr, X_te, y_te = split_train_test(X, y)
    np.random.seed(seed + 1)

    run_probes = (3 ** p) * 2 <= CELL_LIMIT
    probes_cache = ProbesCache(X_tr, y_tr) if run_probes else None
    Z_adassp, BZ_adassp = prepare_adassp_design(X_tr)
    n_tr = len(y_tr)

    def test_mse(beta: np.ndarray | None) -> float:
        if beta is None or not np.isfinite(beta).all():
            return float("nan")
        return mse(y_te, predict(beta, X_te))

    out: dict[str, float] = {"OLS": test_mse(ols(X_tr, y_tr))}

    if run_probes and probes_cache is not None:
        out["Ada-PROBES"] = test_mse(ada_probes_cached(probes_cache, epsilon))
    else:
        out["Ada-PROBES"] = float("nan")

    if p in P_STEINER:
        out["S-PROBES"] = test_mse(sprobes_models[p].fit(X_tr, y_tr, epsilon))
    else:
        out["S-PROBES"] = float("nan")

    out["AdaSSP"] = test_mse(
        adassp(Z_adassp, y_tr, epsilon, delta=_delta(n_tr), BZ=BZ_adassp, BY=1.0),
    )
    out["DiffPrivLib"] = test_mse(
        diffprivlib_linear(X_tr, y_tr, epsilon, bounds_X=(0, 1), bounds_y=(0, 1)),
    )
    out["BinAgg"] = test_mse(
        binagg_linear(X_tr, y_tr, epsilon, delta=_delta(n_tr)),
    )
    return out


def synth_run_config(
    p: int, n: int, eps: float, n_iters: int = 100, base_seed: int = 42,
) -> dict[str, float]:
    """Coefficient MSE on fixed (X, y) — author benchmark CSV protocol."""
    sprobes = build_sprobes_models()
    X, y, true_beta = generate_iid_linear(n, p, seed=base_seed)
    X = np.clip(X, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    probes_cache = ProbesCache(X, y) if (3 ** p) * 2 <= CELL_LIMIT else None
    Z, BZ = prepare_adassp_design(X)

    store: dict[str, list[float]] = {m: [] for m in METHODS}
    mult = n // p if p > 0 else n
    for it in range(n_iters):
        metrics = iid_run_one_coef(
            X, y, p, eps, true_beta, sprobes,
            seed=base_seed + it + 1,
            probes_cache=probes_cache,
            Z_adassp=Z,
            BZ_adassp=BZ,
        )
        for method, val in metrics.items():
            store[method].append(val)
    return {m: float(np.nanmean(v)) for m, v in store.items()}


def synth_run_config_test_mse(
    p: int,
    n: int,
    eps: float,
    n_iters: int = 100,
    base_seed: int = 42,
) -> dict[str, float]:
    """Test MSE with 80/20 split and fresh data per MC seed."""
    sprobes = build_sprobes_models()
    mult = n // p if p > 0 else n
    store: dict[str, list[float]] = {m: [] for m in METHODS}
    for it in range(n_iters):
        seed = _cell_seed(base_seed, p, mult, it)
        metrics = iid_run_one_test(p, eps, n, seed, sprobes_models=sprobes)
        for method, val in metrics.items():
            if method in store:
                store[method].append(val)
    return {m: float(np.nanmean(v)) for m, v in store.items()}
