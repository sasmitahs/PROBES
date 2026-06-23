"""Shared real-world evaluation logic (unit-cube protocol)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from probes.ada_probes import ada_probes_cached
from probes.adassp import adassp, prepare_adassp_design
from probes.binagg import binagg_linear
from probes.diffprivlib_lr import diffprivlib_linear
from probes.ols import ols
from probes.probes import ProbesCache
from probes.sprobes import build_sprobes_models
from utils.datasets import load_dataset
from utils.metrics import rmse
from utils.normalization import build_affine_norm, select_top_p_by_correlation

CELL_LIMIT = 200_000


def _delta(n: int) -> float:
    return min(1e-6, 1.0 / (n ** 2))


def evaluate_config(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    p: int,
    epsilon: float,
    n_iters: int,
    seed: int,
    sprobes_models: dict,
    dataset: str = "",
) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed,
    )
    idx = select_top_p_by_correlation(X_train, y_train, p)
    aff = build_affine_norm(X_train, y_train, idx, feature_names)
    X_tr_01, y_tr_01 = aff.transform(X_train[:, idx], y_train)

    Z, BZ = prepare_adassp_design(X_tr_01)
    ztz = Z.T @ Z
    zty = Z.T @ y_tr_01
    bounds_x = (np.zeros(p), np.ones(p))
    bounds_y = (0.0, 1.0)

    ols_beta = ols(X_tr_01, y_tr_01)
    ols_rmse = rmse(y_test, aff.predict_orig(X_test, ols_beta))

    run_probes = (3 ** p) * 2 <= CELL_LIMIT
    probes_cache = ProbesCache(X_tr_01, y_tr_01) if run_probes else None
    sprobes_model = sprobes_models.get(p)

    store: dict[str, list[float]] = {
        m: [] for m in ["Ada-PROBES", "S-PROBES", "AdaSSP", "DiffPrivLib", "BinAgg"]
    }

    for it in range(n_iters):
        np.random.seed(seed + 10_000 * p + it)

        if run_probes and probes_cache is not None:
            beta = ada_probes_cached(probes_cache, epsilon)
            if beta is not None:
                store["Ada-PROBES"].append(rmse(y_test, aff.predict_orig(X_test, beta)))

        if sprobes_model is not None:
            beta = sprobes_model.fit(X_tr_01, y_tr_01, epsilon)
            if beta is not None:
                store["S-PROBES"].append(rmse(y_test, aff.predict_orig(X_test, beta)))

        beta = adassp(
            Z, y_tr_01, epsilon, delta=_delta(len(y_train)), BZ=BZ, BY=1.0,
            ZTZ_pre=ztz, ZTy_pre=zty,
        )
        if beta is not None:
            store["AdaSSP"].append(rmse(y_test, aff.predict_orig(X_test, beta)))

        beta = diffprivlib_linear(
            X_tr_01, y_tr_01, epsilon, bounds_X=bounds_x, bounds_y=bounds_y,
        )
        if beta is not None:
            store["DiffPrivLib"].append(rmse(y_test, aff.predict_orig(X_test, beta)))

        beta = binagg_linear(X_tr_01, y_tr_01, epsilon, delta=_delta(len(y_train)))
        if beta is not None:
            store["BinAgg"].append(rmse(y_test, aff.predict_orig(X_test, beta)))

    rows = [{
        "dataset": dataset,
        "method": "OLS",
        "n_valid": 1,
        "rmse_mean": ols_rmse,
        "p": p,
        "epsilon": epsilon,
    }]
    for method, vals in store.items():
        if not vals:
            rows.append({
                "dataset": dataset,
                "method": method,
                "n_valid": 0,
                "rmse_mean": np.nan,
                "p": p,
                "epsilon": epsilon,
            })
        else:
            rows.append({
                "dataset": dataset,
                "method": method,
                "n_valid": len(vals),
                "rmse_mean": float(np.mean(vals)),
                "p": p,
                "epsilon": epsilon,
            })
    return pd.DataFrame(rows)


def realworld_run_config(
    dataset: str,
    p: int,
    eps: float,
    n_iters: int = 100,
    seed: int = 42,
    loader_kwargs: dict | None = None,
) -> dict[str, float]:
    loader_kwargs = loader_kwargs or {}
    X, y, names = load_dataset(dataset, **loader_kwargs)
    sprobes = build_sprobes_models()
    df = evaluate_config(X, y, names, p, eps, n_iters, seed, sprobes, dataset=dataset)
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        out[str(row["method"])] = float(row["rmse_mean"])
    return out
