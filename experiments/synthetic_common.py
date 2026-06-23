"""Shared i.i.d. synthetic benchmark logic (coefficient MSE protocol)."""

from __future__ import annotations

import numpy as np

from probes.ada_probes import ada_probes_cached
from probes.adassp import adassp, prepare_adassp_design
from probes.binagg import binagg_linear
from probes.diffprivlib_lr import diffprivlib_linear
from probes.probes import ProbesCache
from probes.sprobes import build_sprobes_models
from utils.metrics import coefficient_mse
from utils.synthetic import generate_iid_linear


def _delta(n: int) -> float:
    return min(1e-6, 1.0 / (n ** 2))


def synth_run_config(
    p: int, n: int, eps: float, n_iters: int = 100, base_seed: int = 42,
) -> dict[str, float]:
    sprobes = build_sprobes_models()
    X, y, true_beta = generate_iid_linear(n, p, seed=base_seed)
    X = np.clip(X, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    probes_cache = ProbesCache(X, y) if (3 ** p) * 2 <= 200_000 else None
    Z, BZ = prepare_adassp_design(X)

    store: dict[str, list[float]] = {
        m: [] for m in ["Ada-PROBES", "S-PROBES", "AdaSSP", "DiffPrivLib", "BinAgg"]
    }
    for it in range(n_iters):
        np.random.seed(base_seed + it + 1)

        if probes_cache is not None:
            store["Ada-PROBES"].append(
                coefficient_mse(ada_probes_cached(probes_cache, eps), true_beta)
            )
        if p in sprobes:
            store["S-PROBES"].append(
                coefficient_mse(sprobes[p].fit(X, y, eps), true_beta)
            )
        store["AdaSSP"].append(
            coefficient_mse(
                adassp(Z, y, eps, delta=_delta(n), BZ=BZ, BY=1.0), true_beta,
            )
        )
        store["DiffPrivLib"].append(
            coefficient_mse(
                diffprivlib_linear(X, y, eps, bounds_X=(0, 1), bounds_y=(0, 1)),
                true_beta,
            )
        )
        store["BinAgg"].append(
            coefficient_mse(binagg_linear(X, y, eps, delta=_delta(n)), true_beta)
        )
    return {m: float(np.nanmean(v)) for m, v in store.items()}
