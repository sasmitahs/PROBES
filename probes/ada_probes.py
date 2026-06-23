"""
Ada-PROBES: PROBES release + noise-calibrated ridge solve (Algorithm 2).

theta_tilde = (A_hat - lambda I + rho I)^{-1} b_hat
  lambda = min(lambda_min(A_hat), 0)
  rho    = (2/epsilon) * 3^((p-1)/2)   for linear regression (d=1)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from probes.probes import ProbesCache, ProbesRelease


def probes_rho(p: int, epsilon: float) -> float:
    """Analytical ridge scale from paper Eq. (ridge_solve) for linear regression."""
    return (2.0 / epsilon) * (3.0 ** ((p - 1) / 2.0))


def ada_probes_solve(
    gram: np.ndarray,
    cross: np.ndarray,
    epsilon: float,
    p: int,
) -> Optional[np.ndarray]:
    """Ridge solve on recovered sufficient statistics (post-processing)."""
    if not (np.isfinite(gram).all() and np.isfinite(cross).all()):
        return None

    dim = gram.shape[0]
    rho = probes_rho(p, epsilon)
    lam = min(float(np.min(np.linalg.eigvalsh(gram))), 0.0)
    reg = gram - lam * np.eye(dim) + rho * np.eye(dim)
    try:
        beta = np.linalg.solve(reg, cross)
        return beta if np.isfinite(beta).all() else None
    except np.linalg.LinAlgError:
        return None


def ada_probes_from_release(
    release: ProbesRelease,
    cache: ProbesCache,
    epsilon: float,
) -> Optional[np.ndarray]:
    """Full Ada-PROBES from a PROBES release."""
    gram, cross = cache.assemble_gram(release)
    return ada_probes_solve(gram, cross, epsilon, cache.p)


def ada_probes_cached(cache: ProbesCache, epsilon: float) -> Optional[np.ndarray]:
    """Ada-PROBES using a pre-built ProbesCache (one MC draw)."""
    release = cache.release(epsilon)
    return ada_probes_from_release(release, cache, epsilon)


def ada_probes(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> Optional[np.ndarray]:
    """End-to-end Ada-PROBES on training data in [0,1]^(p+1)."""
    cache = ProbesCache(X, y)
    release = cache.release(epsilon, rng=rng)
    return ada_probes_from_release(release, cache, epsilon)
