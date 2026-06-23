"""
S-PROBES: Steiner-block extension for large p (Section 6).

Single Laplace release over M blocks at scale M/epsilon, followed by
linear recovery and ridge-regularised OLS solve.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from probes.steiner_designs import (
    AG23_BLOCKS_P8,
    DIAG_BLOCKS_P12,
    DIAG_BLOCKS_P6,
    DIAG_BLOCKS_P8,
    FANO_BLOCKS_P6,
    STS13_BLOCKS_P12,
    build_ag24,
    build_mols_bibd_prime,
    build_pg24,
)


class SProbesModel:
    """
    General S-PROBES for Steiner blocks of size k_s and diagonal blocks
    capturing squared moments.
    """

    def __init__(
        self,
        p: int,
        steiner_blocks: Sequence[Sequence[int]],
        diag_blocks: Sequence[Sequence[int]],
    ):
        self.p = p
        self.steiner_blocks = [tuple(b) for b in steiner_blocks]
        self.diag_blocks = [tuple(b) for b in diag_blocks]
        self.n_steiner = len(steiner_blocks)
        self.n_diag = len(diag_blocks)
        self.M = self.n_steiner + self.n_diag
        self.D_DIM = p + 1

        self.k_s = len(steiner_blocks[0])
        self.k_d_list = [len(b) for b in diag_blocks]
        self.C_s = 2 ** self.k_s
        self.C_d_list = [2 ** k for k in self.k_d_list]

        self.N_S = self.n_steiner * self.C_s
        self.N_D = sum(self.C_d_list)
        self.N = self.N_S + self.N_D

        self._doff = np.zeros(self.n_diag + 1, dtype=np.intp)
        for k in range(self.n_diag):
            self._doff[k + 1] = self._doff[k] + self.C_d_list[k]

        self._build_lookups()
        self._build_gram_maps()
        self.T_A_flat = self.T_A.reshape(self.D_DIM * self.D_DIM, self.N)

    def _si(self, ell: int, c: int) -> int:
        return ell * self.C_s + c

    def _di(self, k: int, c: int) -> int:
        return self.N_S + int(self._doff[k]) + c

    def _build_lookups(self) -> None:
        p, k_s = self.p, self.k_s
        self.pair_to_block: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
        self.var_to_blocks: Dict[int, List[Tuple[int, int]]] = {
            v: [] for v in range(p + 1)
        }
        for ell, blk in enumerate(self.steiner_blocks):
            blk_list = list(blk)
            for bp in range(k_s):
                self.var_to_blocks[blk_list[bp]].append((ell, bp))
            for i in range(k_s):
                for j in range(i + 1, k_s):
                    pr = (min(blk_list[i], blk_list[j]), max(blk_list[i], blk_list[j]))
                    self.pair_to_block[pr] = (ell, i, j)

    def _build_gram_maps(self) -> None:
        p, dim, n_cells, k_s = self.p, self.D_DIM, self.N, self.k_s
        t_a = np.zeros((dim, dim, n_cells), dtype=np.float64)
        t_b = np.zeros((dim, n_cells), dtype=np.float64)

        for k in range(self.n_diag):
            w = 1.0 / self.n_diag
            for c in range(self.C_d_list[k]):
                t_a[0, 0, self._di(k, c)] += w

        for j in range(p):
            bl = self.var_to_blocks[j]
            w = 1.0 / len(bl)
            for ell, bp in bl:
                for c in range(self.C_s):
                    if (c >> (k_s - 1 - bp)) & 1:
                        t_a[0, j + 1, self._si(ell, c)] += w
                        t_a[j + 1, 0, self._si(ell, c)] += w

        for k, blk in enumerate(self.diag_blocks):
            k_d = self.k_d_list[k]
            for pos, j in enumerate(blk):
                if j >= p:
                    continue
                for c in range(self.C_d_list[k]):
                    if (c >> (k_d - 1 - pos)) & 1:
                        t_a[j + 1, j + 1, self._di(k, c)] = 1.0

        for (j, k), (ell, bj, bk) in self.pair_to_block.items():
            if j < p and k < p:
                for c in range(self.C_s):
                    if ((c >> (k_s - 1 - bj)) & 1) and ((c >> (k_s - 1 - bk)) & 1):
                        t_a[j + 1, k + 1, self._si(ell, c)] = 1.0
                        t_a[k + 1, j + 1, self._si(ell, c)] = 1.0

        bl_y = self.var_to_blocks[p]
        wy = 1.0 / len(bl_y)
        for ell, by in bl_y:
            for c in range(self.C_s):
                if (c >> (k_s - 1 - by)) & 1:
                    t_b[0, self._si(ell, c)] += wy

        for j in range(p):
            pr = (min(j, p), max(j, p))
            ell, bj, by = self.pair_to_block[pr]
            for c in range(self.C_s):
                if ((c >> (k_s - 1 - bj)) & 1) and ((c >> (k_s - 1 - by)) & 1):
                    t_b[j + 1, self._si(ell, c)] = 1.0

        self.T_A = t_a
        self.T_b = t_b

    def compute_cells(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact cell counts phi for data in [0,1]^(p+1)."""
        n = len(y)
        vals = np.concatenate([X, y[:, None]], axis=1)
        phi = np.zeros(self.N, dtype=np.float64)
        k_s, c_s = self.k_s, self.C_s
        cr = np.arange(c_s, dtype=np.int64)

        for ell, blk in enumerate(self.steiner_blocks):
            v = vals[:, list(blk)]
            w = np.ones((n, c_s), dtype=np.float64)
            for m in range(k_s):
                bm = ((cr >> (k_s - 1 - m)) & 1).astype(np.float64)
                vm = v[:, m, None]
                w *= bm * vm + (1.0 - bm) * (1.0 - vm)
            base = self._si(ell, 0)
            phi[base : base + c_s] = w.sum(axis=0)

        for k, blk in enumerate(self.diag_blocks):
            k_d, c_d = self.k_d_list[k], self.C_d_list[k]
            sq = X[:, list(blk)] ** 2
            cr_d = np.arange(c_d, dtype=np.int64)
            w = np.ones((n, c_d), dtype=np.float64)
            for m in range(k_d):
                bm = ((cr_d >> (k_d - 1 - m)) & 1).astype(np.float64)
                sm = sq[:, m, None]
                w *= bm * sm + (1.0 - bm) * (1.0 - sm)
            base = self._di(k, 0)
            phi[base : base + c_d] = w.sum(axis=0)

        return phi

    def noisy_cells(
        self,
        phi_exact: np.ndarray,
        epsilon: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        scale = self.M / epsilon
        if rng is None:
            return phi_exact + np.random.laplace(0.0, scale, self.N)
        return phi_exact + rng.laplace(0.0, scale, self.N)

    def gram_from_cells(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.einsum("ijk,k->ij", self.T_A, phi), self.T_b @ phi

    def rho_ridge(self, epsilon: float) -> float:
        scale = self.M / epsilon
        return max(sqrt(2.0 ** k_d) * scale for k_d in self.k_d_list)

    def ridge_solve(
        self,
        gram: np.ndarray,
        cross: np.ndarray,
        epsilon: float,
    ) -> Optional[np.ndarray]:
        rho = self.rho_ridge(epsilon)
        lam = min(float(np.linalg.eigvalsh(gram).min()), 0.0)
        reg = gram + (-lam + rho) * np.eye(self.D_DIM)
        try:
            beta = np.linalg.solve(reg, cross)
            return beta if np.isfinite(beta).all() else None
        except np.linalg.LinAlgError:
            return None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float,
        rng: np.random.Generator | None = None,
    ) -> Optional[np.ndarray]:
        """End-to-end S-PROBES on training data."""
        phi = self.compute_cells(X, y)
        phi_noisy = self.noisy_cells(phi, epsilon, rng=rng)
        gram, cross = self.gram_from_cells(phi_noisy)
        if not (np.isfinite(gram).all() and np.isfinite(cross).all()):
            return None
        return self.ridge_solve(gram, cross, epsilon)


def build_sprobes_models() -> Dict[int, SProbesModel]:
    """Pre-built models for p in {6, 8, 12, 15, 20, 24}."""
    diag_p15 = [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14)]
    diag_p20 = [
        (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
        (12, 13, 14, 15), (16, 17, 18, 19),
    ]
    diag_p24 = [
        (0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11),
        (12, 13, 14, 15), (16, 17, 18, 19), (20, 21, 22, 23),
    ]
    return {
        6: SProbesModel(6, FANO_BLOCKS_P6, DIAG_BLOCKS_P6),
        8: SProbesModel(8, AG23_BLOCKS_P8, DIAG_BLOCKS_P8),
        12: SProbesModel(12, STS13_BLOCKS_P12, DIAG_BLOCKS_P12),
        15: SProbesModel(15, build_ag24(), diag_p15),
        20: SProbesModel(20, build_pg24(), diag_p20),
        24: SProbesModel(24, build_mols_bibd_prime(5), diag_p24),
    }
