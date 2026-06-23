"""Steiner system block definitions for S-PROBES."""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from typing import List, Sequence, Tuple

# p=6: Fano plane S(2,3,7); point 6 = response y
FANO_BLOCKS_P6: List[Tuple[int, ...]] = [
    (0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6),
    (0, 4, 5), (1, 5, 6), (0, 2, 6),
]
DIAG_BLOCKS_P6: List[Tuple[int, ...]] = [(0, 1), (2, 3), (4, 5)]

# p=8: resolvable design on 9 points; point 8 = y
AG23_BLOCKS_P8: List[Tuple[int, ...]] = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (1, 5, 6), (2, 3, 7),
    (0, 5, 7), (1, 3, 8), (2, 4, 6),
]
DIAG_BLOCKS_P8: List[Tuple[int, ...]] = [(0, 1), (2, 3), (4, 5), (6, 7)]


def _build_sts13() -> List[Tuple[int, ...]]:
    base_blocks = [(0, 1, 4), (0, 2, 7)]
    blocks = set()
    for base in base_blocks:
        for shift in range(13):
            blocks.add(tuple(sorted((x + shift) % 13 for x in base)))
    return sorted(blocks)


STS13_BLOCKS_P12: List[Tuple[int, ...]] = _build_sts13()
DIAG_BLOCKS_P12: List[Tuple[int, ...]] = [
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11),
]

# GF(4) arithmetic for AG(2,4) and PG(2,4)
_GF4_ADD = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
_GF4_MUL = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
_GF4_INV = [0, 1, 3, 2]


def _fa(x: int, y: int) -> int:
    return _GF4_ADD[x][y]


def _fm(x: int, y: int) -> int:
    return _GF4_MUL[x][y]


def _pg_canon(v: Sequence[int]) -> Tuple[int, ...] | None:
    for x in v:
        if x:
            s = _GF4_INV[x]
            return tuple(_fm(s, xi) for xi in v)
    return None


def _verify_bibd(lines: Sequence[Tuple[int, ...]], v: int, k: int, b: int) -> None:
    assert len(lines) == b and len(set(lines)) == b
    pair_cnt = {}
    for ln in lines:
        assert len(ln) == k
        for pr in combinations(ln, 2):
            pair_cnt[pr] = pair_cnt.get(pr, 0) + 1
    assert len(pair_cnt) == v * (v - 1) // 2
    assert all(c == 1 for c in pair_cnt.values())


def build_ag24() -> List[Tuple[int, ...]]:
    """S(2,4,16) for p=15."""

    def pt(a: int, b: int) -> int:
        return 4 * a + b

    lines: List[Tuple[int, ...]] = []
    for c in range(4):
        lines.append(tuple(sorted(pt(c, b) for b in range(4))))
    for m in range(4):
        for d in range(4):
            lines.append(tuple(sorted(pt(a, _fa(_fm(m, a), d)) for a in range(4))))
    _verify_bibd(lines, v=16, k=4, b=20)
    return lines


def build_pg24() -> List[Tuple[int, ...]]:
    """S(2,5,21) for p=20."""
    raw = set()
    for a in range(4):
        for b in range(4):
            for c in range(4):
                if (a, b, c) != (0, 0, 0):
                    raw.add(_pg_canon((a, b, c)))
    pts = sorted(raw)
    pidx = {p: i for i, p in enumerate(pts)}
    lset = set()
    for u in range(4):
        for v in range(4):
            for w in range(4):
                if (u, v, w) == (0, 0, 0):
                    continue
                ln = tuple(sorted(
                    pidx[pt]
                    for pt in pts
                    if _fa(_fa(_fm(u, pt[0]), _fm(v, pt[1])), _fm(w, pt[2])) == 0
                ))
                if len(ln) == 5:
                    lset.add(ln)
    lines = sorted(lset)
    _verify_bibd(lines, v=21, k=5, b=21)
    return lines


def build_mols_bibd_prime(q: int) -> List[Tuple[int, ...]]:
    """S(2,5,25) for p=24 when q=5."""
    blocks: List[Tuple[int, ...]] = []
    for i in range(q):
        blocks.append(tuple(range(i * q, (i + 1) * q)))
    for r in range(q):
        for c in range(q):
            blk = [(a - 1) * q + (a * r + c) % q for a in range(1, q)]
            blk.append((q - 1) * q + r)
            blocks.append(tuple(sorted(blk)))
    _verify_bibd(blocks, v=q * q, k=q, b=q * q + q)
    return blocks
