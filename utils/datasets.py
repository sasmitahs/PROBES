"""Real-world dataset loaders for PROBES benchmarks."""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np

SPROBES_P_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 15, 20, 24]
LOW_P_VALUES = [1, 2, 3, 4, 5, 6]

DATASETS_P_MAX: Dict[str, int] = {
    "California": 6,
    "NYC-Taxi": 6,
    "Appliances": 24,
    "Superconductivity": 24,
    "YearMSD": 24,
    "Covertype": 24,
}

_HIGH_P_DATASETS = frozenset({
    "Appliances",
    "Superconductivity",
    "YearMSD",
    "Covertype",
})

_rw_mod = None


def _load_realworld_eval():
    """Optional: import full loaders from author's probes_realworld_eval.py."""
    global _rw_mod
    if _rw_mod is not None:
        return _rw_mod

    candidates = []
    env = os.environ.get("PROBES_REALWORLD_EVAL")
    if env:
        candidates.append(os.path.expanduser(env))
    candidates.append(os.path.expanduser("~/Downloads/probes_realworld_eval.py"))

    eval_path = next((p for p in candidates if p and os.path.isfile(p)), None)
    if eval_path is None:
        raise RuntimeError(
            "Extended dataset loaders require probes_realworld_eval.py.\n"
            "Set PROBES_REALWORLD_EVAL=/path/to/probes_realworld_eval.py\n"
            "California Housing works without it (sklearn)."
        )

    bench_dir = os.path.dirname(os.path.abspath(eval_path))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import probes_realworld_eval as rw  # noqa: WPS433

    _rw_mod = rw
    return rw


def p_values_for_dataset(dataset_name: str, n_features: int) -> List[int]:
    cap = min(int(DATASETS_P_MAX.get(dataset_name, 6)), int(n_features))
    if cap < 1:
        return []
    if dataset_name in _HIGH_P_DATASETS:
        return [p for p in SPROBES_P_VALUES if p <= cap]
    return [p for p in LOW_P_VALUES if p <= cap]


def load_california(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from sklearn.datasets import fetch_california_housing

    csv_path = kwargs.get("csv_path") or os.environ.get("PROBES_CALIFORNIA_CSV")
    if csv_path and os.path.isfile(os.path.expanduser(csv_path)):
        import pandas as pd

        df = pd.read_csv(csv_path)
        y = df["MedHouseVal"].to_numpy(float)
        X = df.drop(columns=["MedHouseVal"]).to_numpy(float)
        names = [c for c in df.columns if c != "MedHouseVal"]
        return X, y, names

    data = fetch_california_housing()
    return data.data, data.target, list(data.feature_names)


def load_nyc_taxi(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rw = _load_realworld_eval()
    return rw.load_nyc_taxi(
        n_max=kwargs.get("n_max", 100_000),
        n_train=kwargs.get("n_train"),
        test_size=kwargs.get("test_size", 0.2),
        csv_path=kwargs.get("csv_path"),
        random_state=kwargs.get("random_state", 0),
    )


def load_appliances(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rw = _load_realworld_eval()
    return rw.load_appliances_energy(csv_path=kwargs.get("csv_path"))


def load_superconductivity(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rw = _load_realworld_eval()
    return rw.load_superconductivity(csv_path=kwargs.get("csv_path"))


def load_year_msd(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rw = _load_realworld_eval()
    return rw.load_year_prediction_msd(
        csv_path=kwargs.get("csv_path"),
        n_train=kwargs.get("n_train"),
        return_full=True,
        local_only=kwargs.get("local_only", True),
        random_state=kwargs.get("random_state", 42),
    )


def load_covertype(**kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rw = _load_realworld_eval()
    return rw.load_covertype_elevation(
        csv_path=kwargs.get("csv_path"),
        n_train=kwargs.get("n_train"),
        test_size=kwargs.get("test_size", 0.2),
        random_state=kwargs.get("random_state", 42),
    )


DATASET_LOADERS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray, List[str]]]] = {
    "California": load_california,
    "NYC-Taxi": load_nyc_taxi,
    "Appliances": load_appliances,
    "Superconductivity": load_superconductivity,
    "YearMSD": load_year_msd,
    "Covertype": load_covertype,
}

PAPER_REALWORLD_DATASETS = list(DATASET_LOADERS.keys())


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if name not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
