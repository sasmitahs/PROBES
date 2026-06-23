"""Paper table reference values (VLDB 2026 PROBES). Metric per table noted below."""

from __future__ import annotations

# tab:realworld_combined — test RMSE, unit-cube, 100 MC seeds
# Keys: (dataset, p, epsilon) -> {method: value}
REALWORLD: dict[tuple[str, int, float], dict[str, float]] = {
    # California Housing (subset for automated validation)
    ("California", 1, 0.5): {"OLS": 0.8421, "Ada-PROBES": 0.8420, "AdaSSP": 0.8788, "DiffPrivLib": 0.8429, "BinAgg": 0.8499},
    ("California", 1, 1.0): {"OLS": 0.8420, "Ada-PROBES": 0.8420, "AdaSSP": 0.8477, "DiffPrivLib": 0.8423, "BinAgg": 0.8443},
    ("California", 4, 0.5): {"OLS": 0.8059, "Ada-PROBES": 0.8112, "AdaSSP": 0.9819, "DiffPrivLib": 4724.9, "BinAgg": 0.8517},
    ("California", 4, 1.0): {"OLS": 0.8087, "Ada-PROBES": 0.8087, "AdaSSP": 0.9153, "DiffPrivLib": 1065.1, "BinAgg": 0.8252},
    ("California", 6, 0.5): {"OLS": 0.7463, "Ada-PROBES": 0.8070, "S-PROBES": 0.8193, "AdaSSP": 1.0320, "BinAgg": 0.8865},
    ("California", 6, 1.0): {"OLS": 0.7463, "Ada-PROBES": 0.7757, "S-PROBES": 0.7970, "AdaSSP": 0.9683, "BinAgg": 0.8320},
    # Covertype p=12 eps=1 (spot check from paper)
    ("Covertype", 12, 1.0): {"OLS": 171.58, "S-PROBES": 176.79, "AdaSSP": 183.36, "BinAgg": 184.10},
}

# tab:synthetic_iid_combined — test MSE at eps=1.0 (paper); repo benchmark uses coef MSE
# Keys: (p, n_mult) where n = mult * p
IID_SYN_EPS = 1.0
IID_SYN: dict[tuple[int, int], dict[str, float]] = {
    (1, 2000): {"Ada-PROBES": 2.8e-5, "AdaSSP": 1.4e-2, "DiffPrivLib": 2.9e-4, "BinAgg": 6.4e-3},
    (4, 2000): {"Ada-PROBES": 3.6e-5, "AdaSSP": 1.8e-3, "DiffPrivLib": 4.7e-4, "BinAgg": 1.4e-3},
    (6, 2000): {"Ada-PROBES": 1.4e-4, "S-PROBES": 3.3e-4, "AdaSSP": 8.6e-4, "DiffPrivLib": 7.2e-4, "BinAgg": 1.7e-3},
    (6, 10000): {"Ada-PROBES": 6.0e-6, "S-PROBES": 1.2e-5, "AdaSSP": 1.9e-4, "DiffPrivLib": 2.5e-5, "BinAgg": 2.3e-4},
    (12, 10000): {"S-PROBES": 4.1e-5, "AdaSSP": 1.1e-4, "DiffPrivLib": 7.0e-5, "BinAgg": 3.5e-3},
}

# tab:synthetic_corr — test RMSE, n=10000p, 100 seeds; paper reports eps in {1.0, 10.0}
CORR_SYN: dict[tuple[int, float], dict[str, float]] = {
    (4, 1.0): {"OLS": 0.0302, "Ada-PROBES": 0.0303, "AdaSSP": 0.0330, "BinAgg": 0.0328, "DiffPrivLib": 0.0304},
    (4, 10.0): {"OLS": 0.0302, "Ada-PROBES": 0.0302, "AdaSSP": 0.0303, "BinAgg": 0.0303, "DiffPrivLib": 0.0302},
    (6, 1.0): {"OLS": 0.0300, "Ada-PROBES": 0.0301, "S-PROBES": 0.0305, "AdaSSP": 0.0322, "BinAgg": 0.0329},
    (6, 10.0): {"OLS": 0.0300, "Ada-PROBES": 0.0300, "S-PROBES": 0.0300, "AdaSSP": 0.0300, "BinAgg": 0.0301},
    (12, 1.0): {"OLS": 0.0303, "S-PROBES": 0.0314, "AdaSSP": 0.0315, "BinAgg": 0.0377},
    (12, 10.0): {"OLS": 0.0303, "S-PROBES": 0.0304, "AdaSSP": 0.0304, "BinAgg": 0.0304},
    (24, 1.0): {"OLS": 0.0300, "S-PROBES": 0.0332, "AdaSSP": 0.0312, "BinAgg": 0.0479},
    (24, 10.0): {"OLS": 0.0300, "S-PROBES": 0.0302, "AdaSSP": 0.0302, "BinAgg": 0.0327},
}

# tab:covertype_ntrain — test RMSE, eps=10
COVERTYPE_NTRAIN: dict[tuple[int, int], dict[str, float]] = {
    (500, 12): {"AdaSSP": 294.19, "S-PROBES": 370.20, "BinAgg": 725.40},
    (5000, 12): {"S-PROBES": 180.33, "AdaSSP": 188.75, "BinAgg": 169.83},
    (100000, 12): {"OLS": 171.58, "S-PROBES": 171.63, "AdaSSP": 171.97, "BinAgg": 172.20},
    (100000, 24): {"OLS": 141.18, "S-PROBES": 141.96, "AdaSSP": 147.92, "BinAgg": 143.15},
}
