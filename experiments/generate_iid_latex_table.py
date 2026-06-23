#!/usr/bin/env python3
"""
Generate LaTeX for tab:synthetic_iid_combined from a summary CSV.

Input CSV: wide or long format with columns p, mult (or n), method, mse.
Output: paper-style table with bold/underline for best/second-best private methods.

Example:
  python experiments/run_synthetic_iid.py --metric test --iters 100 --eps 1.0
  python experiments/generate_iid_latex_table.py \\
      --summary results/synthetic_iid_summary.csv \\
      --out results/synthetic_iid_table.tex
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.synthetic_common import CELL_LIMIT, P_STEINER

P_VALUES = [1, 2, 3, 4, 5, 6, 8, 12, 15, 20, 24]
SAMPLE_MULTIPLIERS = [2000, 5000, 10000]
COL_METHODS = ["Ada-PROBES", "S-PROBES", "AdaSSP", "DiffPrivLib", "BinAgg"]
LATEX_HEADERS = ["Ada-P", "S-P", "AdaSSP", "DPL", "BinAgg"]


def _load_summary(path: str, epsilon: float) -> dict[tuple[int, int], dict[str, float]]:
    df = pd.read_csv(path)
    if path.endswith("_summary.csv"):
        return _load_summary_wide(df, epsilon)
    if "epsilon" in df.columns:
        df = df[df["epsilon"] == epsilon]
    if "method" in df.columns and "mse" in df.columns:
        if "mult" not in df.columns and "n" in df.columns and "p" in df.columns:
            df = df.copy()
            df["mult"] = (df["n"] / df["p"]).astype(int)
        grouped = (
            df.groupby(["p", "mult", "method"])["mse"]
            .apply(_robust_mean)
            .reset_index()
        )
        out: dict[tuple[int, int], dict[str, float]] = {}
        for _, row in grouped.iterrows():
            key = (int(row["p"]), int(row["mult"]))
            out.setdefault(key, {})[row["method"]] = float(row["mse"])
        return out
    return _load_summary_wide(df, epsilon)


def _load_summary_wide(df: pd.DataFrame, epsilon: float) -> dict[tuple[int, int], dict[str, float]]:
    if "epsilon" in df.columns:
        df = df[df["epsilon"] == epsilon]
    out: dict[tuple[int, int], dict[str, float]] = {}
    for _, row in df.iterrows():
        p = int(row["p"])
        mult = int(row.get("mult", row["n"] / p))
        vals = {}
        for m in COL_METHODS:
            if m in row and pd.notna(row[m]):
                vals[m] = float(row[m])
        out[(p, mult)] = vals
    return out


def _ada_probes_available(p: int) -> bool:
    return (3 ** p) * 2 <= CELL_LIMIT


def _format_sci(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "{---}"
    if val == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(val))))
    mant = val / (10 ** exp)
    # Prefer 2 significant digits in mantissa (paper style: 9.11, 1.4, 6.0)
    if mant >= 10:
        mant_s = f"{mant:.1f}".rstrip("0").rstrip(".")
    elif mant >= 1:
        mant_s = f"{mant:.2f}".rstrip("0").rstrip(".")
    else:
        mant_s = f"{mant * 10:.2f}".rstrip("0").rstrip(".")
        exp -= 1
    if exp == 0:
        return f"${mant_s}$"
    return f"${mant_s}{{\\times}}10^{{{exp}}}$"


def _robust_mean(series: pd.Series, cap: float = 1.0) -> float:
    vals = series[np.isfinite(series) & (series <= cap)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def _decorate_group(values: dict[str, float | None]) -> dict[str, str]:
    """Bold best private, underline second-best; skip non-finite and unavailable."""
    finite = {
        m: v for m, v in values.items()
        if v is not None and np.isfinite(v)
    }
    ranked = sorted(finite.items(), key=lambda kv: kv[1])
    best = ranked[0][0] if ranked else None
    second = ranked[1][0] if len(ranked) > 1 else None

    out: dict[str, str] = {}
    for m in COL_METHODS:
        v = values.get(m)
        if v is None or not np.isfinite(v):
            out[m] = "{---}"
            continue
        inner = _format_sci(v)[1:-1]
        if m == best:
            out[m] = f"$\\mathbf{{{inner}}}$"
        elif m == second:
            out[m] = f"$\\underline{{{inner}}}$"
        else:
            out[m] = _format_sci(v)
    return out


def build_latex_table(
    summary: dict[tuple[int, int], dict[str, float]],
    epsilon: float = 1.0,
) -> str:
    lines: list[str] = [
        r"\renewcommand{\arraystretch}{0.80}",
        r"\begin{table*}[!htbp]",
        r"\centering",
        rf"\caption{{Test MSE on i.i.d.\ synthetic data at $\varepsilon = {epsilon:g}$;"
        r" \textbf{Bold} = best private; \underline{underline} = second-best."
        r" Columns group results by training-set size"
        r" $n \in \{2{,}000p,\,5{,}000p,\,10{,}000p\}$.}",
        r"\label{tab:synthetic_iid_combined}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\tableresize{%",
        r"\begin{tabular}{@{}c *{5}{r} *{5}{r} *{5}{r} @{}}",
        r"\toprule",
        r"& \multicolumn{5}{c}{$2{,}000p$} & \multicolumn{5}{c}{$5{,}000p$} & \multicolumn{5}{c}{$10{,}000p$} \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}",
        r"$p$ & "
        + " & ".join(LATEX_HEADERS)
        + " & "
        + " & ".join(LATEX_HEADERS)
        + " & "
        + " & ".join(LATEX_HEADERS)
        + r" \\",
        r"\midrule",
    ]

    for p in P_VALUES:
        row_cells: list[str] = [str(p)]
        for mult in SAMPLE_MULTIPLIERS:
            key = (p, mult)
            raw = summary.get(key, {})
            group_vals: dict[str, float | None] = {}
            for m in COL_METHODS:
                if m == "S-PROBES" and p not in P_STEINER:
                    group_vals[m] = None
                elif m == "Ada-PROBES" and not _ada_probes_available(p):
                    group_vals[m] = None
                else:
                    group_vals[m] = raw.get(m, float("nan"))
            decorated = _decorate_group(group_vals)
            for m in COL_METHODS:
                row_cells.append(decorated[m])
        lines.append(" & ".join(row_cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
        r"\renewcommand{\arraystretch}{1.0}",
        "",
    ])
    return "\n".join(lines)


def summary_to_paper_dict(
    summary: dict[tuple[int, int], dict[str, float]],
) -> dict[tuple[int, int], dict[str, float]]:
    """Round values for paper_tables.py embedding."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, methods in summary.items():
        row: dict[str, float] = {}
        p, mult = key
        for m, v in methods.items():
            if m == "S-PROBES" and p not in P_STEINER:
                continue
            if m == "Ada-PROBES" and not _ada_probes_available(p):
                continue
            if np.isfinite(v):
                row[m] = float(v)
        if row:
            out[key] = row
    return out


def emit_paper_tables_py(summary: dict[tuple[int, int], dict[str, float]]) -> str:
    """Python source for IID_SYN in paper_tables.py."""
    lines = [
        "# tab:synthetic_iid_combined — test MSE at eps=1.0, 80/20 split, 100 seeds",
        "# Keys: (p, n_mult) where n = mult * p",
        "IID_SYN_EPS = 1.0",
        "IID_SYN: dict[tuple[int, int], dict[str, float]] = {",
    ]
    for (p, mult) in sorted(summary.keys()):
        methods = summary[(p, mult)]
        parts = []
        for m in COL_METHODS:
            if m not in methods or not np.isfinite(methods[m]):
                continue
            if m == "S-PROBES" and p not in P_STEINER:
                continue
            if m == "Ada-PROBES" and not _ada_probes_available(p):
                continue
            v = methods[m]
            parts.append(f'"{m}": {v:.4g}')
        if parts:
            lines.append(f"    ({p}, {mult}): {{{', '.join(parts)}}},")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate i.i.d. synthetic LaTeX table")
    parser.add_argument(
        "--summary",
        default=os.path.join(ROOT, "results", "synthetic_iid.csv"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "experiments", "synthetic_iid_table.tex"),
    )
    parser.add_argument(
        "--paper-tables-out",
        default=os.path.join(ROOT, "experiments", "iid_syn_paper_values.txt"),
    )
    parser.add_argument("--eps", type=float, default=1.0)
    args = parser.parse_args()

    summary = _load_summary(args.summary, args.eps)
    latex = build_latex_table(summary, epsilon=args.eps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(latex)
    with open(args.paper_tables_out, "w", encoding="utf-8") as f:
        f.write(emit_paper_tables_py(summary))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.paper_tables_out}")


if __name__ == "__main__":
    main()
