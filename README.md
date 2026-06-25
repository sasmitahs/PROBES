# PROBES

**PROBES: A Bernstein Simplex Approach to Differentially Private OLS Estimation**

Reference implementation of PROBES, Ada-PROBES, S-PROBES, and baseline methods for differentially private ordinary least squares.

## Repository layout

```
PROBES/
├── probes/                # DP algorithms (one module per method)
├── utils/                 # Metrics, normalization, data loaders
├── experiments/           # Optional benchmark runners
├── data/                  # Place downloaded datasets here
└── results/               # CSV outputs from experiment scripts
```

## Algorithms

| File | Method | Privacy |
|------|--------|---------|
| `probes/probes.py` | PROBES | pure ε-DP |
| `probes/ada_probes.py` | Ada-PROBES | pure ε-DP (post-processing) |
| `probes/sprobes.py` | S-PROBES | pure ε-DP |
| `probes/adassp.py` | AdaSSP | (ε,δ)-DP |
| `probes/binagg.py` | BinAgg | (ε,δ)-DP via μ-GDP |
| `probes/diffprivlib_lr.py` | DiffPrivLib | ε-DP (functional mechanism) |

## Setup

```bash
git clone https://github.com/sasmitahs/PROBES.git
cd PROBES
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick example

```python
import numpy as np
from probes.ada_probes import ada_probes
from probes.sprobes import build_sprobes_models
from utils.synthetic import generate_iid_linear

X, y, beta = generate_iid_linear(5000, p=4, seed=0)
b = ada_probes(X, y, epsilon=1.0)
print("Ada-PROBES ||beta - true||:", np.linalg.norm(b - beta))

model = build_sprobes_models()[6]
b6 = model.fit(X[:, :6], y, epsilon=1.0)
```

At large ε, private coefficients should match non-private OLS to numerical precision.

## Experiment scripts

Optional runners under `experiments/` write CSV summaries to `results/`:

```bash
python experiments/run_synthetic_iid.py --iters 100 --eps 1.0
python experiments/run_synthetic_correlated.py --iters 100 --eps 1.0 10.0
python experiments/run_realworld.py --dataset California --iters 100 --eps 1.0
python experiments/run_covertype_ntrain.py --iters 100
```

Real-world datasets can be placed under `data/` or loaded via environment variables (`PROBES_MSD_CSV`, `PROBES_TAXI_CSV`, etc.). See `utils/datasets.py` for supported datasets and paths.

## License

Research code. Dataset licenses apply to downloaded benchmarks.
