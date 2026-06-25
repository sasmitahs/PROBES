# Datasets

This folder is intentionally empty in the repository.

Benchmark datasets are **not committed** because they are large and have separate licenses. Download them from the sources below and place files here, or set the corresponding environment variable.

| Dataset | Suggested filename | Environment variable | Source |
|---------|-------------------|----------------------|--------|
| Year Prediction MSD | `YearPredictionMSD.txt.bz2` | `PROBES_MSD_CSV` | [UCI YearPredictionMSD](https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD) |
| NYC Yellow Taxi | `yellow_tripdata_2022-01.parquet` or `.csv` | `PROBES_TAXI_CSV` | [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Appliances Energy | `energydata_complete.csv` | `PROBES_APPLIANCES_CSV` | [UCI Appliances energy](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) |
| Superconductivity | `train.csv` | `PROBES_SUPERCONDUCTIVITY_CSV` | [UCI Superconductivity](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) |
| Covertype | `covtype.data.gz` | `PROBES_COVTYPE_CSV` | [UCI Covertype](https://archive.ics.uci.edu/ml/datasets/Covertype) |

**No download needed:** synthetic benchmarks generate data in code; California Housing loads via scikit-learn.

See `utils/datasets.py` for loader details.
