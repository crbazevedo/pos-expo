# POS-Expo Experiments

This directory contains benchmark scripts to validate the library against legacy implementations.

## Scripts

- **`benchmark.py`**: Runs the full suite of benchmarks (1D, HD Gauss, Adult) using the library API. Used for general performance tracking.
- **`reproduce_legacy_benchmarks.py`**: Specifically compares the library's output against the "legacy" example scripts in `examples/` (`toy_selection_bias.py` and `adult_income_pos_expo.py`).
- **`plots_legacy_vs_lib.py`**: Generates comparison plots from the results of `reproduce_legacy_benchmarks.py`.

## Reproduction Workflow

To verify that the library faithfully reproduces the behavior of the legacy scripts:

1. Run the reproduction script:
   ```bash
   python experiments/reproduce_legacy_benchmarks.py
   ```
   This will execute the legacy scripts (as subprocesses) and the library implementation, checking for numerical agreement. Results are saved to `experiments/results/legacy_vs_lib.csv`.

2. Generate plots:
   ```bash
   python experiments/plots_legacy_vs_lib.py
   ```
   This generates figures in `docs/figures/`.

3. Update the main `README.md` with the new results if needed.

