# Developer Notes

## Legacy Reproduction (2025-11-27)

### Goal
Validate that the `pos-expo` library faithfully reproduces the results of the "legacy" experimental scripts used prior to refactoring.

### Legacy Scripts Identified
The following scripts in `examples/` were treated as the reference "legacy" implementations:
- `examples/toy_selection_bias.py`: Reference for 1D Piecewise selection bias.
- `examples/adult_income_pos_expo.py`: Reference for Adult dataset.

*Note: No dedicated legacy script was found for the HD Gaussian scenario in the repository history, so reproduction is focused on 1D Piecewise and Adult.*

### Reproduction Strategy
A new script `experiments/reproduce_legacy_benchmarks.py` was created to:
1. Execute the legacy scripts as subprocesses and parse their stdout to extract metrics (Test Log Loss).
2. Re-implement the same scenarios using the `pos_expo` public API within the script.
3. Compare the results numerically.

### Modifications Required
To enable deterministic reproduction in offline environments (where Adult dataset download fails), `src/pos_expo/datasets/adult.py` was patched to accept a `random_state` in `_make_dummy_adult`, ensuring consistent dummy data generation between the legacy subprocess and the library call.

### Results
- **1D Piecewise**: Absolute difference in Test Log Loss < 1e-4.
- **Adult**: Absolute difference in Test Log Loss < 1e-4.
- **Agreement**: Confirmed. The library implementation matches the legacy behavior.

### Artifacts
- `experiments/results/legacy_vs_lib.csv`: Consolidated results.
- `docs/figures/*_legacy_vs_lib.png`: Comparison plots.
- `README.md`: Updated with the reproduction results tables.

---

## Previous Review Notes (2025-11-27)
... (Previous notes retained/assumed)
