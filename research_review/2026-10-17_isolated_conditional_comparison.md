# Isolated conditional comparison — ETTh1 H=12 baseline run

Date: October 17, 2026. Status: **BASELINES_ONLY_PRIMARY_MISSING**.

User requested Priority 2 (FFORMA and other conditional combinations using the
same experts/evaluation), then specified isolated Python code under `scripts/`.
Implementation: `scripts/compare_conditional_combinations.py`; instructions:
`scripts/compare_conditional_combinations.md`. No BasicTS module imports or
changes to frozen method code. No server access, expert retraining, or test access.

## Data and protocol

The completed local run uses the existing ETTh1 walk-forward pool at H=12:
5,546 router_train windows and 2,773 router_val windows, all five experts in order
DLinear, PatchTST, iTransformer, TimesNet, ModernTCN. The original DLinear
final_60 checkpoint's scaler_std defines normalized masked MAE/MSE. No forecast
clamp. Input hashes and causal assertions are in `protocol.json`.

Each conditional family received the same 42 THA features of the cross-channel
mean history, five configurations and four purged expanding folds (20% warmup),
pooled OOF-MAE selection at seed 7, then final refits at seeds 7/11/13/17/19.
Training targets fully resolve before each held-out fold origin and before
router_val. Seeds average losses, not forecasts. Inputs remained unchanged.
The frozen primary's historical tuning budget is not retrospectively matched.

FFORMA is explicitly **FFORMA-MAE adapted to multivariate rolling windows**,
using the official softmax expected-loss objective and existing five-config grid.
It is not an M4 benchmark reproduction. Conditional error softmax predicts each
expert's normalized MAE with XGBoost and converts predicted errors to weights.
Conditional MLP directly minimizes normalized combination MAE. Both are generic
controls, not FAME/TimeRouter/OneNet replications. All use window-level weights;
this does not isolate the effect of CellCatch's cell granularity.

## Results

| Method | MAE | MSE |
| --- | ---: | ---: |
| Equal all-five | 0.371099 | 0.311582 |
| Train-selected single expert | 0.379558 | 0.326528 |
| FFORMA-MAE | 0.369073 | 0.310212 |
| Conditional error softmax | 0.368080 | 0.310985 |
| Conditional MLP | 0.386498 | 0.337496 |

Against equal-five, descriptive paired moving-block MAE intervals (block=24,
10,000 samples, seed 20260821) are:

- FFORMA: delta -0.002026; 95% interval [-0.003795, -0.000412].
- Conditional error softmax: delta -0.003019; interval [-0.007729, +0.001184].
- Conditional MLP: delta +0.015399; interval [+0.009589, +0.022548].

There is no multiplicity adjustment. These intervals compare with equal-five;
they are not a test against CellCatch or the strongest classical comparator.
No overall method-selection rule or frozen research label is changed.

## Software checks and limitations

All three methods passed a planted, predictable two-regime expert-winner test:
MAE approximately 0.005735 / 0.000000004 / 0.021163 versus equal MAE 1.0.
Additional checks passed for convexity, causal purge, the official objective,
masked normalization, underpowered-interval suppression, and primary-reference
rejection on pool/scaler/order/origin/seed-shape mismatch. The planted test is a
software positive control, not evidence that a published benchmark is reproduced.

Feature extraction emitted Intel MKL numerical warnings. Its diagnostics report
zero exceptions escaping a feature group, 22,307 training and 11,138 validation
nonfinite feature values replaced by zero; 35 of 42 training features vary.
Group functions can handle failures internally, so zero counted group exceptions
does not prove that every feature computed successfully. Interpret weak baselines
with these diagnostics. Negative MLP results are retained without retuning.

Artifacts: `experiments/conditional_comparison_etth1_h12_20261017/`, including
protocol, candidate selection audit, feature diagnostics, per-window losses,
results, report, and an exact `source_snapshot.py` of the executed script.
The script subsequently gained bounded history reduction (numerically checked
identical), a primary-normalizer identity check, and explicit bootstrap metadata;
those additions do not alter the completed baseline values.

## Outstanding work

The current wide-top3 per-window result tensors and the long-horizon forecast
caches are not on the laptop. The local topk `_work` contains JSON aggregates
but no corresponding `.pt` loss tensors. Do not manufacture paired intervals by
subtracting the archived aggregate primary score from these loss vectors.

Export a cache-bound primary NPZ from the source run, containing losses, exact
window starts, expert names, the five seeds, and train/validation/normalizer
SHA256 identities. The isolated script accepts it with `--primary` and refuses
mismatched identities. Repeat on the intended server pools/horizons using the
same frozen caches and predeclared seasonal frequencies. Until then, Priority 2
is implemented and partially evaluated, not completed. No locked-test pass is
authorized or performed.
