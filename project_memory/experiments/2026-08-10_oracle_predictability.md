# Oracle Predictability Diagnostic

## Hypothesis

If oracle weights/prototypes can be predicted from history and forecasts, router errors are mainly architectural. If not, the bottleneck is weak out-of-split predictability.

## Configuration

Input:

- 96-step history
- all five expert forecasts

Target:

- train-only fixed-3 oracle weights/prototypes for `PatchTST`, `iTransformer`, `TimesNet`

Validation oracle labels were computed only for measurement, not training or model selection.

## Dataset / Split

ETTh1 router train `20-60%`, router validation `60-80%`. Test not used.

## Baselines

Train-mean oracle weight constant baseline.

## Implementation

Relevant script:

- `experiments/oracle_weight_tournament/oracle_predictability_diagnostic.py`

## Commands

```powershell
python experiments\oracle_weight_tournament\oracle_predictability_diagnostic.py --device cuda --epochs 40 --seed 7
```

## Results

Train:

- R2 overall `0.2810`
- prototype accuracy `28.85%`
- top-1 oracle expert accuracy `60.98%`
- cosine similarity `0.8337`

Validation:

- R2 overall `-0.2898`
- prototype accuracy `8.84%`
- top-1 oracle expert accuracy `32.13%`
- cosine similarity `0.6673`

Train-mean baseline on validation:

- R2 overall `-0.0201`
- prototype accuracy `11.65%`
- top-1 oracle expert accuracy `36.89%`
- cosine similarity `0.7397`

## Interpretation

The model fits train oracle structure but generalizes worse than a simple train-mean baseline on validation.

## Decision

Do not rely on direct per-window oracle utility prediction as the main path. Use causal recent performance and structured adaptation instead.

## Relevant Files

- `experiments/oracle_weight_tournament/predictability_diagnostic/summary.json`
- `experiments/oracle_weight_tournament/predictability_diagnostic/baseline_metrics.json`
