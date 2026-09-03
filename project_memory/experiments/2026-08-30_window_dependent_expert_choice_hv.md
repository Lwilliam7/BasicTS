# Window-Dependent Expert-Choice H×V Routing

Date: 2026-08-30. Directory: `experiments/window_dependent_expert_choice_hv/`.

**Provenance note:** this experiment's implementation was found on disk at the start of a later
Claude Code session already uncommitted (`git status` shows `??` for the whole directory), with
no corresponding `project_memory` entry and no git history. The version described here is the one
verified and reproduced from raw stored artifacts in that later session (win counts, static-parity
diff, integrity checks all recomputed directly from `validation_results.json`/`oof_results.json`/
`integrity_checks.json`/`tensors.pt`, not taken from any prose summary). Treat this record as the
authoritative backfill.

## Question

Given the same frozen K=3 experts and the same static Expert-Choice H×V score `S[h,v,e]` (from
`experiments/expert_choice_hv/`), does making the competence score window-dependent —
`S[t,h,v,e] = static_gain[h,v,e] + predicted_residual_gain[t,h,v,e]` — make expert-side H×V
allocation (Dynamic Expert Choice, CF=1) beat matched cell-side Top1 allocation (Dynamic Token
Choice) using the exact same score/affinity tensor for both?

## Method (frozen, verified from source)

- One SHARED scorer across the K=3 experts (not per-expert): `Linear(input->64)->ReLU->
  Linear(64->32)->ReLU->Linear(32->1)`, seed 7, AdamW lr=1e-3 wd=1e-4, max_epochs=100 patience=10,
  batch_size=32 (an implementation detail not separately specified anywhere, documented here).
- Features: global history (`window_features_group_a`, 6), per-variable history (7), target-free
  cell-local forecast/disagreement features (6), horizon/variable/expert embeddings (4/8/4), plus
  `static_gain[h,v,e]` as an explicit input scalar.
- Residual target: `gain[t,h,v,e] = equal_error[t,h,v] - expert_error[t,h,v,e]`; `static_gain` is
  the fit-only mean over legal fit windows per fold; the scorer predicts `gain - static_gain`.
- Strict causal 4-fold OOF (20% warmup + 4 folds) with full-horizon observability
  (`starts[i]+H <= current_eval_origin`), then a final router_train→router_val fit that purges any
  trailing router_train windows not fully observable before the first router_val origin.
- Affinity: fit-only scalar mean/std standardization of the raw score, then `softmax` over experts
  at temperature 1.0.
- CF=1, capacity `C=round(H·V/E)`. Multi-claim cells → simple average of claiming experts'
  forecasts; zero-claim cells → equal fixed ensemble fallback (verified identical to the static
  `expert_choice_hv` rule).
- Shuffled-current-window control: permutes only the dynamic input FEATURES across windows (seed
  20260830), keeping real forecasts/targets/static_gain attached to their true window.

## Results (router-val MAE, recomputed directly from `validation_results.json`)

| Dataset | Dynamic Token | Dynamic EC | Delta (EC−Token) | Static EC | Frozen HxV |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.377209 | 0.375640 | −0.001568 | 0.375352 | 0.366022 |
| ETTh2 | 0.280452 | 0.280951 | **+0.000499** | 0.277764 | 0.276898 |
| ETTm1 | 0.254659 | 0.253556 | −0.001103 | 0.253570 | 0.250690 |
| Weather | 0.155880 | 0.155621 | −0.000259 | 0.160330 | 0.159818 |
| Electricity | 0.207034 | 0.206356 | −0.000677 | 0.219015 | 0.215355 |

Router-val EC beats Token on **4/5** (ETTh2 the sole loser). Router-train OOF EC beats Token on
**3/5** (ETTh2 and Electricity lose OOF, despite Electricity winning on router-val — a real,
disclosed OOF/val sign flip). Block-24 CI excludes zero (favoring EC) on 4/5 (all but ETTh2).
Shuffled-window context weakened Dynamic EC on 5/5 datasets (correctly-matched context beats
shuffled). Claim maps genuinely window-dependent on 5/5 (mean adjacent claim-change fraction >5%
and mean adjacent Jaccard <0.95 on all datasets).

Final classification: **`WINDOW_DEPENDENT_EC_SUPPORTED`** — all 6 predeclared criteria pass, but
two (`oof_wins_ge_3`=3/5 exact minimum, `val_wins_vs_static_ec`... static-EC win count also at the
3/5 minimum) clear the bar only marginally. Dynamic EC beat Frozen HxV on **0/5** datasets.

## Integrity

Static parity vs `experiments/expert_choice_hv/results.json` exact (0.0 diff). Target-corruption,
targetless-prediction, future-suffix-corruption, frozen-checkpoint-hash, expert-order, and OOF
full-horizon-causality checks all passed on every dataset. `TEST SET ACCESSED: NO` throughout.

## Decision

Treat as real but fragile support, not decisive. Do not cite the win counts as more comfortable
than 4/5 (val) and 3/5 (OOF) — both numbers are exact and reproducible from the stored artifacts.
Motivated two direct follow-ups: `affinity_weighted_expert_choice_hv` (does restoring
affinity-weighted multi-claim combination help?) and `conflict_resolved_expert_choice_hv` (does
forcing exactly one expert per cell help?).
