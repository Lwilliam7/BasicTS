# Prompt Compliance Audit

This is a post-run audit over frozen outputs. It does not train, score, tune, or rerun any model.

## Supplemental Diagnostics

- `perturbation_rms_diagnostics.csv`: adds RMS normalized delta requested by Section 38.
- `expert_order_permutation_checks.csv`: verifies metric equivariance under a fixed expert-axis permutation.

## Result

- Tier remains `ACTIVE_SIGNAL_BUT_REDUNDANT`.
- `proceed_to_router_integration` remains `false`.

## Literal Gaps

- The saved raw-response cache does not include `SharedTotalLearnedProbe` raw response tensors.
- The per-window cache does not explicitly store absolute forecast origins; it stores common indices and predictions/targets.
- The shared delta is stored once per window, not duplicated per expert path. This matches the implemented mechanism but is not the literal Section 32 storage format.
