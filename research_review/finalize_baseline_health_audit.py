"""Validate the finished artifact set and record the software/fidelity distinction."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_review/baseline_health_20261025_cpu"
result = json.loads((OUT / "results.json").read_text())
assert result["status"] == "SOFTWARE_HEALTH_PASS"
assert len(result["methods"]) == 4
case_count = 0
for method, rec in result["methods"].items():
    assert rec["software_health"] == "PASS" and rec["table_eligible"] is False
    assert rec["cross_case_checks"]["expert_name_column_permutation"] is True
    for case, values in rec["cases"].items():
        assert values["software_case_passed"] is True
        for rel, expected in values["artifact_hashes"].items():
            assert hashlib.sha256((OUT / method / case / rel).read_bytes()).hexdigest() == expected
        case_count += 1
assert case_count == 14
assert result["methods"]["onenet"]["cross_case_checks"]["future_target_poison_prefix_unchanged"] is True
assert set(result["methods"]["fame"]["cases"]["predictable"]["diagnostics"]["training_devices"]) == {"cpu"}

policy = dict(date="2026-10-25", software_health="PASS_EXPLICIT_CPU", cases_passed=14,
              paper_fidelity="NOT_ESTABLISHED", table_eligible=False, methods={})
for method, m in result["methods"].items():
    policy["methods"][method] = dict(software_health=m["software_health"], paper_fidelity="NOT_ESTABLISHED",
            table_eligible=False, limitation=m["limitation"],
            extra_open_issue="CUDA exact reproducibility remains unresolved" if method == "fame" else None)
(OUT / "paper_table_gate.json").write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

summary = """# Baseline health audit — 2026-10-25

**All four ports passed 14 known-answer software controls with CPU explicitly selected.**
**Their paper-comparison rows remain NOT CLEARED.** This is a local synthetic audit,
not a published-result reproduction or a server-environment validation.

| Port/control | MAE | Equal control | Oracle MAE | Finding |
| --- | ---: | ---: | ---: | --- |
| FAME, predictable | 0.021294 | 2.420001 | 0.020000 | Learns the planted winner |
| TimeRouter, predictable | 0.020000 | 2.420001 | 0.020000 | Learns the planted winner; no fallback on this case |
| Super-Linear gate, predictable | 0.136117 | 2.420001 | 0.020000 | Learns the planted winner |
| OneNet OCP, branch 1 wins | 0.132602 | 1.020000 | 0.020000 | Learns to favour branch 1 |
| OneNet OCP, branch 2 wins | 0.080158 | 0.980000 | 0.020000 | Learns to favour branch 2 |

These are different software controls, not a ranking of the methods. OneNet's
control uses its actual two branches; the other three use the five-expert fixture.

- All four pass the shuffled-winner guards and the named-column permutation check.
  Permutation differences are exactly zero in the explicit-CPU audit.
- OneNet makes 127 delayed updates per case. Every update was checked against its
  target-resolution time and actual input tensors. Poisoning targets from row 64
  onward leaves predictions through row 64 inclusive exactly unchanged.
- The deliberately poisoned case's full-period MAE is not a performance result;
  its purpose is the exact unchanged-prefix check.
- Pinned official source/hash checks pass for all four, for the components covered
  by their existing port checks. This does not establish parity of an entire paper method.
- Runtime loads were restricted to newly generated audit files. No research cache,
  expert checkpoint or locked test block was loaded.

## FAME execution discrepancy retained

The first audit intended to hide CUDA using an empty CUDA_VISIBLE_DEVICES mask.
That did not disable CUDA on this Windows host. FAME's upstream auto-device
selector chose CUDA; independent seed-7 fits gave differing losses despite
identical canonical fingerprints, loss matrices, forecast tensors and targets.
The first permutation check failed (maximum per-window difference 0.180668).
Those results and their executed source snapshot are retained in
`../baseline_health_20261025/`; diagnostic fits are retained separately.

The second audit explicitly sets FAME's configuration device to CPU and asserts
the actual device. It passes the original, unchanged permutation threshold
exactly. No grids, learning settings or thresholds were tuned. This clears only
the explicit-CPU software control; it does not fix or validate the historical
CUDA runs, whose exact reproducibility remains open.

## Paper-fidelity gate

- FAME: frozen-pool rolling-window adaptation; a published benchmark result was not reproduced.
- TimeRouter: existing custom retrained OVA head; original head training code is not available here.
- OneNet: only the OCP block over two frozen branches, with delayed feedback.
  Full branch co-adaptation is absent. It must not be labelled a full OneNet
  reproduction or ranked as an offline frozen router.
- Super-Linear: the spectral gate over frozen experts; full frequency-expert training is absent.

`paper_table_gate.json` explicitly keeps `table_eligible: false` for every port.
The software-health prerequisite is now supported locally. Faithful scope,
published-result reproduction and protocol-compatible comparisons are still
unresolved before any full-method paper row can be cleared.

## Artifacts

`results.json`, `report.md`, `manifest.json` and per-case folders retain the exact
environment, sources, upstream pins, losses, diagnostics, and OneNet causal trace.
The run used local Python 3.13.5, Torch 2.6.0+cu124, NumPy 2.3.3 and pandas 2.3.3;
it is not evidence of a run under the server's different package versions.
"""
(OUT / "AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")
snapshot = OUT / "source_snapshot"
snapshot.mkdir(exist_ok=True)
for path in (ROOT / "scripts/baseline_health_20261025").iterdir():
    if path.suffix in (".py", ".md"):
        shutil.copy2(path, snapshot / path.name)
print("Verified 14 controls and artifact hashes; paper-table gate remains closed")
