"""Record the completed local software audit in the actual project's research memory."""
import hashlib
import json
from pathlib import Path
import shutil

LOCAL = Path(__file__).resolve().parents[1]
MAIN = Path(r"C:\Data\Documents\Code\BasicTS").resolve()
AUDIT = LOCAL / "research_review/baseline_health_20261025_cpu"
data = json.loads((AUDIT / "results.json").read_text())
assert data["status"] == "SOFTWARE_HEALTH_PASS" and data["table_eligible"] is False
artifact = MAIN / "experiments/baseline_health_local_20261025"
sources = MAIN / "scripts/baseline_health_20261025"
memory = MAIN / "project_memory"
for target in (artifact, sources, memory / "experiments"):
    assert target.resolve().is_relative_to(MAIN) and target.resolve() != MAIN
    target.mkdir(parents=True, exist_ok=True)


def save_new(src, dest):
    if dest.exists():
        assert hashlib.sha256(src.read_bytes()).digest() == hashlib.sha256(dest.read_bytes()).digest(), f"Existing artifact differs: {dest}"
    else:
        shutil.copy2(src, dest)


for name in ("AUDIT_SUMMARY.md", "report.md", "results.json", "manifest.json", "paper_table_gate.json"):
    save_new(AUDIT / name, artifact / name)
for name in ("baseline_health_20261025_source.tar.gz", "baseline_health_20261025_results.tar.gz"):
    save_new(LOCAL / "research_review" / name, artifact / name)
for src in (LOCAL / "scripts/baseline_health_20261025").iterdir():
    if src.suffix in (".py", ".md"):
        save_new(src, sources / src.name)

record = """# Baseline-port synthetic health — 2026-10-25 (Local Software Audit)

User requested known-answer health tests of FAME, TimeRouter, OneNet and Super-Linear,
with paper-comparison rows withheld until their claimed method fidelity is established.

All four pass 14 software controls under explicit CPU execution: planted winners,
shuffled winners, joint expert-name/column permutations, plus OneNet's two known-branch
tests, causal update trace and future-label perturbation. Oracle MAE is 0.02.
FAME predictable MAE 0.021294; TimeRouter 0.020000; Super-Linear gate 0.136117 versus
equal-five 2.420001. OneNet OCP's branch-1/branch-2 controls give MAE 0.132602/0.080158
versus equal-two 1.020000/0.980000. These distinct controls are not a method ranking.

Every named-column permutation has exactly zero per-window loss difference. OneNet
makes 127 delayed updates; each target resolves before use. Poisoning targets from row
64 leaves predictions through row 64 inclusive exactly unchanged. Pinned upstream
hash/source checks pass for the components covered by the ports' existing checks.
Runtime tensor loads are limited to generated audit files; no research cache,
expert checkpoint or locked test split is accessed.

The first CPU-intended audit let FAME's auto selector use CUDA on Windows despite
an empty CUDA_VISIBLE_DEVICES mask. It failed exact permutation/reproducibility
(max per-window difference 0.180668). Canonical input fingerprints, forecasts,
targets and losses were identical. The failed audit and sources are preserved.
An explicit-CPU rerun passes unchanged thresholds and grids. This does not
validate or fix the exact reproducibility of historical CUDA runs.

**Paper fidelity remains NOT_ESTABLISHED; table_eligible is false for all four.**
FAME is a rolling-window adaptation; TimeRouter uses a custom retrained head;
OneNet is only its OCP block on two frozen branches with delayed feedback, not
full branch co-adaptation; Super-Linear covers its spectral gate, not the full
frequency-expert model. Do not label these software checks published-result
reproductions or rank online OneNet as an offline frozen router.

This ran locally, not on the server. Environment: Python 3.13.5, Torch 2.6.0+cu124,
NumPy 2.3.3, pandas 2.3.3. The server environment has not been revalidated here.
No research result, selection label, frozen method or paper row is upgraded.

Artifacts: `experiments/baseline_health_local_20261025/`, including detailed
AUDIT_SUMMARY.md, structured results, source/environment manifest, machine-readable
paper-table gate, source package and complete result archive (including the first
failed run and its source snapshot). Harness: `scripts/baseline_health_20261025/`.
"""
record_path = memory / "experiments/2026-10-25_baseline_port_health.md"
if record_path.exists():
    assert record_path.read_text(encoding="utf-8") == record
else:
    record_path.write_text(record, encoding="utf-8", newline="\n")

entries = {
    "EXPERIMENTS.md": """## Baseline Port Health — 2026-10-25 (Local Software Audit)

FAME, TimeRouter, OneNet OCP and Super-Linear gate pass 14 explicit-CPU known-answer
controls, including shuffled winners, expert-column permutations and OneNet's
delayed-feedback/future-label checks. FAME's initial CUDA reproducibility failure
is retained; explicit CPU passes without changing learning settings or thresholds.
This establishes local software-health evidence, not full-paper reproduction.
All paper-table rows remain NOT CLEARED. No real cache/test access or method/label change.
Record: `project_memory/experiments/2026-10-25_baseline_port_health.md`.

""",
    "TODO.md": """## Baseline Port Health — 2026-10-25: Local Controls Done, Paper Fidelity Open

- Done locally: all four requested ports (FAME, TimeRouter, OneNet OCP,
  Super-Linear gate), 14 known-answer/shuffled/permutation/causality checks on CPU.
- Still open: fidelity to each full paper method and protocol-compatible paper
  comparisons. Keep all four paper rows excluded; `table_eligible: false`.
- FAME: exact CUDA reproducibility remains unresolved. The first failure is
  preserved; the explicit-CPU audit passes unchanged controls.
- OneNet: this is only OCP over two frozen branches with delayed feedback.
  Do not call it full OneNet or rank it with offline frozen routers.
- This is local software evidence, not a new server-environment validation.
  Existing research results, method labels and other pending work are unchanged.
- Record: `project_memory/experiments/2026-10-25_baseline_port_health.md`.

""",
    "CURRENT_STATE.md": """## Addition — 2026-10-25: Four-Port Local Software Health Pass; Paper Rows Still Withheld

All four requested ports pass 14 explicit-CPU known-answer software controls:
FAME, TimeRouter, OneNet OCP and Super-Linear gate. Named-column permutations
are exactly invariant; OneNet passes causal feedback and future-label isolation.
FAME's earlier auto-CUDA reproducibility discrepancy is preserved and unresolved
for CUDA; explicit CPU passes without changing thresholds or learning settings.

This is local software-health evidence, not full-paper reproduction or a server
environment validation. `table_eligible` remains false for all four. OneNet is
its two-frozen-branch, delayed-feedback OCP adaptation; Super-Linear covers its
gate. No research result, selection label or frozen method is changed.
Record: `project_memory/experiments/2026-10-25_baseline_port_health.md`.

""",
    "DECISIONS.md": """## A Synthetic Software Pass Does Not Clear Full-Method Paper Rows (2026-10-25)

User instruction and the local four-port health audit: keep FAME, TimeRouter,
OneNet and Super-Linear paper-comparison rows withheld until claimed fidelity
and comparison protocol are supported. Their 14 explicit-CPU software checks
pass, but `paper_fidelity: NOT_ESTABLISHED` and `table_eligible: false` remain.
OneNet is only the causal OCP block on frozen branches, never full OneNet or an
offline protocol peer. FAME's CUDA exact reproducibility remains open; its CPU
control pass does not validate historical CUDA runs. No method or research label
is promoted. Record: `project_memory/experiments/2026-10-25_baseline_port_health.md`.

""",
}
for name, addition in entries.items():
    path = memory / name
    before = path.read_bytes()
    newline = b"\r\n" if b"\r\n" in before else b"\n"
    text = before.decode("utf-8").replace("\r\n", "\n")
    marker = addition.splitlines()[0]
    if marker in text:
        continue
    if name == "CURRENT_STATE.md":
        stamp = "Last updated: 2026-10-20"
        assert stamp in text[:150], "Unexpected current-state date; preserve and review before modifying"
        text = text.replace(stamp, "Last updated: 2026-10-25\n\n" + addition.rstrip(), 1)
    else:
        first, rest = text.split("\n", 1)
        assert first.startswith("# ")
        text = first + "\n\n" + addition + rest.lstrip("\n")
    data = text.replace("\n", newline.decode()).encode("utf-8")
    assert path.read_bytes() == before, f"Concurrent edit detected: {path}"
    path.write_bytes(data)
print("Saved the local audit and updated EXPERIMENTS, TODO, CURRENT_STATE and DECISIONS; paper rows remain withheld")
