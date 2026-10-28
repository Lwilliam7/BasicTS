"""Prepare isolated deployment sources from the existing checked-in implementations."""
from pathlib import Path
import hashlib
import json

LOCAL = Path(__file__).resolve().parents[1]
MAIN = Path(r"C:\Data\Documents\Code\BasicTS")
SUITE = LOCAL / "scripts/research_suite_20261020"
GRAN = LOCAL / "experiments/gate_granularity_exploratory_20261020"
GRAN.mkdir(parents=True, exist_ok=True)

(SUITE / "conditional.py").write_bytes((LOCAL / "scripts/compare_conditional_combinations.py").read_bytes())
original = MAIN / "experiments/gate_ablation"
for source, target in (("PREDECLARATION.md", "ORIGINAL_PREDECLARATION.md"),
                       ("AMENDMENT_2026-10-19.md", "AMENDMENT_2026-10-19.md")):
    (GRAN / target).write_bytes((original / source).read_bytes())
(GRAN / "PROTOCOL.md").write_bytes((SUITE / "PROTOCOL.md").read_bytes())
body = (original / "run_gate_ablation.py").read_text(encoding="utf-8")
start = body.index("def queue():")
end = body.index("def selftest():", start)
body = body[:start] + body[end:]
replacements = {
    '"""Gate ablation: see PREDECLARATION.md and AMENDMENT_2026-10-19.md.':
        '"""Exploratory post-result granularity follow-up: see PROTOCOL.md.',
    'EVIDENCE = "Development evidence on previously scored 60-80% blocks; no test access; frozen method unchanged."':
        'EVIDENCE = "EXPLORATORY POST-RESULT ANALYSIS. Development evidence on reused 60-80% blocks; no test access; frozen method unchanged."',
    '"experiments/gate_ablation/PREDECLARATION.md", "experiments/gate_ablation/AMENDMENT_2026-10-19.md",\n        "experiments/gate_ablation/run_gate_ablation.py", "experiments/gate_ablation/launch.sh"':
        '"experiments/gate_granularity_exploratory_20261020/ORIGINAL_PREDECLARATION.md",\n        "experiments/gate_granularity_exploratory_20261020/AMENDMENT_2026-10-19.md",\n        "experiments/gate_granularity_exploratory_20261020/PROTOCOL.md",\n        "experiments/gate_granularity_exploratory_20261020/run_granularity.py"',
    '"amendments": ["AMENDMENT_2026-10-19.md"],':
        '"amendments": ["AMENDMENT_2026-10-19.md"], "analysis_status": "EXPLORATORY_POST_RESULT",',
    '"# Gate ablation"': '"# Exploratory post-result granularity analysis"',
    'Only HELPS licenses a claim that input-dependent gating improves on the best static gate at this horizon.':
        'All new readings are exploratory, post-result observations. They do not upgrade the original experiment or license a confirmatory claim.',
    'mode.add_argument("--queue", action="store_true")\n': '',
    '        elif args.queue:\n            queue()\n': '',
}
for before, after in replacements.items():
    if before not in body:
        raise SystemExit(f"Expected source marker missing: {before[:95]!r}")
    body = body.replace(before, after)
# Preserve the indentation of the next mode after removing the --queue line.
body = body.replace('        mode.add_argument("--status"', '    mode.add_argument("--status"')
(GRAN / "run_granularity.py").write_text(body, encoding="utf-8", newline="\n")
record = {"source": str(original / "run_gate_ablation.py"),
          "original_sha256": hashlib.sha256((original / "run_gate_ablation.py").read_bytes()).hexdigest(),
          "derived_sha256": hashlib.sha256((GRAN / "run_granularity.py").read_bytes()).hexdigest(),
          "changes": ["isolated output directory", "exploratory evidence/report text", "source identity paths", "external serial scheduling"],
          "method_and_integrity_functions_changed": False}
(SUITE / "granularity_source_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Prepared conditional copy and isolated exploratory runner")
