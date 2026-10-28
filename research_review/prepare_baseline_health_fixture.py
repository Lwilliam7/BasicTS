"""Snapshot only the existing synthetic fixture/execute helpers; leave the deployed suite unchanged."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "scripts/research_suite_20261020/run_router_checks.py").read_text(encoding="utf-8")
tree = ast.parse(source)
header = '"""Exact helper snapshot from the October 20 suite; no original deployed file is edited."""\nfrom types import SimpleNamespace\n'
segments = [ast.get_source_segment(source, n) for n in tree.body
            if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("EXPERTS", "SEED") for t in n.targets))
            or isinstance(n, ast.FunctionDef) and n.name in ("synthetic_pool", "execute")]
dest = ROOT / "scripts/baseline_health_20261025/reference_fixture.py"
dest.write_text(header + "\n\n".join(segments) + "\n", encoding="utf-8", newline="\n")
print(f"Snapshotted {len(segments)} existing definitions into {dest}")
