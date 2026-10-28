"""Package new health sources and audited synthetic results without replacing old runs."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "scripts/baseline_health_20261025"
archive = ROOT / "research_review/baseline_health_20261025_source.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for p in sorted(source.iterdir()):
        if p.suffix in (".py", ".md"):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
results = ROOT / "research_review/baseline_health_20261025_results.tar.gz"
folders = (ROOT / "research_review/baseline_health_20261025",
           ROOT / "research_review/baseline_health_20261025_cpu",
           ROOT / "research_review/fame_health_diagnostic_20261025")
with tarfile.open(results, "w:gz") as tar:
    for folder in folders:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".md", ".npz", ".pt", ".log", ".py"):
                tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    for p in (ROOT / "research_review/reference_checks/baseline_health_20261025/source_download_manifest.json",
              ROOT / "research_review/fetch_health_reference_sources.py", ROOT / "research_review/diagnose_fame_health.py"):
        tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    for p in source.iterdir():
        if p.suffix in (".py", ".md"):
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
for p in (archive, results):
    with tarfile.open(p) as tar:
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() and not m.name.startswith("/") and ".." not in Path(m.name).parts for m in members)
    print(json.dumps(dict(archive=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
