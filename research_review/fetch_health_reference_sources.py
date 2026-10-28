"""Download public, commit-pinned upstream files for local synthetic port checks only."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(r"C:\Data\Documents\Code\BasicTS")
DEST = ROOT / "research_review/reference_checks/baseline_health_20261025/third_party"
SPECS = (
    ("FAME", "hit636/FAME", "fame_official_replication/run_fame_official_replication.py", "FAME_COMMIT", "FAME_SHA256_LF", ()),
    ("TimeRouter", "UConn-DSIS/TimeRouter", "timerouter_official_replication/run_timerouter_official_replication.py", "TR_COMMIT", "TR_SHA256_LF", ("gift_eval/predictor.py",)),
    ("OneNet", "yfzhang114/OneNet", "onenet_official_replication/run_onenet_official_replication.py", "ON_COMMIT", "ON_SHA256_LF", ()),
    ("SuperLinear", "azencot-group/SuperLinear", "super_linear_replication/run_super_linear_replication.py", "SL_COMMIT", "SL_SHA256_LF", ()),
)


def get_constants(path, names):
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                out[name] = ast.literal_eval(node.value)
    return out


tasks = []
for folder, repo, script, commit_key, hashes_key, extra in SPECS:
    c = get_constants(MAIN / "experiments" / script, {commit_key, hashes_key})
    for rel in (*c[hashes_key], *extra):
        tasks.append((folder, repo, c[commit_key], rel, c[hashes_key].get(rel)))


def fetch(task):
    folder, repo, commit, rel, expected = task
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{rel}"
    path = DEST / folder / rel
    if path.exists():
        data = path.read_bytes()
    else:
        data = urllib.request.urlopen(url, timeout=45).read()
    digest = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError(f"Pinned upstream hash mismatch: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return dict(path=str(path.relative_to(DEST)), url=url, sha256_lf=digest,
                matches_port_pin=expected is not None)


with ThreadPoolExecutor(max_workers=4) as pool:
    records = list(pool.map(fetch, tasks))
DEST.parent.mkdir(parents=True, exist_ok=True)
(DEST.parent / "source_download_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print(f"Verified/downloaded {len(records)} public source files to {DEST}")
