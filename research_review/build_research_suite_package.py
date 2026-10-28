"""Package only the new source directories, with a file manifest; never touch the server."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "scripts/research_suite_20261020"
FOLDERS = (SUITE, ROOT / "experiments/gate_granularity_exploratory_20261020")
files = sorted(p for folder in FOLDERS for p in folder.rglob("*") if p.is_file()
               and p.suffix in (".py", ".json", ".md", ".sh") and p.name != "source_manifest.json")
hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest = SUITE / "source_manifest.json"
manifest.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
files.append(manifest)
archive = ROOT / "research_review/research_suite_20261020.tar.gz"
with tarfile.open(archive, "w:gz") as tar:
    for p in files:
        tar.add(p, arcname=p.relative_to(ROOT).as_posix())
with tarfile.open(archive, "r:gz") as tar:
    names = tar.getnames()
    assert len(names) == len(set(names)) == len(files)
    for member in tar.getmembers():
        assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
        if member.name in hashes:
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == hashes[member.name]
print(json.dumps(dict(archive=str(archive), files=len(files), bytes=archive.stat().st_size,
                      sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))
