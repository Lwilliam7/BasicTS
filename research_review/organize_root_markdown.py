"""Move only repository-root Markdown; keep relative document links valid."""
from pathlib import Path
import hashlib
import os
import re
import subprocess
import sys

ROOT = Path(r"C:\Data\Documents\Code\BasicTS").resolve()
DEST = ROOT / "MDFILES"
sources = sorted(ROOT.glob("*.md"))
mapping = {p: DEST / p.name for p in sources}
assert len(sources) == 10, f"Root inventory changed: {len(sources)}"
assert all(not p.exists() for p in mapping.values()), "Destination collision"

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

frozen = ROOT / "experiments/gate_ablation"
protected = {p: digest(p) for p in frozen.iterdir() if p.is_file()}
changes = {}

def new_target(target, old, new):
    # Preserve URL titles and angle-bracket notation; only relative local paths move.
    target = target.strip()
    if target.startswith("<"):
        end = target.find(">")
        if end < 0:
            return target
        url, suffix, angled = target[1:end], target[end + 1:], True
    else:
        parts = re.split(r'(\s+[\"\'])', target, maxsplit=1)
        url, suffix, angled = parts[0], "".join(parts[1:]), False
    if not url or url.startswith(("#", "/", "\\")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        return target
    path, *anchor = re.split(r"(?=[#?])", url, maxsplit=1)
    resolved = (old.parent / path).resolve()
    final = mapping.get(resolved, resolved)
    if old.parent == new.parent and final == resolved:
        return target
    rel = Path(os.path.relpath(final, new.parent)).as_posix()
    if path.endswith("/"):
        rel += "/"
    value = rel + "".join(anchor)
    return ("<" + value + ">" if angled else value) + suffix

markdown = subprocess.check_output(["rg", "--files", "--hidden", "-g", "*.md", "-g", "!.git/**"], cwd=ROOT, text=True).splitlines()
for relative in markdown:
    old = ROOT / relative
    new = mapping.get(old, old)
    raw = old.read_bytes()
    text = raw.decode("utf-8-sig")
    # Inline links/images, reference links, and HTML links/images.
    text = re.sub(r'(!?\[[^\]\n]*\]\()([^\n)]+)(\))',
                  lambda m: m[1] + new_target(m[2], old, new) + m[3], text)
    text = re.sub(r'(?m)^(\s*\[[^\]\n]+\]:\s*)(\S+)(.*)$',
                  lambda m: m[1] + new_target(m[2], old, new) + m[3], text)
    text = re.sub(r'((?:href|src)=[\"\'])([^\"\']+)([\"\'])',
                  lambda m: m[1] + new_target(m[2], old, new) + m[3], text)
    encoded = (b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b"") + text.encode("utf-8")
    if encoded != raw:
        assert old not in protected, f"Would alter active experiment: {relative}"
        changes[old] = encoded

config = ROOT / "pyproject.toml"
raw = config.read_bytes()
assert b'readme = "README.md"' in raw
changes[config] = raw.replace(b'readme = "README.md"', b'readme = "MDFILES/README.md"')
pack = ROOT / "scripts/package_for_server.py"
raw = pack.read_bytes()
before = b'("requirements.txt", "pyproject.toml", "setup.py", "AGENTS.md", "SERVER_RUNBOOK.md")'
after = b'("requirements.txt", "pyproject.toml", "setup.py",\n                  "MDFILES/README.md", "MDFILES/README_BASICTS.md", "MDFILES/README_CN.md",\n                  "MDFILES/AGENTS.md", "MDFILES/SERVER_RUNBOOK.md")'
assert before in raw
changes[pack] = raw.replace(before, after)
# An unanchored server.local.md ignore pattern already covers the new location.
ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
assert "server.local.md" in ignore.splitlines()

print("Move:", ", ".join(p.name for p in sources))
print("Update references:", ", ".join(str(p.relative_to(ROOT)) for p in changes))
if "--apply" not in sys.argv:
    sys.exit(0)
DEST.mkdir(exist_ok=True)
for source, destination in mapping.items():
    assert source.parent.resolve() == ROOT and destination.parent.resolve() == DEST
    source.rename(destination)
for old, content in changes.items():
    mapping.get(old, old).write_bytes(content)
assert not list(ROOT.glob("*.md"))
assert len(list(DEST.glob("*.md"))) == 10
assert all(digest(p) == h for p, h in protected.items()), "Active experiment changed"
print("PASS: 10 root Markdown files relocated; active gate-ablation files unchanged")
