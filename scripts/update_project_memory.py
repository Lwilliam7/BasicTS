"""Find result artifacts that are not referenced by project_memory/EXPERIMENTS.md.

This utility does not edit memory files. It scans common result directories for
summary/final-report files and prints paths that are not yet mentioned in the
experiment ledger.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATTERNS = (
    "summary.json",
    "final_report.json",
    "finalist_summary.csv",
    "chrono_baseline_comparison.csv",
)


def iter_result_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for base in (root / "results", root / "experiments"):
        if not base.exists():
            continue
        for pattern in DEFAULT_PATTERNS:
            paths.extend(base.rglob(pattern))
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default="project_memory/EXPERIMENTS.md")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ledger_path = root / args.ledger
    if not ledger_path.exists():
        raise FileNotFoundError(f"Missing ledger: {ledger_path}")

    ledger_text = ledger_path.read_text(encoding="utf-8")
    missing: list[Path] = []
    for path in iter_result_files(root):
        rel = path.relative_to(root).as_posix()
        win_rel = str(path.relative_to(root))
        if rel not in ledger_text and win_rel not in ledger_text:
            missing.append(path.relative_to(root))

    if not missing:
        print("All discovered summary/final-report artifacts are referenced in the ledger.")
        return 0

    print("Result artifacts not referenced in project_memory/EXPERIMENTS.md:")
    for path in missing:
        print(f"- {path}")
    print()
    print("Review these manually. Do not auto-update conclusions without reading the files.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
