from pathlib import Path
import shutil
local=Path(r'C:\Users\luwil\OneDrive\Documents\Code\BasicTS')
repo=Path(r'C:\Data\Documents\Code\BasicTS')
source=local/'research_review/conditional_comparison_etth1_h12'
destination=repo/'experiments/conditional_comparison_etth1_h12_20261017'
if destination.exists():
    raise SystemExit(f'Refusing to overwrite {destination}')
shutil.copytree(source,destination)
record='2026-10-17_isolated_conditional_comparison.md'
shutil.copy2(local/'research_review'/record,repo/'project_memory/experiments'/record)
updates={
 'EXPERIMENTS.md': ('# Experiment Ledger', '\n\n## Isolated conditional comparison — 2026-10-17 (Partial Development Run)\n\nStandalone runner in `scripts/compare_conditional_combinations.py`. ETTh1 H=12,\nidentical frozen all-five pool, four purged folds, five configurations/family,\nfive final seeds: FFORMA-MAE MAE 0.369073; conditional error softmax 0.368080;\nconditional MLP 0.386498; equal-five 0.371099. Negative MLP result preserved.\n`BASELINES_ONLY_PRIMARY_MISSING`: wide-top3 per-window references and server-only\nlong-horizon caches still needed; this does not complete Priority 2. No test access\nor expert retraining. Details and limitations: `project_memory/experiments/'+record+'`.\n'),
 'TODO.md': ('# TODO', '\n\n## Addition — 2026-10-17: isolated conditional-comparison runner\n\nImplemented `scripts/compare_conditional_combinations.py` (see adjacent usage guide).\nCompleted an ETTh1 H=12 baseline-only development run, including FFORMA-MAE and\ntwo generic conditional controls; source snapshot and losses retained. Priority 2\nremains open: export matching frozen wide-top3 per-window results with cache and\nnormalizer identities, then run the paired comparisons and intended server pools.\nThis is not a FAME/TimeRouter replication or a locked-test pass. Record:\n`project_memory/experiments/'+record+'`.\n')}
for name,(heading,addition) in updates.items():
    p=repo/'project_memory'/name
    body=p.read_text(encoding='utf-8')
    if heading not in body:
        raise SystemExit(f'Missing expected heading in {p}')
    p.write_text(body.replace(heading,heading+addition,1),encoding='utf-8')
print('Saved baseline artifacts, detailed partial-run record, ledger entry, and remaining TODO')
