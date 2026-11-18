from pathlib import Path
import shutil
local=Path(r'C:\Users\luwil\OneDrive\Documents\Code\BasicTS')
repo=Path(r'C:\Data\Documents\Code\BasicTS')
for name in ['compare_conditional_combinations.py','compare_conditional_combinations.md']:
    shutil.copy2(local/'scripts'/name,repo/'scripts'/name)
p=repo/'server.md'
body=p.read_text(encoding='utf-8')
heading='## Server state (as of 2026-09-19)'
section=(local/'research_review/conditional_server_md_section.md').read_text(encoding='utf-8')
if '## Conditional-combination comparison — prepared 2026-10-17' not in body:
    if heading not in body: raise SystemExit('Expected server.md heading missing')
    p.write_text(body.replace(heading,section+'\n\n'+heading,1),encoding='utf-8')
p=repo/'project_memory/TODO.md'
body=p.read_text(encoding='utf-8')
marker='## Addition — 2026-10-17: isolated conditional-comparison runner'
addition='''## Server handoff — 2026-10-17

The isolated runner now accepts `--server plan|run|status|pack` and reads the
existing server primary/comparator files directly, with numerical and metadata
alignment gates. Manual primary export is no longer needed for the H=96/long-
horizon layouts. Upload the single `scripts/compare_conditional_combinations.py`
file, then use the launch block in `server.md`. Prepared/tested locally; no server
upload or launch performed. Priority 2 remains open until the server run completes.

'''
if '## Server handoff — 2026-10-17' not in body:
    if marker not in body: raise SystemExit('Expected TODO marker missing')
    p.write_text(body.replace(marker,addition+marker,1),encoding='utf-8')
print('Saved the standalone server runner, guide, server.md launch block and TODO handoff')
