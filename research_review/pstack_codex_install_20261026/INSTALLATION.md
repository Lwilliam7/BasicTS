# PStack installed for Codex

Installed and enabled `pstack@cursor-pstack-local`, version `0.15.32`, from
`https://github.com/cursor/plugins`, subdirectory `pstack`, pinned to commit
`ab69380dd50ee47d426b35cb8369cffdddafc218`.

Installed files are at
`C:\Users\luwil\.codex\plugins\cache\cursor-pstack-local\pstack\0.15.32`.
The personal marketplace is at
`C:\Users\luwil\.codex\local-marketplaces\cursor-pstack`.

Verification completed on 2026-10-26:

- `codex plugin list --marketplace cursor-pstack-local --json` reports installed
  and enabled.
- All 200 installed upstream files match the reviewed source byte for byte.
- All 58 skill frontmatter blocks parse and include a name and description.
- A temporary Codex app-server's `skills/list` call discovers all 58 namespaced
  PStack skills with no PStack loading errors. No chat or agent task was started.
  The compact result is saved in `discovery.json`.

Codex skill names include `pstack:poteto-help`, `pstack:poteto-mode`,
`pstack:setup-pstack`, `pstack:interrogate`, and `pstack:tdd`.
Invoke skills with `$`, for example:

```text
$pstack:poteto-help
$pstack:poteto-mode Investigate this bug and verify the fix.
```

The installed skills should become available on the next turn. If the picker
still has its old list, restart Codex and reopen the same chat. `/pstack` alone
is not one of the upstream entry points.

This is the upstream package, preserved without a compatibility rewrite.
PStack is built for Cursor. Its setup skill writes Cursor model rules, and
Cursor model names, Task-tool options, custom agent types, Custom Modes,
Bugbot, `/loop`, and Automations are not provided by installing these files in
Codex. Discovery verification does not establish execution parity for those
workflows. Existing Codex permissions and tool availability still apply.

The earlier Linux process-debugger installation is separate from this plugin.
