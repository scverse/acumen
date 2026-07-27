# `acumen ship` — put a proven skill inside the package

```bash
acumen ship --skill v2 [--force] [--model M] [--max-turns N] [--max-usd X]
            [--stream] [--log-dir logs] [--auth auto|session|api]
```

`--skill VERSION` is **required** — there is no implicit "latest" or "best". Run it only
after the report shows that version earning its place *and* the user has agreed to ship it:
it writes to the target's real checkout and can open a PR.

## What it produces

An agent modifies the target checkout so the package gains a `<dist-name>-install-skills`
console script. The package's users then run, e.g.:

```bash
scanpy-install-skills --agent claude        # or codex | agents | claude-science
scanpy-install-skills --dest ./somewhere    # explicit destination
scanpy-install-skills --print-path          # where the bundled skill lives
```

There is **no default framework** — `--agent` or `--dest` is required. The same
`SKILL.md` + `references/` bundle installs verbatim into every framework; there is no
per-framework conversion. Destinations are `<root>/skills/<skill_name>`, with roots
`~/.claude` (`CLAUDE_CONFIG_DIR`), `~/.codex` (`CODEX_HOME`), `~/.agents`, and the active
org's directory for `claude-science`.

Inside the package the agent creates `<import-pkg>/_skills/` with `install.py` (a canonical
template it must copy verbatim) and `data/` holding the skill files, then wires the console
script and — the step that silently fails — the **build-backend packaging** so those
non-`.py` data files actually ship in the wheel. It verifies by building a wheel, installing
it into a fresh (non-editable) venv, and confirming `SKILL.md` lands.

## Delivery depends on `repo`

- **GitHub URL** → the agent branches, commits, pushes, and opens a PR with `gh` (it does
  not merge). Needs write access; a rejected push is reported, not worked around.
- **Local path** → the change is written straight into your working tree, no branch, no
  commit. Review with `git diff`.

## Non-obvious behaviour

- **The ship agent is deliberately NOT isolated.** Unlike every other acumen agent it runs
  in your real environment: real `HOME`, real network, real git/`gh` credentials, real `uv`,
  with `permission_mode="bypassPermissions"`. Only the model credential is constrained (by
  `--auth`). Run it when you are ready for it to touch your checkout and remote.
- It refuses to run if the target already declares a `*-install-skills` console script or
  has a `_skills/install.py` — pass `--force` to ship anyway.
- It ships exactly one version, copied verbatim; it never authors or edits skill text, and
  it writes no tests.
- Unbounded turns/cost by default. Its final summary (PR URL, packaging changes,
  build-verify result) is printed and returned.
