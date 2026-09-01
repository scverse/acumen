# Python API

Everything is re-exported from the top level: `from acumen import build_report, load_config, …`.
The CLI (`acumen.cli:main`) is a thin shell over it, so anything the CLI does is scriptable.
`main(argv)` returns an exit code and converts the library's error types into `error: …` on
stderr — call the functions directly if you want the exceptions.

## Loading config and tasks

```python
from pathlib import Path
from acumen import load_config, load_tasks, parse_config, parse_tasks

cfg = load_config(Path("config.yaml"))  # validates; resolves a local repo relative to the file
tasks = load_tasks(Path("tasks.yaml"))  # list[Task]; task.split("train") -> TaskSplit(prompt, answer)
```

`Config` is a frozen dataclass — override with `dataclasses.replace(cfg, n_replicates=1)`.
`parse_config` / `parse_tasks` take already-parsed dicts, for building them in memory.

## Preparing the target

```python
from acumen import prepare_target
from acumen.env import DEFAULT_CACHE_ROOT

target = prepare_target(cfg, DEFAULT_CACHE_ROOT, refresh=False)
target.python  # venv interpreter with the package installed
target.bin_dir  # what goes on an agent's PATH
target.fingerprint  # "<pkg> <version>", as recorded in result.json
```

Needs `uv` on PATH. Cached by (repo, ref) under `~/.cache/acumen`.

## Planning and running a pass

```python
import asyncio
from acumen import build_matrix, pending, run_matrix, load_skill, summarize

skill = load_skill(Path("skills"), "v1", expect_name=cfg.skill_name)  # None for the baseline
planned = build_matrix(cfg, tasks, skill="v1", splits=("train",), task_ids=["t1"])
todo = pending(planned, Path("runs"), resume=True)
outcomes = asyncio.run(
    run_matrix(
        todo,
        target=target,
        runs_root=Path("runs"),
        max_concurrency=cfg.max_concurrency,
        auth_mode="api",
        skill=skill,
        on_start=lambda p: None,
        on_done=lambda o: None,
        env_passthrough=cfg.env_passthrough,
    )
)
summarize(outcomes)  # {"ok": 5, "wrong_answer": 1, ...}
```

`run_matrix` and `run_once` are coroutines. `run_once` records agent crashes as failed runs
rather than raising, so one bad run never kills a pass. `run_once` raises `ValueError` if
`key.arm` and the `skill` argument disagree — the arm is the source of truth.

`build_matrix` (and `--dry-run`) is pure planning: no agents, no network, no cost.

## Grading and paths

```python
from acumen import grade_answer, grade_run, run_dir, parse_run_dir, is_complete, RunKey, arm_name

grade_answer("**SPI1**", "SPI1")  # Grade(success=False, reason='format_error', answer='**SPI1**')
run_dir(Path("runs"), RunKey(arm=arm_name("v1"), split="test", model="claude-opus-5", task_id="t1", rep=1))
is_complete(d)  # a non-empty result.json is what "done" means
```

## The meta-agents

`draft_skill`, `improve_skill`, `generate_tasks`, `ship_skill` are all coroutines taking
keyword-only args (`cfg=`, `target=`, plus their own roots) and returning a result dataclass
(`DraftResult`, `ImproveResult`, `TaskGenResult`, `ShipResult`) carrying the new `Skill`,
`cost_usd`, `turns`, and log paths. `max_turns`/`max_usd` default to `None` = **unbounded**.
Pass a `LiveLog` as `log=` for the JSONL feed:

```python
from acumen import LiveLog, draft_skill

log = LiveLog.open(Path("logs"), "draft", stream=False)
with log:
    result = asyncio.run(draft_skill(cfg=cfg, target=target, skills_root=Path("skills"), auth_mode="session", log=log))
```

## Aggregating results

```python
from acumen import load_results, arm_metrics, build_report

df = load_results(Path("runs"))  # one row per result.json, + total_tokens, arm_label
arm_metrics(df[df["split"] == "test"])  # per-arm rate, stderr, tokens, cost, time, n
report = build_report(Path("runs"), Path("report.html"), tasks, skills_root=Path("skills"))
report.n_runs, report.results  # the DataFrame behind the HTML
```

`build_report` writes `report.html` **and `report.csv`** (same stem) and returns `Report`.
Filter `df` to a split yourself before `arm_metrics` — it does not.

## Introspection helpers

`available_versions`, `latest_version`, `next_version`, `skill_hash`, `skill_content`
(for diffing), `installer_exists(src_dir)`, `collect_train_runs(runs_root, arm, tasks)`,
and the two pure guards `find_test_access` / `find_skill_access` (testable without an agent).
`scrubbed_env` / `build_agent_env` / `sandbox` / `install_skill` let you reproduce a run's
exact isolated environment by hand.
