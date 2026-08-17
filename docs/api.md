# API

acumen is a CLI (`acumen init/tasks/check/draft/bench/improve/report/ship`) that is a thin shell over
an importable Python API. Everything below is re-exported from the top-level `acumen` package,
so `from acumen import build_report` works.

## Configuration and tasks

```{eval-rst}
.. currentmodule:: acumen

.. autosummary::
    :toctree: generated

    Config
    load_config
    parse_config
    Task
    TaskSplit
    load_tasks
    parse_tasks
```

## Scaffolding a project

```{eval-rst}
.. autosummary::
    :toctree: generated

    scaffold
```

## Generating tasks

```{eval-rst}
.. autosummary::
    :toctree: generated

    generate_tasks
    TaskGenResult
    dump_tasks
    harvest_scripts
    Harvest
    build_filtered_source
    find_skill_access
    make_skill_guard
```

## Checking task ground truth

`acumen check` runs in two phases, because a task can be broken in two ways.

The first is deterministic and spends nothing: run the reproducer for each task and split —
`tasks/<id>-<split>.py`, which writes its answer to `answer.md` the way a benchmark run does — in
the target venv, and grade it against the answer recorded in `tasks.yaml`. A task that needs no
code to answer sets `needs_script: false` and is reported as `skipped`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    check_tasks
    check_task_split
    run_reproducer
    summarize_checks
    CheckResult
    CheckSummary
    ScriptRun
    script_path
    orphan_scripts
    select_tasks
    import_probe
```

The second is agentic: reproducing the answer proves the code and the answer agree, not that the
*prompt* asks for what they produce. One agent reads every split's prompt, recorded answer and
reproducer together and returns `ok` or `mismatch`, with one line naming the contradiction and one
naming the fix. `--no-review` skips it.

```{eval-rst}
.. autosummary::
    :toctree: generated

    review_tasks
    ReviewResult
    ReviewVerdict
    parse_reviews
    write_packet
```

## Target environment and sandboxing

```{eval-rst}
.. autosummary::
    :toctree: generated

    Target
    prepare_target
    scrubbed_env
    Sandbox
    sandbox
    install_skill
```

## Skills

```{eval-rst}
.. autosummary::
    :toctree: generated

    Skill
    SkillMeta
    load_skill
    skill_hash
    skill_content
    available_versions
    latest_version
    next_version
```

## Benchmarking

```{eval-rst}
.. autosummary::
    :toctree: generated

    PlannedRun
    BenchmarkInvalidError
    build_matrix
    pending
    run_matrix
    run_once
    RunOutcome
    summarize
    grade_answer
    grade_run
    Grade
```

## Drafting and improving

```{eval-rst}
.. autosummary::
    :toctree: generated

    draft_skill
    DraftResult
    improve_skill
    ImproveResult
    collect_train_runs
    TrainRun
    find_test_access
    make_test_guard
```

## Shipping

`ship_skill` wires a `<dist>-install-skills` console script into the target package. The
installed command drops the skill into whichever agentic framework the user names —
`--agent {claude,codex,agents,claude-science}` (or an explicit `--dest`) — copying the same
`SKILL.md` + `references/` bundle verbatim; there is no default framework and no per-framework
conversion.

```{eval-rst}
.. autosummary::
    :toctree: generated

    ship_skill
    ShipResult
    installer_exists
```

## Agent run logs

```{eval-rst}
.. autosummary::
    :toctree: generated

    LiveLog
    locate_transcript
    render_agent_transcript
    render_transcript
    render_codex_transcript
```

## Reporting

```{eval-rst}
.. autosummary::
    :toctree: generated

    build_report
    load_results
    arm_metrics
    resolve_palette
    Report
```

## Run-directory layout

```{eval-rst}
.. autosummary::
    :toctree: generated

    RunKey
    run_dir
    parse_run_dir
    arm_name
    skill_from_arm
    is_complete
```
