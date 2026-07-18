# API

acumen is a CLI (`acumen init/tasks/draft/bench/improve/report`) that is a thin shell over an
importable Python API. Everything below is re-exported from the top-level `acumen` package,
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
    build_filtered_source
    find_skill_access
    make_skill_guard
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

## Reporting

```{eval-rst}
.. autosummary::
    :toctree: generated

    build_report
    load_results
    arm_metrics
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
