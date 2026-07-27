"""Unit tests for the pure parts the CLI sits on: grader, schemas, paths, skills.

Deliberately thin — one test per behaviour that would silently corrupt a benchmark if it
broke, not an exhaustive sweep of each validator.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

from acumen.bench import build_matrix, pending
from acumen.config import ConfigError, derive_skill_name, load_config, parse_config
from acumen.env import (
    AUTH_ENV_VARS,
    EnvError,
    Target,
    api_auth_available,
    auth_available,
    build_agent_env,
    check_auth,
    resolve_auth_mode,
    scrubbed_env,
    session_auth_available,
)
from acumen.grade import grade_answer, grade_run
from acumen.improve import _write_material, collect_train_runs, load_rates
from acumen.paths import RunKey, arm_name, is_complete, parse_run_dir, run_dir
from acumen.procs import label_env, reap, supported, survivors
from acumen.prompts import draft_prompt, feedback_block, improve_prompt
from acumen.report import (
    ReportError,
    _arm_marker,
    _best_cells,
    _holm,
    _integrity_notes,
    _loaded_rank,
    _pareto_front,
    _pareto_steps,
    _runs_table_html,
    _skill_diff_html,
    _split_diff_rows,
    _tests_table_html,
    load_results,
    metrics_figure,
    resolve_palette,
    skill_tests,
    tradeoff_figure,
)
from acumen.runner import StderrFilter, _skill_fired
from acumen.ship import _ship_env
from acumen.skills import SkillError, load_skill, read_meta, skill_hash, write_meta
from acumen.tasks import TaskError, load_tasks, parse_tasks

# --- grading ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected", "success", "reason"),
    [
        ("SPI1", "SPI1", True, "ok"),
        ("  SPI1\n", "SPI1", True, "ok"),  # graded after strip()
        ("STAT1", "SPI1", False, "wrong_answer"),
        ("**SPI1**", "SPI1", False, "format_error"),  # right content, forbidden formatting
        (None, "SPI1", False, "no_answer_file"),
    ],
)
def test_grade_answer(answer: str | None, expected: str, success: bool, reason: str) -> None:
    grade = grade_answer(answer, expected)
    assert (grade.success, grade.reason) == (success, reason)


def test_grade_run_reads_answer_md(tmp_path: Path) -> None:
    (tmp_path / "answer.md").write_text("SPI1\n")
    assert grade_run(tmp_path, "SPI1").success
    assert grade_run(tmp_path / "empty", "SPI1").reason == "no_answer_file"


# --- run paths -------------------------------------------------------------------------


def test_run_dir_round_trips(tmp_path: Path) -> None:
    key = RunKey(arm="skill_v2", split="test", model="claude-opus-5", task_id="tf_activity", rep=3)
    directory = run_dir(tmp_path, key)

    assert directory == tmp_path / "skill_v2/test/claude-opus-5/tf_activity/rep_3"
    assert parse_run_dir(tmp_path, directory) == key
    assert key.skill == "v2"


def test_arm_name() -> None:
    assert arm_name(None) == "noskill"
    assert arm_name("v1") == "skill_v1"


def test_is_complete_needs_a_result_file(tmp_path: Path) -> None:
    assert not is_complete(tmp_path)
    (tmp_path / "result.json").write_text("{}")
    assert is_complete(tmp_path)


# --- config / tasks --------------------------------------------------------------------


def test_config_defaults_and_derived_skill_name() -> None:
    cfg = parse_config({"repo": "https://github.com/scverse/scanpy"})

    assert cfg.skill_name == "scanpy"
    assert cfg.ref == "main"
    assert cfg.n_replicates == 3
    assert cfg.env_passthrough == []
    assert not cfg.is_local
    assert derive_skill_name("git@github.com:scverse/scanpy.git") == "scanpy"

    cfg2 = parse_config({"repo": "https://github.com/scverse/scanpy", "env_passthrough": ["OMP_NUM_THREADS"]})
    assert cfg2.env_passthrough == ["OMP_NUM_THREADS"]


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config({"repo": "https://example.com/pkg", "modles": ["x"]})


def test_load_config_resolves_a_local_repo(project: Path) -> None:
    cfg = load_config(project / "config.yaml")

    assert cfg.is_local
    assert Path(cfg.repo) == (project / "target").resolve()


def test_load_tasks(project: Path) -> None:
    (task,) = load_tasks(project / "tasks.yaml")

    assert task.id == "example_task"
    assert task.split("train").answer == "TRAIN_ANSWER"
    assert task.split("test").prompt.startswith("Do the same")


def test_tasks_reject_duplicate_ids() -> None:
    entry = {"id": "dup", "train": {"prompt": "p", "answer": "a"}, "test": {"prompt": "p", "answer": "a"}}
    with pytest.raises(TaskError, match="duplicate task id"):
        parse_tasks({"tasks": [entry, dict(entry)]})


# --- matrix ----------------------------------------------------------------------------


def test_build_matrix_and_resume(project: Path, model: str, make_result) -> None:
    cfg = load_config(project / "config.yaml")
    tasks = load_tasks(project / "tasks.yaml")

    planned = build_matrix(cfg, tasks, skill="v1")
    assert len(planned) == 2  # 1 task x 2 splits x 1 model x 1 replicate
    assert {p.key.split for p in planned} == {"train", "test"}
    assert all(p.key.arm == "skill_v1" for p in planned)

    runs = project / "runs"
    make_result(runs, RunKey(arm="skill_v1", split="train", model=model, task_id="example_task", rep=1))
    assert [p.key.split for p in pending(planned, runs)] == ["test"]
    assert len(pending(planned, runs, resume=False)) == 2


def test_skill_fired_matches_the_skill_under_test_only(tmp_path: Path) -> None:
    def transcript(*skills: str) -> Path:
        path = tmp_path / f"{'-'.join(skills) or 'none'}.jsonl"
        records = [
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "Skill", "input": {"skill": name}}]},
            }
            for name in skills
        ]
        path.write_text("".join(f"{json.dumps(r)}\n" for r in records))
        return path

    assert _skill_fired(transcript("target"), "target") is True
    # Skills bundled with the CLI are reachable in either arm and are not what is measured.
    assert _skill_fired(transcript("dataviz", "init"), "target") is False
    assert _skill_fired(transcript("dataviz", "target"), "target") is True
    assert _skill_fired(transcript(), "target") is False
    # A missing transcript is unknown, not a miss.
    assert _skill_fired(tmp_path / "absent.jsonl", "target") is None


def test_stderr_filter_keeps_first_of_each_line() -> None:
    import io

    sink = io.StringIO()
    emit = StderrFilter(sink=sink)
    warn = "⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY is set"

    for _ in range(4):  # the same per-spawn warning fires once per run in a real pass
        emit(warn)
    emit("a distinct line")
    emit(warn)  # a later repeat is still dropped

    assert sink.getvalue() == f"{warn}\na distinct line\n"


# --- skills ----------------------------------------------------------------------------


def test_load_skill(skills_root: Path) -> None:
    skill = load_skill(skills_root, "v1", expect_name="target")

    assert (skill.version, skill.name, skill.number) == ("v1", "target", 1)
    assert skill.hash.startswith("sha256:")


def test_load_skill_rejects_a_name_mismatch(skills_root: Path) -> None:
    with pytest.raises(SkillError, match="config.skill_name"):
        load_skill(skills_root, "v1", expect_name="something_else")


def test_skill_hash_ignores_meta_json(skills_root: Path) -> None:
    """``meta.json`` carries the hash, so hashing it would be circular."""
    directory = skills_root / "v1"
    before = skill_hash(directory)

    (directory / "meta.json").write_text('{"rationale": "rewritten"}')
    assert skill_hash(directory) == before

    (directory / "SKILL.md").write_text("---\nname: target\ndescription: d\n---\nnew body\n")
    assert skill_hash(directory) != before


def test_feedback_block_absent_is_empty_and_present_is_subordinated() -> None:
    """No ``--feedback`` must leave the prompt byte-identical; present text is subordinated."""
    assert feedback_block(None) == ""
    assert feedback_block("   ") == ""

    base_kwargs = {
        "package": "p",
        "version": "1",
        "src": Path("/s"),
        "python": Path("/py"),
        "out": Path("/o"),
        "skill_name": "p",
    }
    assert draft_prompt(**base_kwargs) == draft_prompt(**base_kwargs, feedback=None)

    steered = draft_prompt(**base_kwargs, feedback="skip the plotting API")
    assert "skip the plotting API" in steered
    assert "does NOT override" in steered
    # Guidance sits after the how-to rules but before the closing deliverable reminder.
    assert steered.index("skip the plotting API") < steered.index("When you are done")


def test_write_meta_persists_feedback_but_omits_it_when_absent(tmp_path: Path) -> None:
    directory = tmp_path / "v1"
    directory.mkdir()
    (directory / "SKILL.md").write_text("---\nname: target\ndescription: d\n---\nbody\n")

    write_meta(directory, parent=None, rationale="initial draft")
    assert "feedback" not in (directory / "meta.json").read_text()
    assert read_meta(directory).feedback is None

    write_meta(directory, parent="v1", rationale="fixed", feedback="  emphasise pseudobulk  ")
    assert read_meta(directory).feedback == "emphasise pseudobulk"


# --- improve evidence ------------------------------------------------------------------


def _train_evidence(project: Path, make_result, loaded: list[bool | None]) -> list:
    """Write one train run per entry of ``loaded``, then collect them as improver evidence."""
    runs = project / "runs"
    models = ("model_a", "model_b")
    for i, flag in enumerate(loaded):
        key = RunKey(arm="skill_v1", split="train", model=models[i % 2], task_id="example_task", rep=i + 1)
        make_result(runs, key, success=flag is True, skill_loaded=flag)
    return collect_train_runs(runs, "skill_v1", load_tasks(project / "tasks.yaml"))


def test_collect_train_runs_carries_load_status(project: Path, make_result) -> None:
    """A run's ``skill_loaded`` must reach the improver, with undetermined kept distinct."""
    runs = _train_evidence(project, make_result, [True, False, None])

    assert sorted(r.skill_loaded is True for r in runs).count(True) == 1
    assert [r.skill_loaded for r in runs].count(False) == 1
    # Unreadable transcripts stay None — undetermined is not the same as "did not load".
    assert [r.skill_loaded for r in runs].count(None) == 1


def test_load_rates_split_by_model_and_count_undetermined(project: Path, make_result) -> None:
    """Rates are per model, since the load rate varies more by model than by skill version."""
    rates = load_rates(_train_evidence(project, make_result, [True, True, False, None]))

    # model_a took reps 1 and 3 (True, False); model_b took reps 2 and 4 (True, None).
    assert rates == {"model_a": (1, 2, 0), "model_b": (1, 2, 1)}


def test_written_evidence_reports_load_rate_and_marks_each_run(project: Path, make_result, tmp_path: Path) -> None:
    """The improver reads SUMMARY.md, so the load signal has to survive into the file."""
    runs = _train_evidence(project, make_result, [True, False, None])
    train_dir = tmp_path / "train"

    _write_material(train_dir, runs)

    summary = (train_dir / "SUMMARY.md").read_text()
    assert "Did the skill load at all?" in summary
    assert "| `model_a` |" in summary and "| `model_b` |" in summary
    assert "skill LOADED" in summary
    assert "skill NOT LOADED" in summary
    assert "skill load UNDETERMINED" in summary

    # Every per-run page states it too, so a reader who opens one run isn't left guessing.
    pages = [p.read_text() for p in train_dir.rglob("run.md")]
    assert len(pages) == 3
    assert all("- Skill: skill " in page for page in pages)


def test_improve_prompt_separates_loading_from_the_body() -> None:
    """The prompt must not let a never-loaded run be read as evidence against the body."""
    prompt = improve_prompt(
        package="p",
        version="1",
        python=Path("/py"),
        skill_dir=Path("/skill"),
        train_dir=Path("/train"),
        rationale_path=Path("/r.md"),
        skill_name="p",
        parent_version="v1",
        new_version="v2",
    )

    assert "The skill never loaded" in prompt
    assert "description" in prompt
    # Raising the load rate must not become licence to name the train tasks in it.
    assert prompt.index("Do not overfit it to the train tasks") < prompt.index("When you are done")


# --- auth preflight --------------------------------------------------------------------


def test_auth_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point credential discovery at an empty throwaway dir and strip every auth variable,
    # so the only auth in play is what the test sets.
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    for var in AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    # No variable and no credentials file → unauthenticated, and check_auth fails loudly.
    assert auth_available() is False
    with pytest.raises(EnvError, match="no Claude credentials"):
        check_auth()

    # An empty variable does not count.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert auth_available() is False

    # A non-empty auth variable authenticates.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert auth_available() is True
    check_auth()  # does not raise

    # So does a seeded credentials file, even with no auth variable set.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".credentials.json").write_text("{}")
    assert auth_available() is True
    check_auth()


def _clear_auth(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    """Isolate auth detection: empty credential dir, every auth variable stripped."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    for var in AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_oauth_credentials(config_dir: Path) -> None:
    """Write a credentials file shaped like a real `claude` subscription login."""
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / ".credentials.json").write_text('{"claudeAiOauth": {"accessToken": "x"}}')


def test_session_and_api_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)

    # Nothing present → neither mode is available.
    assert session_auth_available() is False
    assert api_auth_available() is False

    # An API key is API auth, not a subscription.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert api_auth_available() is True
    assert session_auth_available() is False

    # The OAuth token is a subscription credential, not API auth.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    assert session_auth_available() is True
    assert api_auth_available() is False

    # A bare credentials file with no OAuth block is not a subscription…
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    (tmp_path / ".credentials.json").write_text("{}")
    assert session_auth_available() is False
    # …but a real `claude` login (claudeAiOauth) is.
    _write_oauth_credentials(tmp_path)
    assert session_auth_available() is True


def test_resolve_auth_mode_for_meta_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)

    # No credentials at all → auto cannot resolve.
    with pytest.raises(EnvError, match="no Claude credentials"):
        resolve_auth_mode("auto", allow_session=True)

    # auto prefers the subscription when a login exists.
    _write_oauth_credentials(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_auth_mode("auto", allow_session=True) == "session"

    # auto falls back to the API when there is no subscription.
    (tmp_path / ".credentials.json").unlink()
    assert resolve_auth_mode("auto", allow_session=True) == "api"

    # Forcing a mode requires that mode's credential.
    assert resolve_auth_mode("api", allow_session=True) == "api"
    with pytest.raises(EnvError, match="--auth session"):
        resolve_auth_mode("session", allow_session=True)
    _write_oauth_credentials(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_auth_mode("session", allow_session=True) == "session"
    with pytest.raises(EnvError, match="--auth api"):
        resolve_auth_mode("api", allow_session=True)


def test_resolve_auth_mode_for_bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)

    # bench never bills the subscription, even when only a subscription is available.
    _write_oauth_credentials(tmp_path)
    with pytest.raises(EnvError, match="session is not available for it"):
        resolve_auth_mode("session", allow_session=False)
    with pytest.raises(EnvError, match="must bill the API"):
        resolve_auth_mode("api", allow_session=False)

    # With an API credential, bench resolves to api.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert resolve_auth_mode("api", allow_session=False) == "api"


def test_scrubbed_env_auth_mode_filters_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    home = tmp_path / "home"

    # The unwanted credential is set to "" (not omitted) so it overrides the inherited value
    # under the SDK's {**os.environ, **options.env} merge — see the merge regression below.
    # session keeps only the subscription token; api keeps only the API key.
    session_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session")
    assert session_env["ANTHROPIC_API_KEY"] == ""
    assert session_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"

    api_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api")
    assert api_env["ANTHROPIC_API_KEY"] == "sk-test"
    assert api_env["CLAUDE_CODE_OAUTH_TOKEN"] == ""

    # No mode leaves both credentials in place (the historical behavior).
    both = scrubbed_env(config_dir=tmp_path / "cfg", home=home)
    assert both["ANTHROPIC_API_KEY"] == "sk-test"
    assert both["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


def test_session_mode_neutralizes_the_api_key_under_the_sdk_env_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The credential drop must survive the SDK's env merge, not just the returned dict.

    The SDK builds the agent subprocess env as ``{**os.environ, **options.env}``, so a
    credential we *omit* from our mapping falls back through from ``os.environ`` and the run
    bills the wrong path. Setting it to "" is what actually neutralizes it. This guards the
    session-mode meta-agents (draft/improve/tasks and the unscrubbed ship env) from silently
    billing the API when ``ANTHROPIC_API_KEY`` is present in the environment.
    """
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token")
    home = tmp_path / "home"
    target = Target(
        source="/pkg",
        ref="main",
        src_dir=tmp_path / "src",
        venv_dir=tmp_path / "venv",
        commit="abc123",
        pkg_name="pkg",
        pkg_version="1.0",
    )

    # scrubbed_env (draft/improve/tasks) and the unscrubbed ship env must both hold up.
    for agent_env in (
        scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session"),
        _ship_env(target, "session"),
    ):
        merged = {**os.environ, **agent_env}  # what the SDK actually hands the subprocess
        assert not merged["ANTHROPIC_API_KEY"], "API key leaked into a session-mode agent"
        assert merged["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


def test_scrubbed_env_blanks_ambient_vars_under_the_sdk_env_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-allowlisted ambient variable must not reach the agent through the env merge.

    The SDK builds the agent env as ``{**os.environ, **options.env}``, so a variable
    scrubbed_env merely omits falls straight through from the operator's shell into the
    web-enabled agent. scrubbed_env therefore blanks every inherited variable it did not
    keep, and the check that matters is on the *merged* mapping, not the returned dict.
    """
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # allowlisted — must survive
    monkeypatch.setenv("MY_APP_SECRET", "hunter2")  # ambient secret — must be neutralized
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # target-needed — kept only via env_passthrough
    home = tmp_path / "home"

    env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api")
    merged = {**os.environ, **env}  # what the SDK actually hands the subprocess
    assert merged["ANTHROPIC_API_KEY"] == "sk-test"  # allowlisted credential preserved
    assert not merged["MY_APP_SECRET"], "ambient secret leaked into a benchmark agent"
    assert not merged["OMP_NUM_THREADS"], "non-allowlisted var leaked without env_passthrough"
    # Our throwaway overrides still land.
    assert merged["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert merged["HOME"] == str(home)


def test_env_passthrough_carries_declared_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable named in env_passthrough survives the scrub; an undeclared one does not."""
    _clear_auth(monkeypatch, tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MY_APP_SECRET", "hunter2")
    home = tmp_path / "home"

    env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api", extra_allow=["OMP_NUM_THREADS"])
    merged = {**os.environ, **env}
    assert merged["OMP_NUM_THREADS"] == "8", "declared passthrough var was dropped"
    assert not merged["MY_APP_SECRET"], "undeclared var leaked"

    # A declared var that isn't actually set in the shell is simply absent, not blanked to "".
    env2 = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api", extra_allow=["NOT_SET_ANYWHERE"])
    assert "NOT_SET_ANYWHERE" not in env2


def test_build_agent_env_seeds_only_in_session_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _clear_auth(monkeypatch, source)
    _write_oauth_credentials(source)  # the user's real login, discovered via CLAUDE_CONFIG_DIR
    home = tmp_path / "home"

    session_cfg = tmp_path / "session_cfg"
    build_agent_env(config_dir=session_cfg, home=home, auth_mode="session")
    assert (session_cfg / ".credentials.json").is_file()  # seeded

    api_cfg = tmp_path / "api_cfg"
    build_agent_env(config_dir=api_cfg, home=home, auth_mode="api")
    assert not (api_cfg / ".credentials.json").exists()  # not seeded


def test_resolve_palette_overrides_by_id_and_by_label() -> None:
    """An override wins over the tier default, keyed by the raw id or its display form."""
    models = ["claude-opus-5", "claude-haiku-4-5-20251001"]

    colors = resolve_palette(models, {"claude-opus-5": "#3b7ea1", "claude-haiku-4-5": "rebeccapurple"})

    assert colors["claude-opus-5"] == "#3b7ea1"
    # The legend strips the snapshot date, so that form is accepted as a key too.
    assert colors["claude-haiku-4-5-20251001"] == "rebeccapurple"
    # Every model gets an entry, overridden or not.
    assert resolve_palette(models, None) == resolve_palette(models, {})
    assert set(resolve_palette(models, None)) == set(models)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"claude-opus-5": "not-a-colour"}, "not a colour"),
        ({"gpt-4": "#3b7ea1"}, "no such model"),
    ],
)
def test_resolve_palette_rejects_bad_input(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ReportError, match=message):
        resolve_palette(["claude-opus-5"], overrides)


def test_metrics_figure_paints_bars_with_the_palette(runs_root: Path, model: str) -> None:
    """The resolved colour reaches the bars themselves, not just the legend."""
    df = load_results(runs_root)
    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        faces = {to_hex(patch.get_facecolor()) for ax in figure.axes for patch in ax.patches}
        assert "#3b7ea1" in faces
    finally:
        plt.close(figure)


def test_metrics_figure_pools_every_model_into_a_last_grey_bar(runs_root: Path, model: str, make_result) -> None:
    """With several models, each cell ends in a grey bar over all of them pooled.

    It is the rate across every run, not the mean of the per-model rates — an under-sampled
    model must not weigh as much as a well-sampled one.
    """
    other = "claude-opus-5"
    key = RunKey(arm="noskill", split="test", model=other, task_id="example_task", rep=1)
    make_result(runs_root, key, success=False)
    make_result(runs_root, replace(key, rep=2), success=False)
    df = load_results(runs_root)  # the fixture's one passing test run, plus two failing ones

    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model, other]))
    try:
        (rate_ax,) = [ax for ax in figure.axes if ax.get_title() == "Success rate"]
        bars = rate_ax.patches
        widths = [bar.get_width() for bar in bars]
        pooled_color = to_hex(bars[-1].get_facecolor())
    finally:
        plt.close(figure)

    # Most potent model first, then the pooled bar last: opus 0/2, haiku 1/1, overall 1/3.
    assert widths == pytest.approx([0.0, 1.0, 1 / 3])
    assert pooled_color not in {to_hex(c) for c in resolve_palette([model, other]).values()}


def test_skill_loaded_column_counts_undetermined_runs_as_not_loaded(runs_root: Path, model: str, make_result) -> None:
    """The load-rate bar is the share of runs that loaded the skill under test.

    A run whose transcript could not be read is undetermined, not evidence of a load, so it
    counts against the rate — the same reading the runs CSV takes.
    """
    for rep, loaded in ((1, True), (2, None)):
        key = RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=rep)
        make_result(runs_root, key, skill_loaded=loaded)
    df = load_results(runs_root)

    figure = metrics_figure(df, split_hue=False, colors=resolve_palette([model]))
    try:
        # Only the top row is titled, so the title locates the column; the grid is
        # row-major with one row per arm — noskill first, then skill v1.
        columns = [i for i, ax in enumerate(figure.axes) if ax.get_title() == "Skill loaded"]
        assert len(columns) == 1
        n_cols = len(figure.axes) // 2
        widths = [figure.axes[columns[0] + row * n_cols].patches[0].get_width() for row in range(2)]
    finally:
        plt.close(figure)

    assert widths == [0.0, 0.5]  # the baseline loaded nothing; one of two skill runs is determined


@pytest.mark.parametrize(("loads", "warned"), [((True, True, False), False), ((True, False, False), True)])
def test_load_warning_needs_most_of_the_arm_to_miss(
    runs_root: Path, model: str, make_result, loads: tuple[bool, ...], warned: bool
) -> None:
    """A skill arm is only flagged once fewer than half its runs loaded the skill.

    Misses in a minority of runs are ordinary, and the per-run table already marks each one,
    so warning about them at the top of the report would cry wolf on a healthy arm.
    """
    for rep, loaded in enumerate(loads, start=1):
        key = RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=rep)
        make_result(runs_root, key, skill_loaded=loaded)

    notes = _integrity_notes(load_results(runs_root))

    assert bool(notes) is warned


# --- the cost/success trade-off figure --------------------------------------------------

#: The two markers matplotlib gives an error bar's caps; neither is a data point.
_CAP_MARKERS = {"|", "_"}


def _pooled_marks(figure: plt.Figure) -> list[tuple[float, float]]:
    """``(cost, rate)`` of each pooled mark, in the order the arms are drawn.

    The pooled marks are the only ones carrying error bars, so they are exactly the axes'
    error-bar containers.
    """
    return [(float(c.lines[0].get_xdata()[0]), float(c.lines[0].get_ydata()[0])) for c in figure.axes[0].containers]


def _model_marks(figure: plt.Figure) -> list[plt.Line2D]:
    """Every per-model point mark — no pooled marks, no error-bar caps, no reference lines."""
    ax = figure.axes[0]
    pooled = {id(c.lines[0]) for c in ax.containers}
    return [
        line for line in ax.lines if str(line.get_marker()) not in _CAP_MARKERS | {"None"} and id(line) not in pooled
    ]


def _frontier(figure: plt.Figure) -> list[tuple[float, float]]:
    """The vertices of the drawn Pareto staircase — the only marker-less line on the axes."""
    (line,) = [ln for ln in figure.axes[0].lines if str(ln.get_marker()) == "None"]
    return [(float(x), float(y)) for x, y in zip(*line.get_data(), strict=True)]


def test_tradeoff_pooled_mark_averages_runs_not_per_model_means(runs_root: Path, model: str, make_result) -> None:
    """The pooled mark sits where an average *run* lands, whichever model drew it.

    Averaging the per-model points instead would let a model with two runs count for no more
    than one with a single run — the same trap the grid's grey bar avoids.
    """
    other = "claude-opus-5"
    key = RunKey(arm="noskill", split="test", model=other, task_id="example_task", rep=1)
    make_result(runs_root, key, success=False, cost_usd=0.30)
    make_result(runs_root, replace(key, rep=2), success=False, cost_usd=0.30)
    df = load_results(runs_root)  # the fixture's one passing $0.12 run, plus two failing $0.30 ones

    figure = tradeoff_figure(df)
    try:
        (pooled,) = _pooled_marks(figure)
    finally:
        plt.close(figure)

    # Over runs: ($0.12 + $0.30 + $0.30)/3 and 1 of 3 passing. Over per-model means it would
    # have been $0.21 and 50%.
    assert pooled == pytest.approx((0.24, 1 / 3))


def test_tradeoff_shape_carries_the_arm_and_colour_carries_the_model(runs_root: Path, model: str, make_result) -> None:
    """Two channels, two meanings: an ✕ then a widening polygon per version, hue left to the model."""
    for arm in ("skill_v1", "skill_v2"):
        make_result(runs_root, RunKey(arm=arm, split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    figure = tradeoff_figure(df, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        markers = {str(line.get_marker()) for line in _model_marks(figure)}
        colors = {to_hex(line.get_color()) for line in _model_marks(figure)}
    finally:
        plt.close(figure)

    # noskill is off the ladder; v1 and v2 are the 3- and 4-sided polygons.
    assert markers == {"X", "(3, 0, 0)", "(4, 0, 0)"}
    assert colors == {"#3b7ea1"}  # the override reaches the marks, not just the legend


def test_tradeoff_frontier_steps_between_the_marks_nothing_beats(runs_root: Path, model: str, make_result) -> None:
    """The staircase holds each rate until the next frontier point's price, then steps up.

    A dominated arm — dearer *and* worse — must leave no trace on the line, and the path must
    never cut a diagonal between two points, which would claim a result nobody measured.
    """
    other = "claude-opus-5"
    # Cheap and good, dear and bad, dear and best: only the first and last are non-dominated.
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=other, task_id="example_task", rep=1))
    make_result(
        runs_root,
        RunKey(arm="skill_v2", split="test", model=other, task_id="example_task", rep=1),
        cost_usd=0.50,
        success=False,
    )
    df = load_results(runs_root)  # plus the fixture's $0.12 passing baseline run

    figure = tradeoff_figure(df)
    try:
        vertices = _frontier(figure)
        y_min = figure.axes[0].get_ylim()[0]
    finally:
        plt.close(figure)

    # Both $0.12 runs pass, so the cheapest 100% mark alone survives; the $0.50 failure is
    # dominated and the riser starts at the axis floor.
    assert vertices == [(pytest.approx(0.12), pytest.approx(y_min)), (pytest.approx(0.12), pytest.approx(1.0))]


def test_tradeoff_handles_an_arm_where_nothing_succeeded(runs_root: Path, model: str, make_result) -> None:
    """A 0% rate is a real result, not a hole: the mark is plotted and the frontier still draws."""
    make_result(
        runs_root,
        RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=1),
        success=False,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        assert _pooled_marks(figure) == [(pytest.approx(0.12), 0.0)]
        assert _frontier(figure)  # degenerate but drawn, rather than crashing on the empty case
    finally:
        plt.close(figure)


def test_tradeoff_plots_the_test_split_only(runs_root: Path, model: str, make_result) -> None:
    """Train runs feed the improver; a report measures held-out performance and must not mix them."""
    make_result(
        runs_root,
        RunKey(arm="noskill", split="train", model=model, task_id="example_task", rep=2),
        cost_usd=99.0,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        (pooled,) = _pooled_marks(figure)
    finally:
        plt.close(figure)

    assert pooled == pytest.approx((0.12, 1.0))  # the fixture's test run alone


def test_tradeoff_keeps_the_two_corner_tick_labels_apart(runs_root: Path, model: str, make_result) -> None:
    """The cost label at the origin is centred on the corner the rate floor label also sits on.

    Left to itself that puts half of one under the other, which is unreadable exactly where the
    reader looks to learn the rate axis is truncated.
    """
    make_result(
        runs_root,
        RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1),
        success=False,
    )
    df = load_results(runs_root)

    figure = tradeoff_figure(df)
    try:
        figure.canvas.draw()  # tick labels have no position until the figure has been laid out
        ax = figure.axes[0]
        # The locators run past the view, so the corner pair are the innermost *visible* labels.
        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        cost = min(
            (t for t in ax.get_xticklabels() if x_lo <= t.get_position()[0] <= x_hi),
            key=lambda t: t.get_position()[0],
        )
        rate = min(
            (t for t in ax.get_yticklabels() if y_lo <= t.get_position()[1] <= y_hi),
            key=lambda t: t.get_position()[1],
        )
        renderer = figure.canvas.get_renderer()
        overlaps = cost.get_window_extent(renderer).overlaps(rate.get_window_extent(renderer))
    finally:
        plt.close(figure)

    assert not overlaps


def test_tradeoff_skill_key_is_unfilled_and_the_model_key_is_not(runs_root: Path, model: str, make_result) -> None:
    """Only one of the two keys is about colour, and the other must not look like it is."""
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    figure = tradeoff_figure(df, colors=resolve_palette([model], {model: "#3b7ea1"}))
    try:
        keys = {legend.get_title().get_text(): legend.legend_handles for legend in figure.legends}
        skill = {handle.get_markerfacecolor() for handle in keys["skill"]}
        models = {to_hex(handle.get_markerfacecolor()) for handle in keys["model"]}
    finally:
        plt.close(figure)

    assert skill == {"none"}  # shape is the channel, so an outline is the whole handle
    assert models == {"#3b7ea1"}  # the model key is the colour key, and keeps its fill


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # Dearer and worse than (0.1, 0.8), so nothing would make you pick it.
        ([(0.1, 0.8), (0.2, 0.7)], [(0.1, 0.8)]),
        # Dearer but better: a real choice, so both stay.
        ([(0.1, 0.8), (0.2, 0.9)], [(0.1, 0.8), (0.2, 0.9)]),
        # Cheaper and better at once — one point dominates the whole set.
        ([(0.2, 0.7), (0.1, 0.9), (0.3, 0.5)], [(0.1, 0.9)]),
        # Same price, so only the better rate survives.
        ([(0.1, 0.8), (0.1, 0.6)], [(0.1, 0.8)]),
        ([], []),
    ],
)
def test_pareto_front_keeps_only_what_nothing_beats_on_both_counts(
    points: list[tuple[float, float]], expected: list[tuple[float, float]]
) -> None:
    """Cheapest first, and a point survives only when nothing is both no dearer and no worse."""
    assert _pareto_front(points) == expected


def test_pareto_steps_never_cuts_a_diagonal() -> None:
    """Between frontier points the best rate on offer is the cheaper one's, so the path holds it.

    Sloping straight from one point to the next would assert results at prices nobody ran.
    """
    xs, ys = _pareto_steps([(0.1, 0.8), (0.2, 0.9)], y_min=0.5)

    assert list(zip(xs, ys, strict=True)) == [(0.1, 0.5), (0.1, 0.8), (0.2, 0.8), (0.2, 0.9)]


# --- significance -----------------------------------------------------------------------

#: Enough tasks that a resample can actually resolve a difference. The bootstrap draws whole
#: tasks, so an effect carried by a single task is invisible in the (n-1 / n)**n of resamples
#: that happen to miss it -- with four tasks that is a third of them.
_TASKS = ("alpha", "beta", "gamma", "delta", "epsilon")


def _arena(runs_root: Path, model: str, make_result, spec: dict[str, tuple[float, list[bool]]]) -> pd.DataFrame:
    """A run tree of ``arm -> (cost per run, one success flag per task)``, one rep each."""
    for arm, (cost, outcomes) in spec.items():
        for task, ok in zip(_TASKS, outcomes, strict=True):
            key = RunKey(arm=arm, split="test", model=model, task_id=task, rep=1)
            make_result(runs_root, key, cost_usd=cost, success=ok)
    return load_results(runs_root)


def test_cheaper_but_worse_does_not_dominate(runs_root: Path, model: str, make_result) -> None:
    """The behaviour the whole design exists for: price cannot buy past a worse success rate.

    A single combined score would rank the cheap arm first — it is a fraction of the cost and only
    slightly less accurate. Requiring *both* axes to improve refuses it, because the evidence for
    dominance is only as strong as its weaker half.
    """
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, True, False, False, False]),
            "skill_v1": (0.10, [True, False, False, False, False]),  # far cheaper, strictly worse
            "skill_v2": (0.10, [True, True, True, True, True]),  # cheaper and better everywhere
        },
    )

    tests = skill_tests(df, resamples=4000)
    by_arm = tests.comparisons.set_index("challenger")

    # Overwhelming evidence on cost, none on rate -> the max is large and the claim fails.
    assert by_arm.loc["skill_v1", "p_cost"] < 0.05
    assert by_arm.loc["skill_v1", "p_rate"] > 0.5
    assert by_arm.loc["skill_v1", "p"] == by_arm.loc["skill_v1", "p_rate"]
    # Better on both axes, so the same rule passes it.
    assert by_arm.loc["skill_v2", "p"] < 0.05


def test_frontier_probability_agrees_with_the_plotted_frontier(runs_root: Path, model: str, make_result) -> None:
    """An arm that dominates every resample is never off the frontier, and a dominated one never on."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.10, [True, True, True, True, True]),  # cheaper and better, always
        },
    )

    tests = skill_tests(df, resamples=2000)
    frontier = tests.arms.set_index("arm")["frontier"]
    observed = _pareto_front(list(zip(tests.arms["cost"], tests.arms["rate"], strict=True)))

    assert frontier["skill_v1"] == 1.0
    assert frontier["noskill"] == 0.0
    # The column and the plot's staircase must pick out the same arm on the observed data.
    assert observed == [(pytest.approx(0.10), pytest.approx(1.0))]


def test_skill_tests_compares_every_version_with_the_baseline_only(runs_root: Path, model: str, make_result) -> None:
    """One family, one correction.

    Version-against-version claims are deliberately absent: correcting them alongside the
    baseline ones spends the budget on comparisons nobody makes and can bury the real result.
    """
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.50, [True, True, False, False, False]),
            "skill_v2": (0.10, [True, True, True, True, True]),
        },
    )

    tests = skill_tests(df, resamples=2000)

    assert tests.baseline == "noskill"
    assert set(tests.comparisons["reference"]) == {"noskill"}
    assert list(tests.comparisons["challenger"]) == ["skill_v1", "skill_v2"]
    assert (tests.comparisons["p_adjusted"] >= tests.comparisons["p"]).all()  # adjusting only costs
    assert tests.comparisons["p_adjusted"].notna().all()  # every row is covered by the correction


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Smallest p times the family size, then one fewer, and so on.
        ([0.01, 0.04, 0.30], [0.03, 0.08, 0.30]),
        # Monotone: 0.021 alone would adjust to 0.021, but the running maximum lifts it to
        # match the more significant result above it, so neither can overtake the other.
        ([0.02, 0.021], [0.04, 0.04]),
        ([0.5, 0.9], [1.0, 1.0]),  # clamped at 1
    ],
)
def test_holm_is_monotone_and_scales_by_remaining_tests(raw: list[float], expected: list[float]) -> None:
    assert _holm(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "highest", "expected"),
    [
        ([0.2, 0.9, 0.5], True, {1}),
        ([0.2, 0.9, 0.5], False, {0}),
        ([None, None, 0.8], True, set()),  # a lone value has beaten nothing, so no emphasis
        ([0.3, 0.3, 0.8], False, {0, 1}),  # a tie marks both rather than crowning the first
        ([None, 0.4, 0.1], False, {2}),  # the baseline's blank cells never win a column
    ],
)
def test_best_cells_follows_each_column_own_direction(
    values: list[float | None], highest: bool, expected: set[int]
) -> None:
    """Best means highest for the rates and lowest for cost and p, so the caller states which."""
    assert _best_cells(values, highest=highest) == expected


def test_tests_table_bolds_the_winner_in_each_column(runs_root: Path, model: str, make_result) -> None:
    """The bold cells must track the column's direction, not simply the largest number."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {
            "noskill": (1.00, [True, False, False, False, False]),
            "skill_v1": (0.10, [True, True, True, True, True]),
        },
    )

    html = _tests_table_html(skill_tests(df, resamples=2000))

    assert "<strong>100.0%</strong>" in html  # highest success rate wins its column
    assert "<strong>$0.100</strong>" in html  # *lowest* cost wins its own
    assert "<strong>$1.000</strong>" not in html
    # The comparison columns sit under one header, so it is clear they all refer to the baseline.
    assert 'colspan="4">Compared with No skill' in html


def test_skill_tests_is_deterministic(runs_root: Path, model: str, make_result) -> None:
    """A rebuilt report must reach the same conclusion — a p-value that drifts is worse than none."""
    df = _arena(
        runs_root,
        model,
        make_result,
        {"noskill": (1.00, [True, False, True, False, False]), "skill_v1": (0.20, [True, True, True, True, False])},
    )

    first, second = skill_tests(df, resamples=2000), skill_tests(df, resamples=2000)

    assert first.comparisons["p"].tolist() == second.comparisons["p"].tolist()
    assert first.arms["frontier"].tolist() == second.arms["frontier"].tolist()


def test_skill_tests_declines_to_test_too_few_tasks(runs_root: Path, model: str, make_result) -> None:
    """With one task every resample is identical, so p would be 0 or 1 by construction.

    The fixture tree has a single task, which is exactly the case that must not silently produce
    confident-looking numbers.
    """
    make_result(runs_root, RunKey(arm="skill_v1", split="test", model=model, task_id="example_task", rep=1))
    df = load_results(runs_root)

    tests = skill_tests(df)

    assert tests.n_clusters == 1
    assert not tests.usable
    assert tests.comparisons.empty
    assert "at least" in _tests_table_html(tests)  # a note, not a table of numbers


def test_skill_tests_uses_the_test_split_only(runs_root: Path, model: str, make_result) -> None:
    """Train runs feed the improver, so letting them into the test would be marking its own work."""
    spec = {"noskill": (1.00, [True, False, True, False, False]), "skill_v1": (0.20, [True, True, True, True, False])}
    before = skill_tests(_arena(runs_root, model, make_result, spec), resamples=2000)

    for task in _TASKS:  # a pile of cheap, always-passing train runs for the baseline
        make_result(runs_root, RunKey(arm="noskill", split="train", model=model, task_id=task, rep=9), cost_usd=0.001)
    after = skill_tests(load_results(runs_root), resamples=2000)

    assert after.comparisons["p"].tolist() == before.comparisons["p"].tolist()
    assert after.arms["cost"].tolist() == before.arms["cost"].tolist()


def test_arm_marker_widens_with_the_version() -> None:
    """The baseline is off the ladder; each version adds a side, so the order is legible."""
    assert _arm_marker("noskill") == "X"
    assert [_arm_marker(f"skill_v{n}") for n in (1, 2, 3, 9)] == [(3, 0, 0), (4, 0, 0), (5, 0, 0), (11, 0, 0)]


# --- the runs table --------------------------------------------------------------------


def test_runs_table_marks_every_column_sortable_but_the_transcript_link(runs_root: Path, tmp_path: Path) -> None:
    """The link column has no ordering, so offering to sort on it would be a dead control."""
    table = _runs_table_html(load_results(runs_root), tmp_path)
    headings = re.findall(r"<th([^>]*)>([^<]+)</th>", table)

    sortable = {text for attrs, text in headings if "data-sortable" in attrs}
    inert = {text for attrs, text in headings if "data-sortable" not in attrs}
    assert inert == {"transcript"}
    assert "cost $" in sortable and "total tok" in sortable


def test_runs_table_sorts_formatted_numbers_by_their_value(
    runs_root: Path, model: str, make_result, tmp_path: Path
) -> None:
    """Displayed text is formatted for reading, so the sort key has to carry the raw number.

    Sorting on the text would put 9 above 10 and a 3-minute run above a 40-second one.
    """
    make_result(
        runs_root,
        RunKey(arm="noskill", split="test", model=model, task_id="example_task", rep=2),
        turns=9,
        input_tokens=1_200_000,
        output_tokens=34_567,
        cost_usd=12.5,
        duration_s=185.0,
    )
    table = _runs_table_html(load_results(runs_root), tmp_path)
    (row,) = [r for r in table.split("<tr") if "1,234,567" in r]

    keys = {text: key for key, text in re.findall(r'<td data-sort="([^"]+)">([^<]*)</td>', row)}
    # The separators, the padded cost and the humanised duration all read differently to
    # the values they stand for.
    assert keys["1,234,567"] == "1234567.0"
    assert keys["12.500"] == "12.5"
    assert float(keys["3.1m"]) == 185.0
    assert keys["9"] == "9.0"


def test_runs_table_ranks_a_skill_that_failed_to_load_above_one_that_loaded() -> None:
    """Sorting on 'skill loaded' is how you find the runs that did not measure their skill."""
    baseline = pd.Series({"arm": "noskill", "skill_loaded": None})
    loaded = pd.Series({"arm": "skill_v1", "skill_loaded": True})
    missed = pd.Series({"arm": "skill_v1", "skill_loaded": False})
    unknown = pd.Series({"arm": "skill_v1", "skill_loaded": None})

    ranks = [_loaded_rank(r) for r in (missed, loaded, unknown, baseline)]

    assert ranks == sorted(ranks, reverse=True)


# --- split diff ------------------------------------------------------------------------


def test_split_diff_pairs_a_rewrite_and_leaves_the_other_side_empty() -> None:
    """A changed line sits opposite its replacement; where one side runs out, it faces nothing."""
    rows = _split_diff_rows(["keep", "old"], ["keep", "new", "extra"])
    assert [(r.left, r.right) for r in rows] == [("keep", "keep"), ("old", "new"), (None, "extra")]
    assert [(r.left_no, r.right_no) for r in rows] == [(1, 1), (2, 2), (None, 3)]
    assert [r.kind for r in rows[1:]] == ["replace", "replace"]  # one run, so both rows tint


def test_split_diff_collapses_untouched_stretches_but_keeps_context() -> None:
    """Three lines of context survive on each side of a change; the rest becomes one gap row."""
    before = [f"line {n}" for n in range(20)]
    after = [*before[:10], "inserted", *before[10:]]
    rows = _split_diff_rows(before, after)
    assert [r.kind for r in rows] == ["gap", *["equal"] * 3, "insert", *["equal"] * 3, "gap"]
    assert [r.left for r in rows if r.kind == "equal"] == [f"line {n}" for n in (7, 8, 9, 10, 11, 12)]


def test_skill_diff_marks_only_the_words_that_changed() -> None:
    """Word-level highlighting inside a rewritten line, on both sides, without touching the rest."""
    diff = _skill_diff_html({"SKILL.md": "use ulm here\n"}, {"SKILL.md": "use mlm here\n"}, ("v001", "v002"))
    assert "<mark>ulm</mark>" in diff and "<mark>mlm</mark>" in diff
    assert "<mark>here</mark>" not in diff


def test_skill_diff_reports_an_unchanged_version_and_a_first_draft() -> None:
    """Neither case has a diff to show, and each says which case it is."""
    content = {"SKILL.md": "same\n"}
    assert "No content change" in _skill_diff_html(content, content, ("v001", "v002"))
    assert "no parent" in _skill_diff_html(None, content, ("", "v001"))


# --- orphan reaping --------------------------------------------------------------------

#: A child that outlives its parent, and a parent that exits immediately after starting it.
#: Reproduces the leak: the SDK terminates the agent, and the commands it started live on
#: with no parent left to find them by.
_ORPHAN = "import time; time.sleep(60)"
_ABANDONING_PARENT = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {_ORPHAN!r}])"

#: Linux, macOS and Windows all qualify; this only skips where psutil cannot read a
#: process's environment, which is the one thing the reaper is built on.
pytestmark_procs = pytest.mark.skipif(not supported(), reason="psutil cannot read process environments here")


def _spawn_orphan(holder: Path) -> None:
    """Start a labelled process whose parent then exits, leaving it running and unparented."""
    parent = subprocess.Popen(
        [sys.executable, "-c", _ABANDONING_PARENT],
        env=label_env(dict(os.environ), holder),
    )
    parent.wait()


@pytestmark_procs
def test_reap_kills_a_command_that_outlived_its_agent(tmp_path: Path) -> None:
    """The whole point: a process whose parent is gone is still found, by its marker, and killed."""
    holder = tmp_path / "run"
    holder.mkdir()
    _spawn_orphan(holder)

    orphans = survivors(holder)
    assert orphans, "the orphaned command was not found after its parent exited"

    assert set(reap(holder)) == set(orphans)
    deadline = time.monotonic() + 5
    while survivors(holder) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not survivors(holder)


@pytestmark_procs
def test_reap_leaves_everything_outside_the_run_alone(tmp_path: Path) -> None:
    """Only the marked run is reaped — not the test process, and not another run's agent."""
    mine = tmp_path / "mine"
    other = tmp_path / "other"
    for path in (mine, other):
        path.mkdir()
    _spawn_orphan(other)
    bystander = subprocess.Popen([sys.executable, "-c", _ORPHAN])
    try:
        assert os.getpid() not in survivors(mine)
        assert bystander.pid not in survivors(mine)
        assert reap(mine) == []
        assert survivors(other), "reaping one run killed another run's processes"
        assert bystander.poll() is None
    finally:
        bystander.kill()
        reap(other)


@pytestmark_procs
def test_label_env_marks_a_run_without_disturbing_the_rest(tmp_path: Path) -> None:
    """Stamping is additive — an agent's carefully built environment is otherwise untouched."""
    holder = tmp_path / "run"
    base = {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-test"}

    marked = label_env(dict(base), holder)

    assert marked.items() >= base.items()
    assert str(holder) in marked.values()
