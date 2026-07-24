"""Unit tests for the pure parts the CLI sits on: grader, schemas, paths, skills.

Deliberately thin — one test per behaviour that would silently corrupt a benchmark if it
broke, not an exhaustive sweep of each validator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from acumen.bench import build_matrix, pending
from acumen.config import ConfigError, derive_skill_name, load_config, parse_config
from acumen.env import (
    AUTH_ENV_VARS,
    EnvError,
    api_auth_available,
    auth_available,
    build_agent_env,
    check_auth,
    resolve_auth_mode,
    scrubbed_env,
    session_auth_available,
)
from acumen.grade import grade_answer, grade_run
from acumen.paths import RunKey, arm_name, is_complete, parse_run_dir, run_dir
from acumen.prompts import draft_prompt, feedback_block
from acumen.runner import StderrFilter
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
    key = RunKey(arm="skill_v2", split="test", model="claude-opus-4-8", task_id="tf_activity", rep=3)
    directory = run_dir(tmp_path, key)

    assert directory == tmp_path / "skill_v2/test/claude-opus-4-8/tf_activity/rep_3"
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
    assert not cfg.is_local
    assert derive_skill_name("git@github.com:scverse/scanpy.git") == "scanpy"


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

    # session keeps only the subscription token; api keeps only the API key.
    session_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="session")
    assert "ANTHROPIC_API_KEY" not in session_env
    assert session_env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"

    api_env = scrubbed_env(config_dir=tmp_path / "cfg", home=home, auth_mode="api")
    assert api_env["ANTHROPIC_API_KEY"] == "sk-test"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in api_env

    # No mode leaves both credentials in place (the historical behavior).
    both = scrubbed_env(config_dir=tmp_path / "cfg", home=home)
    assert both["ANTHROPIC_API_KEY"] == "sk-test"
    assert both["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"


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
