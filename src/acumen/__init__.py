"""Build, benchmark, and optimize Claude skills for Python packages."""

from importlib.metadata import version

from acumen.bench import PlannedRun, build_matrix, pending, run_matrix, summarize
from acumen.config import Config, ConfigError, load_config, parse_config
from acumen.draft import DraftError, DraftResult, draft_skill
from acumen.env import EnvError, Target, prepare_target, scrubbed_env
from acumen.grade import Grade, Reason, grade_answer, grade_run
from acumen.paths import RunKey, Split, arm_name, is_complete, parse_run_dir, run_dir, skill_from_arm
from acumen.report import Report, ReportError, arm_metrics, build_report, load_results
from acumen.runner import RunOutcome, run_once
from acumen.sandbox import Sandbox, install_skill, sandbox
from acumen.skills import (
    Skill,
    SkillError,
    SkillMeta,
    available_versions,
    latest_version,
    load_skill,
    next_version,
    skill_hash,
)
from acumen.tasks import Task, TaskError, TaskSplit, load_tasks, parse_tasks

__all__ = [
    "Config",
    "ConfigError",
    "DraftError",
    "DraftResult",
    "EnvError",
    "Grade",
    "PlannedRun",
    "Reason",
    "Report",
    "ReportError",
    "RunKey",
    "RunOutcome",
    "Sandbox",
    "Skill",
    "SkillError",
    "SkillMeta",
    "Split",
    "Target",
    "Task",
    "TaskError",
    "TaskSplit",
    "__version__",
    "arm_metrics",
    "arm_name",
    "available_versions",
    "build_matrix",
    "build_report",
    "draft_skill",
    "grade_answer",
    "grade_run",
    "install_skill",
    "is_complete",
    "latest_version",
    "load_config",
    "load_results",
    "load_skill",
    "load_tasks",
    "next_version",
    "parse_config",
    "parse_run_dir",
    "parse_tasks",
    "pending",
    "prepare_target",
    "run_dir",
    "run_matrix",
    "run_once",
    "sandbox",
    "scrubbed_env",
    "skill_from_arm",
    "skill_hash",
    "summarize",
]

__version__ = version("acumen")
