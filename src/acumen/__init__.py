"""Build, benchmark, and optimize Claude skills for Python packages."""

from importlib.metadata import version

from acumen.bench import PlannedRun, build_matrix, pending, run_matrix, summarize
from acumen.config import Config, ConfigError, load_config, parse_config
from acumen.draft import DraftError, DraftResult, draft_skill
from acumen.env import EnvError, Target, auth_available, check_auth, prepare_target, scrubbed_env
from acumen.grade import Grade, Reason, grade_answer, grade_run
from acumen.improve import (
    ImproveError,
    ImproveResult,
    TrainRun,
    collect_train_runs,
    find_test_access,
    improve_skill,
    make_test_guard,
)
from acumen.logs import LiveLog
from acumen.paths import RunKey, Split, arm_name, is_complete, parse_run_dir, run_dir, skill_from_arm
from acumen.report import Report, ReportError, arm_metrics, build_report, load_results
from acumen.runner import RunOutcome, run_once
from acumen.sandbox import Sandbox, install_skill, sandbox
from acumen.scaffold import InitError, scaffold
from acumen.ship import ShipError, ShipResult, installer_exists, ship_skill
from acumen.skills import (
    Skill,
    SkillError,
    SkillMeta,
    available_versions,
    latest_version,
    load_skill,
    next_version,
    skill_content,
    skill_hash,
)
from acumen.taskgen import (
    TaskGenError,
    TaskGenResult,
    build_filtered_source,
    dump_tasks,
    find_skill_access,
    generate_tasks,
    make_skill_guard,
)
from acumen.tasks import Task, TaskError, TaskSplit, load_tasks, parse_tasks
from acumen.transcript import locate_transcript, render_transcript

__all__ = [
    "Config",
    "ConfigError",
    "DraftError",
    "DraftResult",
    "EnvError",
    "Grade",
    "ImproveError",
    "ImproveResult",
    "InitError",
    "LiveLog",
    "PlannedRun",
    "Reason",
    "Report",
    "ReportError",
    "RunKey",
    "RunOutcome",
    "Sandbox",
    "ShipError",
    "ShipResult",
    "Skill",
    "SkillError",
    "SkillMeta",
    "Split",
    "Target",
    "Task",
    "TaskError",
    "TaskGenError",
    "TaskGenResult",
    "TaskSplit",
    "TrainRun",
    "__version__",
    "arm_metrics",
    "arm_name",
    "auth_available",
    "available_versions",
    "check_auth",
    "build_filtered_source",
    "build_matrix",
    "build_report",
    "collect_train_runs",
    "draft_skill",
    "dump_tasks",
    "find_skill_access",
    "find_test_access",
    "generate_tasks",
    "grade_answer",
    "grade_run",
    "improve_skill",
    "install_skill",
    "installer_exists",
    "is_complete",
    "make_skill_guard",
    "make_test_guard",
    "latest_version",
    "load_config",
    "locate_transcript",
    "load_results",
    "load_skill",
    "load_tasks",
    "next_version",
    "parse_config",
    "parse_run_dir",
    "parse_tasks",
    "pending",
    "prepare_target",
    "render_transcript",
    "run_dir",
    "run_matrix",
    "run_once",
    "sandbox",
    "scaffold",
    "scrubbed_env",
    "ship_skill",
    "skill_content",
    "skill_from_arm",
    "skill_hash",
    "summarize",
]

__version__ = version("acumen")
