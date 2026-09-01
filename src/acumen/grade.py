"""Exact-match grader.

Grading is exact string comparison of ``answer.md`` after :meth:`str.strip`, case
sensitive. Nothing is normalized away before comparing — a run that would only pass
under normalization is still a failure, but it is recorded as ``format_error`` rather
than ``wrong_answer`` so the report can show format noise separately from genuinely
wrong answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acumen.paths import ANSWER_FILE

Reason = Literal[
    "ok",
    "wrong_answer",
    "format_error",
    "no_answer_file",
    "budget",
    "max_turns",
    "error",
    "provider_exhausted",
    "sandbox_blocked",
]

#: Reasons that mark a run as a harness failure rather than evidence about a model. A run
#: with one of these is recorded for diagnosis but must never reach a report, an improve
#: pass, or a resume as though it measured something.
INVALID_REASONS: frozenset[Reason] = frozenset({"provider_exhausted", "sandbox_blocked"})

_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(?P<body>.*?)\n?\s*```\s*$", re.DOTALL)
_HEADER_LINE_RE = re.compile(r"^\s*#+\s")
_LABEL_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:=]\s*|^(?:the\s+)?(?:final\s+)?answer\s+is\s+",
    re.IGNORECASE,
)
_WRAPPERS = ("**", "__", "`", "*", "_")


@dataclass(frozen=True)
class Grade:
    """The outcome of grading one run."""

    success: bool
    reason: Reason
    answer: str | None


def _loosen(text: str) -> str:
    """Strip the formatting the harness prompt forbids, for format-noise detection only.

    This never makes a run pass — it only tells ``wrong_answer`` apart from an answer
    whose content was right but whose formatting broke the exact match.
    """
    out = text.strip()
    fenced = _FENCE_RE.match(out)
    if fenced:
        out = fenced.group("body").strip()

    lines = [line for line in out.splitlines() if line.strip()]
    # A markdown header above the answer ("# Answer\nSPI1") is formatting, not content.
    body = [line for line in lines if not _HEADER_LINE_RE.match(line)] or lines
    if len(body) == 1:
        out = body[0].strip()

    out = _LABEL_RE.sub("", out).strip()

    # Emphasis and quoting can nest — `**"SPI1"**` — so peel until stable.
    while True:
        before = out
        for pair in _WRAPPERS:
            if len(out) > 2 * len(pair) and out.startswith(pair) and out.endswith(pair):
                out = out[len(pair) : -len(pair)].strip()
        if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
            out = out[1:-1].strip()
        if out == before:
            break

    return out.rstrip(".").strip()


def grade_answer(answer: str | None, expected: str) -> Grade:
    """Grade a raw ``answer.md`` body against the expected answer.

    Parameters
    ----------
    answer
        The file's contents, or ``None`` if the agent never wrote it.
    expected
        The ground-truth answer from ``tasks.yaml``.

    Returns
    -------
    The grade, whose ``reason`` distinguishes format noise from a wrong answer.
    """
    if answer is None:
        return Grade(success=False, reason="no_answer_file", answer=None)
    stripped = answer.strip()
    want = expected.strip()
    if stripped == want:
        return Grade(success=True, reason="ok", answer=stripped)
    if stripped and _loosen(stripped) == _loosen(want):
        return Grade(success=False, reason="format_error", answer=stripped)
    return Grade(success=False, reason="wrong_answer", answer=stripped)


def grade_run(run_dir: Path, expected: str) -> Grade:
    """Grade a run directory by reading its ``answer.md``."""
    path = run_dir / ANSWER_FILE
    try:
        answer = path.read_text()
    except OSError:
        return Grade(success=False, reason="no_answer_file", answer=None)
    return grade_answer(answer, expected)
