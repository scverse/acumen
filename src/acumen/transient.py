"""Transient platform failures: the platform's problem, not the run's.

A rate limit, an overloaded API, a subscription session limit or a dropped network says nothing
about a task or a skill. The benchmark side already records such runs as invalid and leaves the
cell pending (:data:`acumen.grade.INVALID_REASONS`). This module covers the same ground for the
*meta* agents (draft, improve, pollinate) and the long-running loops, which must pause rather than
treat the failure as a bad candidate — and exit with a distinct code so an unattended runner knows
to wait and rerun the identical command.
"""

from __future__ import annotations

#: Substrings of an agent error that mark it as the platform's problem. Matched case-insensitively.
#: A dropped connection is as transient as a rate limit and just as silent: one overnight draft
#: recorded 14 ENOTFOUND runs as task failures, which put that arm 9 tasks below its siblings and
#: made a fine skill look like a bad one.
TRANSIENT_MARKERS = (
    "session limit",
    "usage limit",
    "rate limit",
    "rate_limit",
    "overloaded",
    " 429",
    " 529",
    "enotfound",
    "econnreset",
    "econnrefused",
    "etimedout",
    "eai_again",
    "socket hang up",
    "can't reach the api server",
    "connection error",
    " 502",
    " 503",
    " 504",
)


def is_transient(error: str) -> bool:
    """Whether an agent error is a transient platform failure that a later retry would not hit."""
    lower = f" {error.lower()}"
    return any(marker in lower for marker in TRANSIENT_MARKERS)


class TransientLimitError(RuntimeError):
    """A pass stopped because the platform refused runs (limit, overload or network drop).

    Raised after the pass has wound down, so a caller stops instead of proceeding on a partial
    result. Nothing was recorded for the refused work; rerunning the identical command resumes
    exactly there.
    """
