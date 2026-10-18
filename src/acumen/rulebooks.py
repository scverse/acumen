"""The rulebook artifact — the versioned SKILL-generating instructions being optimized.

A *rulebook* is the template text the drafting agent's prompt is built from. The built-in
:data:`acumen.prompts.DRAFT_PROMPT` is one such template; here it becomes ``rulebooks/vN/rulebook.md``
— the same text, the same ``{package}``/``{src}``/``{out}``/``{skill_name}``/… placeholders — so the
outer loop can version it, mutate it from evidence, and draft skills from each version. The rulebook,
not the skill, is what the loop improves; skills are intermediates drafted fresh from whichever
rulebook version is on trial.

A rulebook version is an artifact in exactly the sense a skill version is (:mod:`acumen.skills`):

* **content-hashed** — ``sha256`` over the content file, so two versions with the same text have the
  same hash and a score can be attributed to *this* text, not to a directory name;
* **immutable** — a version directory is written once and never overwritten; the loop always writes
  the next version. Loading re-hashes the content and compares it to the hash recorded at write
  time, so a rulebook edited in place after the fact is detected rather than silently scored as if
  it were the version its name claims;
* **provenanced** — ``meta.json`` records version, parent, rationale, hash, and any maintainer
  feedback. The format is *shared with skills* on purpose (the same :func:`acumen.skills.write_meta`
  writes both): one provenance schema, one reader, one chain to follow from a shipped skill back
  through the rulebook that drafted it and the rulebook that came before.

``meta.json`` is bookkeeping, not content: it is excluded from the hash (it contains the hash) and
from what a drafting agent ever sees (only the template text reaches the prompt).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from acumen.prompts import DRAFT_PROMPT
from acumen.skills import SkillError, SkillMeta, read_meta, skill_hash, version_name, version_number, write_meta

#: The single content file each rulebook version directory holds.
RULEBOOK_FILE = "rulebook.md"

#: Placeholders a rulebook template MUST keep, or the skill it drafts is unusable: ``{out}`` tells
#: the agent where to write the skill, ``{skill_name}`` the name its frontmatter must declare. The
#: outer-improve agent edits the template, so these are the ones worth guarding structurally rather
#: than trusting the prompt to preserve.
REQUIRED_PLACEHOLDERS = ("{out}", "{skill_name}")

#: The fields :func:`acumen.prompts.draft_prompt` fills a rulebook template with — used to check a
#: template ``.format()``s cleanly (no stray ``{placeholder}`` the drafter can't supply).
_TEMPLATE_FIELDS = ("package", "version", "src", "python", "out", "skill_name", "feedback")

#: The rationale recorded on the seeded baseline, so the chain starts with a stated origin.
SEED_RATIONALE = "seeded verbatim from the built-in DRAFT_PROMPT — the baseline the loop optimizes away from"


class RulebookError(ValueError):
    """Raised when a rulebook directory or template is missing, malformed, or tampered with."""


@dataclass(frozen=True)
class Rulebook:
    """A loaded, validated rulebook version."""

    version: str
    directory: Path
    text: str
    hash: str

    @property
    def number(self) -> int:
        """The numeric version, e.g. ``1`` for ``"v1"``."""
        return version_number(self.version)

    @property
    def path(self) -> Path:
        """The content file, ``<directory>/rulebook.md``."""
        return self.directory / RULEBOOK_FILE


def rulebook_dir(rulebooks_root: Path, version: str | int) -> Path:
    """Return the directory holding a rulebook version."""
    name = version_name(version) if isinstance(version, int) else version
    version_number(name)  # validate the shape
    return rulebooks_root / name


def diff_path(rulebooks_root: Path, version: str, parent: str) -> Path:
    """Where a version's diff against its parent is written: ``<root>/diffs/<version>-from-<parent>.diff``.

    Deliberately *outside* the version directory. A version dir is hashed over every content file
    when its ``meta.json`` is written, so anything dropped in afterwards — a diff, a note — makes the
    version read as tampered the next time it is loaded. The first multi-iteration loop tripped on
    exactly that, one iteration in.
    """
    directory = rulebooks_root / "diffs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{version}-from-{parent}.diff"


def rulebook_hash(directory: Path) -> str:
    """Hash a rulebook version's content — ``"sha256:<hex>"``.

    The same length-prefixed path+bytes digest as :func:`acumen.skills.skill_hash`, over the same
    notion of "content files" (everything but ``meta.json``), so the two artifact kinds are hashed
    by one definition.
    """
    try:
        return skill_hash(directory)
    except SkillError as err:
        raise RulebookError(str(err)) from err


def available_versions(rulebooks_root: Path) -> list[str]:
    """Return the rulebook versions present, in numeric order."""
    if not rulebooks_root.is_dir():
        return []
    found = [
        p.name for p in rulebooks_root.iterdir() if p.is_dir() and (p / RULEBOOK_FILE).is_file() and _is_version(p.name)
    ]
    return sorted(found, key=version_number)


def _is_version(name: str) -> bool:
    try:
        version_number(name)
    except ValueError:
        return False
    return True


def latest_version(rulebooks_root: Path) -> str | None:
    """Return the highest rulebook version present, or ``None`` if there are none."""
    versions = available_versions(rulebooks_root)
    return versions[-1] if versions else None


def next_version(rulebooks_root: Path) -> str:
    """Return the version a new rulebook should be written as."""
    latest = latest_version(rulebooks_root)
    return version_name(version_number(latest) + 1 if latest else 1)


def validate_rulebook(text: str) -> None:
    """Fail loudly if a rulebook template could not produce a working draft prompt.

    Two checks, because the outer-improve agent rewrites this text and either mistake silently
    breaks the next draft: (1) it must ``.format()`` with exactly the fields the drafter supplies —
    a stray ``{something}`` raises ``KeyError`` at draft time deep in the SDK; (2) it must keep the
    load-bearing placeholders (:data:`REQUIRED_PLACEHOLDERS`), without which the drafted skill has
    nowhere to go or no name.

    Raises
    ------
    RulebookError
        If the template references an unknown field or dropped a required placeholder.
    """
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in text]
    if missing:
        raise RulebookError(
            f"rulebook is missing required placeholder(s) {missing} — a draft prompt built from it "
            "would not tell the agent where to write the skill or what to name it"
        )
    try:
        text.format(**dict.fromkeys(_TEMPLATE_FIELDS, "x"))
    except (KeyError, IndexError, ValueError) as err:
        raise RulebookError(
            f"rulebook has a placeholder the drafter cannot fill ({err}) — templates may use only "
            f"{{{', '.join(_TEMPLATE_FIELDS)}}}; escape any literal brace as {{{{ or }}}}"
        ) from err


def rulebook_meta(rulebooks_root: Path, version: str | int) -> SkillMeta | None:
    """Read a rulebook version's ``meta.json`` provenance, or ``None`` if it has none.

    A version without ``meta.json`` is a pre-provenance (prototype-era) directory; it still loads,
    but nothing can be verified or chained for it.
    """
    try:
        return read_meta(rulebook_dir(rulebooks_root, version))
    except SkillError as err:
        raise RulebookError(str(err)) from err


def load_rulebook(rulebooks_root: Path, version: str | int) -> Rulebook:
    """Load and validate one rulebook version, verifying its content against ``meta.json``.

    Immutability is enforced on the way *in* as well as on the way out (:func:`write_rulebook`
    refuses to overwrite): the content is re-hashed and, when a ``meta.json`` exists, compared with
    the hash recorded at write time. A mismatch means the file was edited in place — the version's
    name no longer identifies its text, and any score recorded against it would be misattributed —
    so it is an error, not a warning.

    Raises
    ------
    RulebookError
        If the version is missing, its template is invalid, or its content no longer matches its
        recorded hash.
    """
    directory = rulebook_dir(rulebooks_root, version)
    path = directory / RULEBOOK_FILE
    if not path.is_file():
        raise RulebookError(f"no such rulebook version: {path}")
    try:
        text = path.read_text()
    except OSError as err:
        raise RulebookError(f"cannot read {path}: {err}") from err
    validate_rulebook(text)
    digest = rulebook_hash(directory)
    meta = rulebook_meta(rulebooks_root, version)
    if meta is not None and meta.hash and meta.hash != digest:
        raise RulebookError(
            f"{path} has been modified since it was written (recorded {meta.hash}, now {digest}) — "
            "rulebook versions are immutable; write a new version instead of editing one in place"
        )
    return Rulebook(version=directory.name, directory=directory, text=text, hash=digest)


def write_rulebook(
    rulebooks_root: Path,
    version: str | int,
    text: str,
    *,
    parent: str | None,
    rationale: str,
    feedback: str | None = None,
) -> Rulebook:
    """Write a new, immutable rulebook version with its provenance.

    Refuses if the version directory exists at all (not merely the content file): a half-written
    version is as much a collision as a complete one. The template is validated before anything
    touches disk, so an invalid rulebook never becomes a version. ``meta.json`` is written last,
    hashing the content as it stands — the same ordering ``improve`` uses for skills.

    Parameters
    ----------
    parent
        The version this one was derived from, or ``None`` for a seeded baseline.
    rationale
        Why this version differs from its parent — the improve agent's stated reasoning.
    feedback
        Maintainer ``--feedback`` that shaped this version, if any.
    """
    validate_rulebook(text)
    directory = rulebook_dir(rulebooks_root, version)
    if directory.exists():
        raise RulebookError(f"{directory} already exists — rulebook versions are immutable and never overwritten")
    directory.mkdir(parents=True)
    (directory / RULEBOOK_FILE).write_text(text)
    try:
        meta = write_meta(directory, parent=parent, rationale=rationale, feedback=feedback)
    except SkillError as err:
        raise RulebookError(str(err)) from err
    return Rulebook(version=directory.name, directory=directory, text=text, hash=meta.hash)


def seed_default(rulebooks_root: Path) -> str:
    """Ensure ``rulebooks/v1`` exists, seeded from the built-in draft prompt; always return ``"v1"``.

    v1 is the current hardcoded :data:`acumen.prompts.DRAFT_PROMPT` verbatim, so the loop's first
    iteration reproduces exactly today's drafting behaviour — the baseline the rulebook is optimized
    away from. Idempotent: if v1 is already present it is left untouched. This returns the *baseline*
    (``"v1"``), not the latest version, so a resumed loop still anchors on the same baseline it
    started from rather than treating an already-improved version as the starting point.
    """
    if "v1" not in available_versions(rulebooks_root):
        write_rulebook(rulebooks_root, "v1", DRAFT_PROMPT, parent=None, rationale=SEED_RATIONALE)
    return "v1"
