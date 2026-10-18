"""Skill directories: loading, validation, and a stable content hash.

A skill is a directory — ``SKILL.md`` plus an optional ``references/`` tree — and
versions are immutable: ``improve`` writes ``skills/v{n+1}/``, never mutates ``v{n}``.

``meta.json`` is acumen's own bookkeeping (version, parent, rationale, hash), not part of
the skill. It is therefore excluded from :func:`skill_hash` — it *contains* the hash, so
including it would be circular — and excluded from what gets copied to an agent, which
has no business reading the rationale for its own skill.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SKILL_FILE = "SKILL.md"
META_FILE = "meta.json"
REFERENCES_DIR = "references"

#: Files that are acumen bookkeeping rather than skill content.
_NOT_CONTENT = {META_FILE}

_VERSION_RE = re.compile(r"^v(\d+)$")
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(?P<body>.*?)\r?\n---\s*\r?\n?", re.DOTALL)


class SkillError(ValueError):
    """Raised when a skill directory is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class SkillMeta:
    """Provenance for one skill version, as stored in ``meta.json``."""

    version: int
    parent: str | None
    rationale: str
    hash: str
    #: Maintainer ``--feedback`` that shaped this version, or ``None`` if none was given.
    feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the ``meta.json`` payload.

        ``feedback`` is omitted when absent, so a version drafted/improved without ``--feedback``
        writes the same ``meta.json`` it did without the field.
        """
        payload: dict[str, Any] = {
            "version": self.version,
            "parent": self.parent,
            "rationale": self.rationale,
            "hash": self.hash,
        }
        if self.feedback:
            payload["feedback"] = self.feedback
        return payload


@dataclass(frozen=True)
class Skill:
    """A loaded, validated skill version."""

    version: str
    directory: Path
    name: str
    description: str
    hash: str
    #: Bytes of skill content (``SKILL.md`` + references) — the leanness axis. Recorded in every
    #: ``result.json`` as ``skill_bytes`` so the report can trade size against success without
    #: needing the skill tree at hand. ``0`` is what "no skill" weighs.
    size: int = 0

    @property
    def number(self) -> int:
        """The numeric version, e.g. ``1`` for ``"v1"``."""
        return version_number(self.version)


def version_name(number: int) -> str:
    """Return the directory name for a numeric version, e.g. ``1`` -> ``"v1"``."""
    if number < 1:
        raise SkillError(f"skill version must be >= 1, got {number}")
    return f"v{number}"


def version_number(version: str) -> int:
    """Parse a version directory name, e.g. ``"v1"`` -> ``1``."""
    match = _VERSION_RE.match(version)
    if not match:
        raise SkillError(f"skill version must look like 'v1', got {version!r}")
    number = int(match.group(1))
    if number < 1:
        raise SkillError(f"skill version must be >= 1, got {version!r}")
    return number


def skill_dir(skills_root: Path, version: str | int) -> Path:
    """Return the directory holding a skill version."""
    name = version_name(version) if isinstance(version, int) else version
    version_number(name)  # validate
    return skills_root / name


def content_files(directory: Path) -> list[Path]:
    """Return the skill's content files, sorted, excluding acumen bookkeeping.

    Sorting by POSIX-relative path keeps the order stable across filesystems, which is
    what makes :func:`skill_hash` reproducible.
    """
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and not (path.parent == directory and path.name in _NOT_CONTENT)
    ]
    return sorted(files, key=lambda p: p.relative_to(directory).as_posix())


def skill_content(directory: Path) -> dict[str, str]:
    """Return the skill's text content as ``{posix-relpath: text}``, excluding bookkeeping.

    Used by the report to diff one version against its parent. Files that do not decode as
    UTF-8 (there should be none in a skill) are represented by a short placeholder so a diff
    still lists them without choking on bytes.

    Returns
    -------
    A mapping from each content file's POSIX-relative path to its decoded text.
    """
    content: dict[str, str] = {}
    for path in content_files(directory):
        rel = path.relative_to(directory).as_posix()
        try:
            content[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content[rel] = "<binary file>\n"
    return content


def skill_size(directory: Path) -> int:
    """Total bytes of a skill's content files — what an agent has to read to use it.

    This is the *leanness* measure the report trades against success. Bytes rather than tokens,
    because bytes are exact, model-independent, and need no tokenizer; and content files only, for
    the same reason as :func:`skill_hash` — ``meta.json`` never reaches an agent.
    """
    if not directory.is_dir():
        raise SkillError(f"skill directory does not exist: {directory}")
    return sum(path.stat().st_size for path in content_files(directory))


def skill_hash(directory: Path) -> str:
    """Hash a skill directory's content — the value recorded in every ``result.json``.

    Both the relative path and the bytes of each file are folded in, with lengths, so
    that renaming a file or moving text between files changes the hash. Byte content is
    used rather than text, so a line-ending change is a real change.

    Returns
    -------
    ``"sha256:<hex>"``.
    """
    if not directory.is_dir():
        raise SkillError(f"skill directory does not exist: {directory}")
    digest = hashlib.sha256()
    for path in content_files(directory):
        rel = path.relative_to(directory).as_posix().encode()
        data = path.read_bytes()
        # Length-prefix both parts so no concatenation of names and bodies can collide.
        digest.update(f"{len(rel)}:".encode())
        digest.update(rel)
        digest.update(f"{len(data)}:".encode())
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the YAML frontmatter of a ``SKILL.md``.

    Returns
    -------
    The frontmatter mapping; empty if the file has none.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group("body"))
    except yaml.YAMLError as err:
        raise SkillError(f"SKILL.md frontmatter is not valid YAML: {err}") from err
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SkillError(f"SKILL.md frontmatter must be a mapping, got {type(data).__name__}")
    return data


def normalize_frontmatter(text: str) -> str:
    r"""Quote frontmatter scalars YAML would misread, leaving valid frontmatter untouched.

    A drafting agent writes ``description: Analyze data (Visium: yes) — use when …`` and YAML
    rejects the second colon ("mapping values are not allowed here"). The content is fine; only the
    quoting is missing, and the frontmatter is *our* format constraint, so repairing it is a
    normalization rather than an edit of the agent's work: every unquoted top-level ``key: value``
    line is rewritten with the value double-quoted (escaping ``\\`` and ``"``). Applied only when
    the frontmatter fails to parse, and returns ``text`` unchanged if the repair does not parse
    either — the validator then reports the real problem.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    body = match.group("body")
    try:
        yaml.safe_load(body)
        return text
    except yaml.YAMLError:
        pass
    fixed_lines = []
    for line in body.splitlines():
        key, sep, value = line.partition(":")
        value = value.strip()
        if sep and key and not key[0].isspace() and value and value[0] not in "\"'[{|>":
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            fixed_lines.append(f'{key}: "{escaped}"')
        else:
            fixed_lines.append(line)
    fixed_body = "\n".join(fixed_lines)
    try:
        yaml.safe_load(fixed_body)
    except yaml.YAMLError:
        return text
    return text[: match.start("body")] + fixed_body + text[match.end("body") :]


def load_skill(skills_root: Path, version: str | int, *, expect_name: str | None = None) -> Skill:
    """Load and validate one skill version.

    Parameters
    ----------
    skills_root
        The ``skills/`` root directory.
    version
        ``"v1"`` or ``1``.
    expect_name
        If given, the skill's frontmatter ``name`` must match it — this is
        ``config.skill_name``, and a mismatch means the skill would be injected under a
        directory name the agent's own frontmatter disagrees with.

    Returns
    -------
    The loaded skill, with its content hash computed.
    """
    directory = skill_dir(skills_root, version)
    if not directory.is_dir():
        raise SkillError(f"no such skill version: {directory}")
    skill_md = directory / SKILL_FILE
    if not skill_md.is_file():
        raise SkillError(f"{directory} has no {SKILL_FILE} — a skill is a directory containing one")
    try:
        text = skill_md.read_text()
    except OSError as err:
        raise SkillError(f"cannot read {skill_md}: {err}") from err

    front = parse_frontmatter(text)
    name = front.get("name")
    description = front.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillError(f"{skill_md} frontmatter is missing a non-empty 'name'")
    if not isinstance(description, str) or not description.strip():
        raise SkillError(
            f"{skill_md} frontmatter is missing a non-empty 'description' — "
            "the description is how the agent decides whether to load the skill at all"
        )
    if expect_name is not None and name.strip() != expect_name:
        raise SkillError(
            f"{skill_md} declares name {name.strip()!r} but config.skill_name is {expect_name!r} — "
            "they must match, or the skill is injected under a name its own frontmatter disagrees with"
        )

    return Skill(
        version=directory.name,
        directory=directory,
        name=name.strip(),
        description=description.strip(),
        hash=skill_hash(directory),
        size=skill_size(directory),
    )


def available_versions(skills_root: Path) -> list[str]:
    """Return the skill versions present, in numeric order."""
    if not skills_root.is_dir():
        return []
    found = [p.name for p in skills_root.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
    return sorted(found, key=version_number)


def latest_version(skills_root: Path) -> str | None:
    """Return the highest skill version present, or ``None`` if there are none."""
    versions = available_versions(skills_root)
    return versions[-1] if versions else None


def next_version(skills_root: Path) -> str:
    """Return the version a new skill should be written as."""
    latest = latest_version(skills_root)
    return version_name(version_number(latest) + 1 if latest else 1)


def read_meta(directory: Path) -> SkillMeta | None:
    """Read a skill version's ``meta.json``, or ``None`` if it has none."""
    path = directory / META_FILE
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        raise SkillError(f"cannot read {path}: {err}") from err
    feedback = raw.get("feedback")
    return SkillMeta(
        version=int(raw.get("version", 0)),
        parent=raw.get("parent"),
        rationale=str(raw.get("rationale", "")),
        hash=str(raw.get("hash", "")),
        feedback=str(feedback) if feedback else None,
    )


def write_meta(directory: Path, *, parent: str | None, rationale: str, feedback: str | None = None) -> SkillMeta:
    """Write ``meta.json`` for a skill version, hashing its content as it stands.

    Call this only once the skill's content files are final — the hash is computed here.

    ``feedback`` records the maintainer ``--feedback`` that shaped this version, for the
    report's skill-versions provenance; it is omitted from ``meta.json`` when empty.
    """
    text = (feedback or "").strip()
    meta = SkillMeta(
        version=version_number(directory.name),
        parent=parent,
        rationale=rationale.strip(),
        hash=skill_hash(directory),
        feedback=text or None,
    )
    (directory / META_FILE).write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
    return meta
