"""Tests for the full rulebook artifact — content hash, immutability, provenance chain, tamper detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from acumen import rulebooks as rb
from acumen.prompts import DRAFT_PROMPT
from acumen.rulebooks import RulebookError, seed_default
from acumen.skills import META_FILE

TEMPLATE_A = "Write the skill to {out} named {skill_name}.\n"
TEMPLATE_B = "Write the skill to {out} named {skill_name}. Be terse.\n"


def test_write_records_provenance_and_hash(tmp_path: Path) -> None:
    v1 = rb.write_rulebook(tmp_path, "v1", TEMPLATE_A, parent=None, rationale="seed")
    v2 = rb.write_rulebook(tmp_path, "v2", TEMPLATE_B, parent="v1", rationale="tightened", feedback="be terse")

    assert v1.hash.startswith("sha256:") and v1.hash != v2.hash
    meta1 = rb.rulebook_meta(tmp_path, "v1")
    meta2 = rb.rulebook_meta(tmp_path, "v2")
    assert meta1 is not None and meta1.parent is None and meta1.rationale == "seed" and meta1.hash == v1.hash
    assert (
        meta2 is not None and meta2.parent == "v1" and meta2.rationale == "tightened" and meta2.feedback == "be terse"
    )
    # The chain is followable from the newest version back to the seed.
    assert meta2.parent == v1.version and meta1.parent is None
    # meta.json is bookkeeping, not content: it does not perturb the hash.
    assert rb.rulebook_hash(v2.directory) == v2.hash
    assert (v2.directory / META_FILE).is_file()


def test_same_text_same_hash(tmp_path: Path) -> None:
    a = rb.write_rulebook(tmp_path / "a", "v1", TEMPLATE_A, parent=None, rationale="x")
    b = rb.write_rulebook(tmp_path / "b", "v3", TEMPLATE_A, parent=None, rationale="y")
    assert a.hash == b.hash  # a score attaches to the text, not the directory name


def test_load_verifies_content_against_recorded_hash(tmp_path: Path) -> None:
    v1 = rb.write_rulebook(tmp_path, "v1", TEMPLATE_A, parent=None, rationale="seed")
    assert rb.load_rulebook(tmp_path, "v1") == v1
    # Editing in place is detected: the version's name no longer identifies its text.
    v1.path.write_text(TEMPLATE_B)
    with pytest.raises(RulebookError, match="modified since it was written"):
        rb.load_rulebook(tmp_path, "v1")


def test_load_without_meta_is_allowed(tmp_path: Path) -> None:
    # A prototype-era version (no meta.json) still loads — nothing to verify against.
    d = tmp_path / "v1"
    d.mkdir()
    (d / rb.RULEBOOK_FILE).write_text(TEMPLATE_A)
    loaded = rb.load_rulebook(tmp_path, "v1")
    assert loaded.text == TEMPLATE_A and rb.rulebook_meta(tmp_path, "v1") is None


def test_write_refuses_existing_directory_even_without_content(tmp_path: Path) -> None:
    (tmp_path / "v1").mkdir()  # a half-written version is still a collision
    with pytest.raises(RulebookError, match="immutable"):
        rb.write_rulebook(tmp_path, "v1", TEMPLATE_A, parent=None, rationale="x")


def test_write_validates_before_touching_disk(tmp_path: Path) -> None:
    with pytest.raises(RulebookError, match="required placeholder"):
        rb.write_rulebook(tmp_path, "v1", "no placeholders", parent=None, rationale="x")
    assert not (tmp_path / "v1").exists()  # an invalid rulebook never becomes a version


def test_seed_default_writes_provenance(tmp_path: Path) -> None:
    assert seed_default(tmp_path) == "v1"
    meta = rb.rulebook_meta(tmp_path, "v1")
    assert meta is not None and meta.parent is None and "DRAFT_PROMPT" in meta.rationale
    assert rb.load_rulebook(tmp_path, "v1").text == DRAFT_PROMPT
    raw = json.loads((tmp_path / "v1" / META_FILE).read_text())
    assert raw["version"] == 1 and raw["hash"] == meta.hash


def test_versions_and_next(tmp_path: Path) -> None:
    assert rb.available_versions(tmp_path) == [] and rb.latest_version(tmp_path) is None
    assert rb.next_version(tmp_path) == "v1"
    rb.write_rulebook(tmp_path, "v1", TEMPLATE_A, parent=None, rationale="x")
    rb.write_rulebook(tmp_path, "v2", TEMPLATE_B, parent="v1", rationale="y")
    (tmp_path / "v9").mkdir()  # a directory with no rulebook.md is not a version
    assert rb.available_versions(tmp_path) == ["v1", "v2"]
    assert rb.next_version(tmp_path) == "v3"
    assert rb.load_rulebook(tmp_path, 2).number == 2


def test_diff_lives_outside_the_version_dir_so_the_hash_holds(tmp_path: Path) -> None:
    v2 = rb.write_rulebook(tmp_path, "v2", TEMPLATE_B, parent="v1", rationale="x")
    path = rb.diff_path(tmp_path, "v2", "v1")
    path.write_text("--- v1\n+++ v2\n")
    assert path.parent == tmp_path / "diffs" and not path.is_relative_to(v2.directory)
    assert rb.load_rulebook(tmp_path, "v2") == v2  # still verifies against its recorded hash
    # The old behaviour — a file dropped into the version dir — is exactly what the check catches.
    (v2.directory / "from-v1.diff").write_text("x")
    with pytest.raises(RulebookError, match="modified since"):
        rb.load_rulebook(tmp_path, "v2")


def test_normalize_frontmatter_quotes_a_colon_in_the_description() -> None:
    from acumen.skills import normalize_frontmatter, parse_frontmatter

    raw = "---\nname: squidpy\ndescription: Analyze spatial data (Visium: yes, Xenium: yes) — use when asked\n---\n\n# body\n"
    fixed = normalize_frontmatter(raw)
    front = parse_frontmatter(fixed)
    assert front["name"] == "squidpy"
    assert front["description"] == "Analyze spatial data (Visium: yes, Xenium: yes) — use when asked"
    assert fixed.endswith("---\n\n# body\n")  # only the frontmatter block changed
    # Valid frontmatter is returned byte-identical.
    ok = '---\nname: x\ndescription: "already: quoted"\n---\nbody\n'
    assert normalize_frontmatter(ok) == ok
    # A file with no frontmatter is left alone (the validator reports it).
    assert normalize_frontmatter("# no frontmatter\n") == "# no frontmatter\n"
