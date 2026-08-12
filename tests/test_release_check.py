"""Golden tests for tools/release_check.py, the offline release-cut gate.

The fixture CHANGELOG below is modeled on the real CHANGELOG.md's actual
shape: an ``## [Unreleased]`` section with four ``###`` category
subsections, a dated version section, an older dated version section, and a
trailing non-version ``## Future work`` section -- so the boundary logic
(a ``###`` subsection heading does NOT end a version section; any ``## ``
heading does, version or not) is exercised the same way it would be against
the real file.

Each gate -- pyproject version match (including near-miss/prefix mutations),
CHANGELOG section/date (including strict-shape date variants fromisoformat
alone would accept, and duplicate-heading detection), notes-length cap, and
Unreleased-above -- gets its own failure-case test, plus a positive case
that pins the extracted notes to an exact string, and a combined case (a
version matching neither pyproject nor any CHANGELOG section) proving
failures are collected rather than reported one at a time. The real-repo
test at the bottom reads pyproject.toml's CURRENT version dynamically rather
than hardcoding one, specifically so it keeps passing across every future
release cut rather than breaking at the first one (see its docstring).
"""
import sys
try:
    import tomllib  # stdlib on Python >= 3.11
except ModuleNotFoundError:  # Python 3.10 (CI matrix floor)
    import tomli as tomllib  # type: ignore[no-redef]

import tools.release_check as release_check

FIXTURE_CHANGELOG = """# Changelog

## [Unreleased]

### Added
- New thing added.

### Changed
- Something changed.

### Removed
- Something removed.

### Fixed
- A bug fixed.

## [1.2.3] - 2026-05-01

### Added
- Feature X shipped in 1.2.3.

### Changed
- Behavior Y changed in 1.2.3.

## [1.2.2] - 2026-01-15 — Old Release

### Fixed
- Old fix.

## Future work

- Some idea not yet scheduled.
"""

EXPECTED_1_2_3_NOTES = (
    "### Added\n"
    "- Feature X shipped in 1.2.3.\n"
    "\n"
    "### Changed\n"
    "- Behavior Y changed in 1.2.3."
)

EXPECTED_1_2_2_NOTES = "### Fixed\n- Old fix."


# ── extract_changelog_section: the CHANGELOG gate ───────────────────────────

def test_extract_section_positive_matches_expected_notes_exactly():
    result = release_check.extract_changelog_section(FIXTURE_CHANGELOG, "1.2.3")
    assert result.errors == []
    assert result.date == "2026-05-01"
    assert result.notes == EXPECTED_1_2_3_NOTES


def test_extract_section_stops_at_any_two_hash_heading_not_just_versions():
    # 1.2.2's body is followed by '## Future work', not another version --
    # the boundary check must be "starts with '## '", not "is a version".
    result = release_check.extract_changelog_section(FIXTURE_CHANGELOG, "1.2.2")
    assert result.errors == []
    assert result.notes == EXPECTED_1_2_2_NOTES


def test_extract_section_missing_section():
    result = release_check.extract_changelog_section(FIXTURE_CHANGELOG, "9.9.9")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "no '## [9.9.9]" in result.errors[0]


def test_extract_section_rejects_malformed_iso_date():
    # 13 is not a real month, 45 is not a real day -- must be rejected even
    # though it matches the YYYY-MM-DD shape.
    text = "## [Unreleased]\n\n## [9.0.0] - 2026-13-45\n\nbody\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "valid ISO 8601 date" in result.errors[0]


def test_extract_section_rejects_iso_week_date_variant():
    # datetime.date.fromisoformat alone accepts ISO week dates since Python
    # 3.11 -- the strict YYYY-MM-DD regex must reject this even though
    # fromisoformat("2026-W01-1") would happily parse it.
    text = "## [Unreleased]\n\n## [9.0.0] - 2026-W01-1\n\nbody\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "valid ISO 8601 date" in result.errors[0]


def test_extract_section_rejects_basic_dashless_date_variant():
    # Same idea for the basic (dashless) ISO form -- fromisoformat accepts
    # "20260609", the strict shape regex must not.
    text = "## [Unreleased]\n\n## [9.0.0] - 20260609\n\nbody\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "valid ISO 8601 date" in result.errors[0]


def test_extract_section_rejects_duplicate_version_section():
    text = (
        "## [Unreleased]\n\n"
        "## [9.0.0] - 2026-05-01\n\nfirst body\n\n"
        "## [9.0.0] - 2026-06-01\n\nsecond body\n"
    )
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert any(
        "more than one" in e and "[9.0.0]" in e for e in result.errors
    ), result.errors


def test_extract_section_rejects_duplicate_unreleased_heading():
    text = (
        "## [Unreleased]\n\n"
        "## [9.0.0] - 2026-05-01\n\nbody\n\n"
        "## [Unreleased]\n"
    )
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert any(
        "more than one" in e and "Unreleased" in e for e in result.errors
    ), result.errors


def test_extract_section_rejects_notes_over_length_cap():
    # GitHub's release-body cap is 125,000 chars; MAX_NOTES_CHARS (120,000)
    # is a safety margin below it. 10,000 x 14-char lines = 140,000 chars,
    # comfortably over.
    huge_body = "- filler line\n" * 10_000
    text = (
        f"## [Unreleased]\n\n"
        f"## [9.0.0] - 2026-05-01\n\n{huge_body}\n"
        f"## [8.0.0] - 2026-01-01\n\nold\n"
    )
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert any(str(release_check.MAX_NOTES_CHARS) in e for e in result.errors)


def test_extract_section_requires_unreleased_strictly_above():
    # '## [Unreleased]' exists in the text, but BELOW the version section --
    # presence alone must not satisfy the gate, position matters.
    text = "## [9.0.0] - 2026-05-01\n\nbody\n\n## [Unreleased]\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "no '## [Unreleased]' heading above" in result.errors[0]


def test_extract_section_missing_unreleased_entirely():
    text = "## [9.0.0] - 2026-05-01\n\nbody\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.notes is None
    assert len(result.errors) == 1
    assert "no '## [Unreleased]' heading above" in result.errors[0]


def test_extract_section_extracts_to_eof_when_no_trailing_heading():
    text = "## [Unreleased]\n\n## [9.0.0] - 2026-05-01\n\nlast section body\n"
    result = release_check.extract_changelog_section(text, "9.0.0")
    assert result.errors == []
    assert result.notes == "last section body"


# ── check_pyproject_version: the version gate ───────────────────────────────

def test_check_pyproject_version_ok(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[project]\nname = "x"\nversion = "1.2.3"\n')
    assert release_check.check_pyproject_version("1.2.3", pp) is None


def test_check_pyproject_version_mismatch_names_both_values(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[project]\nname = "x"\nversion = "1.2.3"\n')
    err = release_check.check_pyproject_version("9.9.9", pp)
    assert err is not None
    assert "1.2.3" in err
    assert "9.9.9" in err


def test_check_pyproject_version_rejects_prefix_near_miss_pyproject_longer(tmp_path):
    # Guards the comparison staying exact-string '!=', not a prefix/startswith
    # check -- '1.2.3' is a prefix of '1.2.30' but they are different releases.
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[project]\nname = "x"\nversion = "1.2.30"\n')
    err = release_check.check_pyproject_version("1.2.3", pp)
    assert err is not None
    assert "1.2.30" in err and "1.2.3" in err


def test_check_pyproject_version_rejects_prefix_near_miss_version_longer(tmp_path):
    # Same guard, mismatch in the other direction (a version-with-suffix
    # given while pyproject has the bare version).
    pp = tmp_path / "pyproject.toml"
    pp.write_text('[project]\nname = "x"\nversion = "1.2.3"\n')
    err = release_check.check_pyproject_version("1.2.3.post1", pp)
    assert err is not None
    assert "1.2.3" in err


# ── run_release_check: the full gate, wired together ────────────────────────

def _write_fixture(tmp_path, pyproject_version="1.2.3", changelog=FIXTURE_CHANGELOG):
    pp = tmp_path / "pyproject.toml"
    pp.write_text(f'[project]\nname = "x"\nversion = "{pyproject_version}"\n')
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(changelog)
    return pp, cl


def test_run_release_check_positive(tmp_path):
    pp, cl = _write_fixture(tmp_path, pyproject_version="1.2.3")
    result = release_check.run_release_check("1.2.3", pp, cl)
    assert result.ok is True
    assert result.notes == EXPECTED_1_2_3_NOTES
    assert result.errors == []


def test_run_release_check_fake_version_lists_every_missing_thing(tmp_path):
    pp, cl = _write_fixture(tmp_path, pyproject_version="1.2.3")
    result = release_check.run_release_check("9.9.9", pp, cl)
    assert result.ok is False
    assert result.notes is None
    # Both the pyproject mismatch AND the missing CHANGELOG section are
    # reported together -- one run, two named problems.
    assert len(result.errors) == 2
    joined = "\n".join(result.errors)
    assert "1.2.3" in joined and "9.9.9" in joined
    assert "no '## [9.9.9]" in joined


def test_run_release_check_version_matches_pyproject_but_section_missing(tmp_path):
    pp, cl = _write_fixture(tmp_path, pyproject_version="9.9.9")
    result = release_check.run_release_check("9.9.9", pp, cl)
    assert result.ok is False
    assert len(result.errors) == 1
    assert "no '## [9.9.9]" in result.errors[0]


# ── main(): the CLI surface ──────────────────────────────────────────────────

def test_main_exit_0_and_writes_notes_out(tmp_path, capsys):
    pp, cl = _write_fixture(tmp_path, pyproject_version="1.2.3")
    notes_out = tmp_path / "release-notes.md"
    rc = release_check.main([
        "--version", "1.2.3",
        "--pyproject", str(pp),
        "--changelog", str(cl),
        "--notes-out", str(notes_out),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK: release checks passed for 1.2.3" in out
    assert notes_out.read_text() == EXPECTED_1_2_3_NOTES + "\n"


def test_main_exit_1_and_names_failures_on_stderr(tmp_path, capsys):
    pp, cl = _write_fixture(tmp_path, pyproject_version="1.2.3")
    rc = release_check.main([
        "--version", "9.9.9",
        "--pyproject", str(pp),
        "--changelog", str(cl),
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "release_check FAILED for version 9.9.9" in captured.err
    assert "1.2.3" in captured.err and "9.9.9" in captured.err


def test_cli_subprocess_end_to_end(tmp_path):
    # One true subprocess invocation, matching exactly how
    # .github/workflows/release.yml calls this script.
    import subprocess

    pp, cl = _write_fixture(tmp_path, pyproject_version="2.0.0")
    notes_out = tmp_path / "notes.md"
    proc = subprocess.run(
        [
            sys.executable, release_check.__file__,
            "--version", "9.9.9",
            "--pyproject", str(pp),
            "--changelog", str(cl),
            "--notes-out", str(notes_out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "release_check FAILED" in proc.stderr
    assert not notes_out.exists()


# ── real repo state (documents actual, current behavior; not a fixture) ─────

def test_real_repo_current_pyproject_version_has_a_valid_changelog_cut():
    # Deliberately NOT hardcoded to a specific version or date (that was
    # this test's original form, and it broke the invariant it meant to
    # protect: the day pyproject.toml bumps past 0.3.0 for a real release,
    # a test pinned to "0.3.0" would start failing -- reddening both this
    # suite AND ci.yml's `test` job at the exact commit that performs the
    # release cut, and release.yml gates publication on the suite passing,
    # so the very first legitimate release could never ship. Reading the
    # CURRENT version out of pyproject.toml instead states the invariant
    # that actually stays true across every future release: whatever
    # version pyproject.toml declares right now, its CHANGELOG cut must be
    # structurally valid. (Also not parametrized: a single dynamically-read
    # value has no cases to parametrize over.)
    data = tomllib.loads(release_check.DEFAULT_PYPROJECT.read_text())
    current_version = data["project"]["version"]

    result = release_check.run_release_check(
        current_version, release_check.DEFAULT_PYPROJECT, release_check.DEFAULT_CHANGELOG
    )
    assert result.ok is True, result.errors
    assert result.section_date is not None
    assert result.notes is not None and len(result.notes) > 0
