#!/usr/bin/env python3
"""Validate a release cut before CI builds and publishes anything.

The release decision, version choice, and CHANGELOG cut are the human's: pick
a version, cut CHANGELOG.md's ``[Unreleased]`` section into a dated
``[X.Y.Z] - YYYY-MM-DD`` section (leaving a fresh ``[Unreleased]`` above it),
bump ``pyproject.toml``, commit, tag ``vX.Y.Z``, push. This script is what
``.github/workflows/release.yml`` runs to VALIDATE that cut, mechanically,
before building or publishing anything:

  (a) the given version matches ``pyproject.toml``'s ``[project].version``
  (b) ``CHANGELOG.md`` has exactly one ``## [X.Y.Z] - YYYY-MM-DD`` section,
      the date is a real ISO 8601 calendar date in strict YYYY-MM-DD shape
      (not just parseable -- ``datetime.date.fromisoformat`` alone also
      accepts ISO week dates and the basic YYYYMMDD form, neither of which
      this CHANGELOG's own documented format uses), and the extracted notes
      fit under GitHub's release-body size cap
  (c) exactly one fresh ``## [Unreleased]`` heading still sits above that
      section

It never writes to the repo (aside from an optional ``--notes-out`` file, an
output artifact, not a repo edit). Zero third-party imports -- stdlib only --
so the workflow can run it before installing anything beyond python itself,
and so it stays fully unit-testable offline (see tests/test_release_check.py).
The ``vX.Y.Z`` tag -> ``X.Y.Z`` version-string strip happens in the workflow's
shell, one line, not here -- this script only ever sees a bare version, which
is what keeps it testable without any git/tag machinery.

Known limitation (not handled): this is a purely line-prefix structural scan
-- it does not track fenced ``` code blocks, so a line starting with '## '
(or a literal '## [Unreleased]') INSIDE a fenced block would be misread as a
real heading. The real CHANGELOG.md has no fenced blocks today; if that ever
changes, this scan needs fence-awareness first.

Exit codes:
    0  all checks passed (notes written to --notes-out, if given)
    1  one or more checks failed (every failure named on stderr, not just
       the first -- a release attempt that is wrong in two ways shouldn't
       need two round trips to find out)
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
try:
    import tomllib  # stdlib on Python >= 3.11
except ModuleNotFoundError:  # Python 3.10 (CI test matrix floor)
    import tomli as tomllib  # type: ignore[no-redef]  # dev extra, py<3.11 marker
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PYPROJECT = REPO / "pyproject.toml"
DEFAULT_CHANGELOG = REPO / "CHANGELOG.md"

UNRELEASED_HEADING = "## [Unreleased]"

# Strict YYYY-MM-DD shape check, ahead of datetime.date.fromisoformat's own
# calendar-validity check. fromisoformat alone is too permissive for a
# format guard: since Python 3.11 it also accepts ISO week dates
# ("2026-W01-1") and the basic, dashless form ("20260609"), neither of which
# matches this file's own documented "YYYY-MM-DD" heading format.
_ISO_DATE_SHAPE = re.compile(r"\d{4}-\d{2}-\d{2}")

# GitHub caps a release body at 125,000 characters. Stay clear of it rather
# than fail loudly, so a release attempt is rejected here -- the first step
# -- instead of after a full install/build/test cycle when `gh release
# create` finally rejects the oversized notes. The real [Unreleased] section
# was measured at 105,244 chars (84% of the cap) the day this guard was
# added, so this is not a theoretical concern.
MAX_NOTES_CHARS = 120_000


def check_pyproject_version(version: str, pyproject_path: Path) -> Optional[str]:
    """Return None if pyproject_path's [project].version == version, else an
    actionable error string naming both values."""
    try:
        raw = pyproject_path.read_text()
    except OSError as e:
        return f"could not read {pyproject_path}: {e}"
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        return f"{pyproject_path} is not valid TOML: {e}"
    try:
        actual = data["project"]["version"]
    except KeyError:
        return f"{pyproject_path} has no [project].version key"
    if actual != version:
        return (
            f"version mismatch: release version is '{version}' but "
            f"{pyproject_path} has [project].version = '{actual}' -- bump "
            f"pyproject.toml's version to '{version}' (or fix the tag) "
            f"before releasing"
        )
    return None


@dataclass
class SectionResult:
    notes: Optional[str]       # extracted body, stripped; None unless ALL checks pass
    date: Optional[str]        # the raw date string found in the heading, if any
    errors: list


def extract_changelog_section(text: str, version: str) -> SectionResult:
    """Extract the '## [version] - YYYY-MM-DD' section body from CHANGELOG text.

    Contract: find '## [version] - YYYY-MM-DD', return everything up to (but
    excluding) the next line starting with '## ' (a '### ' subsection does
    NOT count -- only three characters differ, but that's what keeps the
    body's own '### Added' etc. subsections inside the extracted notes),
    stripped. The date must be a real ISO 8601 calendar date in strict
    YYYY-MM-DD shape (rejects e.g. 2026-13-45, and rejects ISO variants like
    week dates that fromisoformat alone would accept). A fresh
    '## [Unreleased]' heading must appear somewhere above the version
    heading, and neither heading may appear more than once (the file is
    fully scanned regardless, so this costs nothing extra to check). The
    extracted notes must fit under MAX_NOTES_CHARS. Every check is attempted
    and every failure collected, rather than stopping at the first, so a
    single run names every problem.
    """
    lines = text.splitlines()
    heading_prefix = f"## [{version}] - "

    version_heading_lines = [
        i for i, line in enumerate(lines) if line.startswith(heading_prefix)
    ]
    unreleased_lines = [
        i for i, line in enumerate(lines) if line.strip() == UNRELEASED_HEADING
    ]

    errors = []

    if not version_heading_lines:
        errors.append(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section -- cut "
            f"the [Unreleased] section into '{heading_prefix}<today's ISO date>' "
            f"before tagging"
        )
        return SectionResult(None, None, errors)

    heading_idx = version_heading_lines[0]
    date_str = lines[heading_idx][len(heading_prefix):len(heading_prefix) + 10]

    if len(version_heading_lines) > 1:
        errors.append(
            f"CHANGELOG.md has more than one '## [{version}] - ...' section "
            f"(lines {', '.join(str(i + 1) for i in version_heading_lines)}) "
            f"-- exactly one dated section per version is expected"
        )

    if len(unreleased_lines) > 1:
        errors.append(
            f"CHANGELOG.md has more than one '{UNRELEASED_HEADING}' heading "
            f"(lines {', '.join(str(i + 1) for i in unreleased_lines)}) -- "
            f"exactly one is expected"
        )

    date_ok = bool(date_str) and bool(_ISO_DATE_SHAPE.fullmatch(date_str))
    if date_ok:
        try:
            datetime.date.fromisoformat(date_str)
        except ValueError:
            date_ok = False
    if not date_ok:
        errors.append(
            f"CHANGELOG.md line {heading_idx + 1} ('{lines[heading_idx]}') does "
            f"not carry a valid ISO 8601 date (strict YYYY-MM-DD, e.g. "
            f"2026-08-10) right after the version -- got '{date_str}'"
        )

    unreleased_above = any(i < heading_idx for i in unreleased_lines)
    if not unreleased_above:
        errors.append(
            f"CHANGELOG.md has no '{UNRELEASED_HEADING}' heading above the "
            f"'## [{version}]' section (line {heading_idx + 1}) -- add a fresh "
            f"'{UNRELEASED_HEADING}' heading above the cut version section"
        )

    body_lines = []
    for line in lines[heading_idx + 1:]:
        if line.startswith("## "):
            break
        body_lines.append(line)
    notes = "\n".join(body_lines).strip()

    if len(notes) > MAX_NOTES_CHARS:
        errors.append(
            f"CHANGELOG.md's '## [{version}]' section is {len(notes)} chars, "
            f"over the {MAX_NOTES_CHARS}-char safety threshold (GitHub caps a "
            f"release body at 125,000 chars) -- trim the section before tagging"
        )

    if errors:
        return SectionResult(None, date_str, errors)
    return SectionResult(notes, date_str, errors)


@dataclass
class ReleaseCheckResult:
    ok: bool
    notes: Optional[str]
    section_date: Optional[str]
    errors: list


def run_release_check(
    version: str, pyproject_path: Path, changelog_path: Path
) -> ReleaseCheckResult:
    """Run all release gates against real files. Collects every failure
    rather than stopping at the first (see extract_changelog_section)."""
    errors = []

    pv_error = check_pyproject_version(version, pyproject_path)
    if pv_error:
        errors.append(pv_error)

    try:
        changelog_text = changelog_path.read_text()
    except OSError as e:
        errors.append(f"could not read {changelog_path}: {e}")
        return ReleaseCheckResult(False, None, None, errors)

    section = extract_changelog_section(changelog_text, version)
    errors.extend(section.errors)

    if errors:
        return ReleaseCheckResult(False, None, section.date, errors)
    return ReleaseCheckResult(True, section.notes, section.date, errors)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Validate a release cut: pyproject version, CHANGELOG "
        "section + date, fresh Unreleased above it."
    )
    ap.add_argument(
        "--version", required=True,
        help="bare version to release, e.g. 1.2.3 (no leading 'v' -- the "
        "workflow strips that from the tag before calling this)",
    )
    ap.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    ap.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG)
    ap.add_argument(
        "--notes-out", type=Path, default=None,
        help="on success, write the extracted CHANGELOG section body here "
        "(release-notes source for `gh release create --notes-file`)",
    )
    args = ap.parse_args(argv)

    result = run_release_check(args.version, args.pyproject, args.changelog)

    if not result.ok:
        print(f"release_check FAILED for version {args.version}:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"OK: release checks passed for {args.version}")
    print(f"  pyproject.toml version matches: {args.version}")
    print(
        f"  CHANGELOG.md '## [{args.version}] - {result.section_date}' section "
        f"found, with a fresh '{UNRELEASED_HEADING}' above it"
    )
    notes = result.notes or ""
    print(
        f"  extracted {len(notes)} chars / {len(notes.splitlines())} lines of "
        f"release notes"
    )
    if args.notes_out:
        args.notes_out.write_text(notes + "\n")
        print(f"  notes written to {args.notes_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
