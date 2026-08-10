#!/usr/bin/env python3
"""Draft the mechanical parts of an upstream-arch sync patch.

``tools/check_upstream_drift.py`` tells you WHICH GGUF architectures the
installed ``gguf`` package knows that ``magicquant/gguf/source.py``'s
``arch_map`` does not yet translate. Turning that into a patch has always
been a human hand-writing ``arch_map`` lines and ``_HF_TO_GGUF_PATTERNS``
regexes from scratch, cross-checking ``gguf.tensor_mapping.TensorNameMap`` by
eye. This script does the mechanical half of that and stops: it drafts
candidate lines and a markdown report, and NEVER decides which architectures
are worth keeping or claims a mapping is verified. See "Control points" in
the generated report for what stays a human decision.

Trust anchor: same one ``check_upstream_drift.py`` uses -- the INSTALLED
``gguf`` package (``gguf.constants``, ``gguf.tensor_mapping``), not the
llama.cpp git tree. No network access, no scraping.

Two things are genuinely NOT derivable from that trust anchor, by design,
and this script never guesses at them silently:

  1. The HF ``model_type`` string a new architecture will actually use.
     ``gguf.constants.MODEL_ARCH`` only has the enum member name (e.g.
     ``MUSE_GLIMMER``); the real ``model_type`` in a model's ``config.json``
     is chosen by that model's authors and is NOT guaranteed to match a
     mechanical transform of the enum name. This script's guess
     (``enum_name.lower()``) is right more often than not, but is KNOWN to
     be wrong for at least one already-mapped architecture (see
     ``ARCH_MAP_HEURISTIC_CAVEAT`` below for the ``gpt_neox``/``GPTNEOX``
     counter-example) -- every derived ``arch_map`` line is reported as
     UNVERIFIED for exactly this reason.
  2. Value-level transforms (RMSNorm +1, Q/K permute, expert-tensor
     reshaping, ...) and vision/audio tensor handling. ``gguf.tensor_mapping``
     only maps NAMES; it has no concept of value transforms, and this script
     does not parse llama.cpp's ``convert_hf_to_gguf.py`` (that would be the
     scraping this project deliberately avoids). Both always land in the
     report's manual checklist, never in the derived diff.

Target list, by default, is the drift report's ``newly_appeared_architectures``
-- the DELTA against ``tools/upstream_baseline.json``, matching
``check_upstream_drift.py``'s own "only NEW drift is news" philosophy (its
exit code is 1 iff that delta is non-empty). This is deliberate: the standing
backlog (``unmapped_architectures``, ~70+ architectures MagicQuant has simply
never mapped) is not something a weekly bot run should redraft every time --
running against it once wrote 613 lines into source.py on a week with ZERO
new upstream drift. ``--full-backlog`` opts into the full backlog instead,
for a deliberate, one-time human-run clearing pass; the automated workflow
must never pass it.

Usage:
    python tools/draft_upstream_sync.py                    # dry-run report to stdout
    python tools/draft_upstream_sync.py --json drift.json  # use a saved drift report
    python tools/draft_upstream_sync.py --check             # CI gate; exit 0/1, no writes
    python tools/draft_upstream_sync.py --apply             # write source.py + report file
    python tools/draft_upstream_sync.py --full-backlog ...  # target the WHOLE backlog, not
                                                              # just new-since-baseline drift
                                                              # (one-time human use ONLY --
                                                              # never from the workflow)

Exit codes:
    0  ran fine (default/--apply mode), or nothing to draft (--check)
    1  --check found derivable work
    2  could not run (missing `gguf`, unparseable inputs, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent.parent
SOURCE_PY = REPO / "magicquant" / "gguf" / "source.py"

ARCH_MAP_HEURISTIC_CAVEAT = (
    "the model_type KEY is a heuristic guess (enum_name.lower()) -- it is "
    "right for most archs (e.g. GEMMA3 -> gemma3) but wrong whenever the "
    "gguf registry's enum name drops underscores that HF's real model_type "
    "keeps: arch_map already has to map model_type 'gpt_neox' to arch "
    "'gptneox' because the enum member is GPTNEOX (no underscore) while HF's "
    "config.json field is 'gpt_neox' (with one) -- lower()-ing the enum name "
    "alone would have produced the wrong key. Verify against the real "
    "model's config.json before merging."
)

# Prefixes/substrings that mark a MODEL_TENSOR kind as vision or audio side-car
# rather than text-decoder. gguf.constants names every vision tensor type
# with a 'V_' prefix and every audio one with 'A_' (V_MMPROJ*, V_ENC_*, A_ENC_*,
# A_MM_*, ...) and renders them to on-disk names starting 'v.'/'mm.'/'a.' --
# never 'blk.'. Failure mode: a hypothetical future MODEL_TENSOR kind that
# starts with 'A_' for a non-audio reason, or a text tensor whose HF name
# happens to contain one of the substrings below, would be wrongly diverted
# to the manual checklist (safe direction -- undercounts derived patterns,
# never smuggles a vision tensor into the auto-applied diff).
_VISION_AUDIO_TYPE_PREFIXES = ("V_", "A_", "VISEXP_")
_VISION_AUDIO_NAME_HINTS = (
    "vision", "visual", "image", "audio", "multimodal", "multi_modal", "mm_",
)


def _is_vision_or_audio(type_name: str, hf_template: str, gguf_template: str) -> bool:
    if type_name.upper().startswith(_VISION_AUDIO_TYPE_PREFIXES):
        return True
    haystack = f"{hf_template} {gguf_template}".lower()
    return any(hint in haystack for hint in _VISION_AUDIO_NAME_HINTS)


# source.py's _HF_TO_GGUF_PATTERNS deliberately does NOT map per-expert MoE
# projections 1:1 -- SafetensorsSource._ensure_loaded() intercepts them via
# _detect_moe_expert_tensor() and STACKS each layer's experts into one 3-D
# ffn_{gate,up,down}_exps tensor BEFORE this table is consulted; the file's
# own comment says matching them here "would collapse all experts of a
# projection onto the same GGUF name (last expert silently wins), producing
# an unloadable GGUF." A candidate regex this script derives from
# TensorNameMap can never know whether a new arch's on-disk format is
# already-fused (safe to map 1:1) or per-expert-indexed (needs the same kind
# of stacking as the existing MoE archs) -- so every stacked-expert kind
# (any GGUF target ending "_exps"/"_chexps": FFN_GATE_EXP, FFN_UP_EXP,
# FFN_DOWN_EXP, FFN_GATE_UP_EXP, FFN_NORM_EXP, the *_CHEXP family) is routed
# to the manual checklist instead of the derived diff. Shared-expert kinds
# (GGUF target ending "_shexp", singular -- always one tensor, never
# per-expert-indexed) are NOT in this category and are auto-derived normally.
def _is_stacked_moe_expert(gguf_template: str) -> bool:
    stem = gguf_template.rsplit(".", 1)[-1]
    return stem.endswith("_exps") or stem.endswith("_chexps")


# llama.cpp's converters SKIP these -- rope inv_freq / rope_factors buffers
# are recomputed at load time from `<arch>.rope.freq_base` and related
# metadata, never packed as stored tensors, regardless of whether the HF
# checkpoint ships a literal `rotary_emb.inv_freq` buffer. gguf.constants
# still defines GGUF-side names for them (legacy / niche archs did once pack
# them), so TensorNameMap happily offers candidates -- but auto-deriving a
# pattern for one would write bytes downstream code never reads and llama.cpp
# never expects to load. Named exclusion set, not a substring sniff, because
# "rope"/"rot" is too common a substring to safely blacklist by text alone.
_ROPE_BUFFER_TYPES = {"ROPE_FREQS", "ROPE_FACTORS_LONG", "ROPE_FACTORS_SHORT", "ATTN_ROT_EMBD"}

# A NON-block (no `{bid}` layer capture) candidate whose HF name is a single
# bare path segment (no '.') is dangerously generic once it lands in
# _HF_TO_GGUF_PATTERNS, which is consumed GLOBALLY across every architecture
# MagicQuant ever converts -- e.g. `^dense\.weight$` or `^classifier\.weight$`
# could just as easily belong to an unrelated tensor in some future model.
# Never auto-derived; always routed to the manual checklist instead.
def _is_over_generic_anchor(hf_template: str, is_block: bool) -> bool:
    return not is_block and "." not in hf_template


# More than this many surviving candidate spellings for one NOVEL tensor kind
# is a sign TensorNameMap's static table is showing conventions from SEVERAL
# unrelated architectures, not necessarily anything close to the one being
# drafted -- e.g. GRANITE_HYBRID's SSM_A kind pulled in every known "A_log"
# spelling across mamba/mamba2/jamba/rwkv/qwen3.5-linear-attn, most of which
# have nothing to do with granite-hybrid's actual on-disk format. Past this
# threshold, NONE of that kind's spellings are auto-derived -- the whole
# list goes to the manual checklist for a human to pick the one that matches
# the real model, rather than cross-arch-dumping all of them as candidates.
_MAX_SURVIVING_SPELLINGS = 2


def _tensor_type_name(tensor) -> str:
    """Name of a MODEL_TENSOR-like key, for both real enum members (which
    have .name) and the plain-string sentinels the test fixture uses."""
    name = getattr(tensor, "name", None)
    return name if name is not None else str(tensor)


# ---------------------------------------------------------------------------
# Loading the drift report and the installed gguf package
# ---------------------------------------------------------------------------

def _load_drift_report(json_path: Optional[str]) -> dict:
    """Return the drift report dict: from a saved file, or freshly computed."""
    if json_path:
        return json.loads(Path(json_path).read_text())
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_upstream_drift.py"), "--json"],
        capture_output=True, text=True,
    )
    # check_upstream_drift.py's exit code encodes drift-since-baseline (0/1),
    # not failure -- only 2 (missing `gguf`) is a real error here.
    if proc.returncode == 2:
        raise RuntimeError(
            proc.stderr.strip() or "tools/check_upstream_drift.py --json failed"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"tools/check_upstream_drift.py --json produced unparseable "
            f"output: {exc}"
        ) from exc


def _load_installed_gguf():
    """Return (gguf.constants, gguf.tensor_mapping.TensorNameMap)."""
    import gguf.constants as gc
    import gguf.tensor_mapping as tm
    return gc, tm.TensorNameMap


def resolve_arch_string(enum_name: str, gc_module) -> Optional[str]:
    """Look up the GGUF architecture STRING for a MODEL_ARCH enum member name.

    This is the one piece that is NEVER guessed: gguf.constants.MODEL_ARCH_NAMES
    is the same table llama.cpp itself derives the on-disk `general.architecture`
    string from, so 'MUSE_GLIMMER' -> 'muse-glimmer' (note the hyphen -- NOT a
    lowercase-and-replace-underscore transform of the enum name; e.g. COMMAND_R
    -> 'command-r' while COHERE2 -> 'cohere2', two different shapes that a naive
    string transform would get wrong. Returns None if the installed gguf
    package does not know this enum member at all (only possible when --json
    points at a drift report computed against a different gguf version).
    """
    member = gc_module.MODEL_ARCH.__members__.get(enum_name)
    if member is None:
        return None
    return gc_module.MODEL_ARCH_NAMES.get(member)


# ---------------------------------------------------------------------------
# arch_map: parsing the existing block + deriving candidates
# ---------------------------------------------------------------------------

_ARCH_MAP_BLOCK_RE = re.compile(r"arch_map\s*=\s*\{(.*?)\n    \}", re.S)
_ARCH_MAP_PAIR_RE = re.compile(r'"([a-zA-Z0-9_]+)":\s*"([^"]+)"')


def _parse_arch_map(text: str):
    """Return (block_match, [(key, value, absolute_match_start_offset), ...])."""
    m = _ARCH_MAP_BLOCK_RE.search(text)
    if not m:
        raise RuntimeError(
            "could not locate arch_map in magicquant/gguf/source.py -- this "
            "generator greps it the same way tools/check_upstream_drift.py "
            "does, so a refactor there needs a matching edit here"
        )
    block = m.group(1)
    block_start = m.start(1)
    pairs = [
        (pm.group(1), pm.group(2), block_start + pm.start())
        for pm in _ARCH_MAP_PAIR_RE.finditer(block)
    ]
    return m, pairs


@dataclass
class ArchCandidate:
    enum_name: str
    model_type: str
    arch_string: Optional[str]
    already_present: bool
    is_text_variant: bool


def derive_arch_map_candidates(
    unmapped_enum_names: list, existing_pairs: list, gc_module
) -> list:
    """Candidate arch_map lines for each unmapped enum name.

    For every enum name this proposes TWO model_type keys: the bare guess and
    a '_text' variant (mirroring the qwen3_5/qwen3_5_text precedent already in
    arch_map -- composite/multimodal HF configs commonly nest a '..._text'
    model_type under text_config). Both are speculative; the report tells a
    human to delete whichever doesn't apply rather than silently guessing
    which one is right.

    Already-mapped archs are excluded even if the drift JSON is stale: this
    checks the CURRENT arch_map's VALUES (normalized the same way
    check_upstream_drift.py does), not just candidate keys, so an arch already
    reachable under a differently-spelled key is still skipped.
    """
    existing_keys = {k for k, _, _ in existing_pairs}
    existing_values_norm = {v.upper().replace("-", "_") for _, v, _ in existing_pairs}

    out = []
    for enum_name in unmapped_enum_names:
        arch_string = resolve_arch_string(enum_name, gc_module)
        if arch_string is not None:
            if arch_string.upper().replace("-", "_") in existing_values_norm:
                continue  # already mapped under some other key -- exclude
        base_model_type = enum_name.lower()
        for suffix, is_text in (("", False), ("_text", True)):
            model_type = base_model_type + suffix
            out.append(ArchCandidate(
                enum_name=enum_name,
                model_type=model_type,
                arch_string=arch_string,
                already_present=model_type in existing_keys,
                is_text_variant=is_text,
            ))
    return out


def _insert_arch_map_line(text: str, key: str, value: str, enum_name: str):
    """Insert a new arch_map line, alphabetically placed. Returns (text, inserted).

    Carries the same "# <enum> -- draft_upstream_sync, UNVERIFIED" trailing
    comment convention _insert_pattern_line uses -- earlier the arch_map side
    had no marker at all, which made a merged-without-review line
    indistinguishable from a hand-verified one.
    """
    m, pairs = _parse_arch_map(text)
    if any(k == key for k, _, _ in pairs):
        return text, False  # idempotent: already there
    anchor_offset = None
    for k, _, off in pairs:
        if k > key:
            anchor_offset = off
            break
    new_line = (
        f'        "{key}": "{value}",'
        f'  # {enum_name} -- draft_upstream_sync, UNVERIFIED\n'
    )
    if anchor_offset is not None:
        line_start = text.rfind("\n", 0, anchor_offset) + 1
        new_text = text[:line_start] + new_line + text[line_start:]
    else:
        # New key sorts after everything already there: insert as the last
        # entry, right before the block's closing "\n    }".
        insert_at = m.start(1) + len(m.group(1)) + 1  # just past the final "\n"
        new_text = text[:insert_at] + new_line + text[insert_at:]
    return new_text, True


# ---------------------------------------------------------------------------
# _HF_TO_GGUF_PATTERNS: parsing the existing table + deriving candidates
# ---------------------------------------------------------------------------

_PATTERNS_BLOCK_RE = re.compile(r"_HF_TO_GGUF_PATTERNS\s*=\s*\[(.*?)\n\]", re.S)
_PATTERN_REGEX_RE = re.compile(r'\(\s*r"([^"]*)"')

_SYNC_HEADER = (
    "    # --- draft_upstream_sync.py candidates "
    "(UNVERIFIED -- see PR body checklist before merging) ---\n"
)


def _patterns_block(text: str):
    m = _PATTERNS_BLOCK_RE.search(text)
    if not m:
        raise RuntimeError(
            "could not locate _HF_TO_GGUF_PATTERNS in magicquant/gguf/"
            "source.py -- refactor there needs a matching edit here"
        )
    return m


def _existing_pattern_sources(text: str) -> list:
    return _PATTERN_REGEX_RE.findall(_patterns_block(text).group(1))


def _render_example(hf_template: str) -> str:
    """A concrete example HF tensor path for a template, for match-testing."""
    return hf_template.replace("{bid}", "0") + ".weight"


def _regex_and_replacement(hf_template: str, gguf_template: str):
    """Build (regex_source, replacement_source, is_block) matching the style
    already used throughout _HF_TO_GGUF_PATTERNS (anchored, .weight-suffixed,
    lambda replacement for block-scoped tensors)."""
    is_block = "{bid}" in hf_template
    if is_block:
        escaped = r"(\d+)".join(re.escape(p) for p in hf_template.split("{bid}"))
        regex_source = f"^{escaped}\\.weight$"
        gguf_out = gguf_template.replace("{bid}", "{m.group(1)}")
        replacement_source = f'lambda m: f"{gguf_out}.weight"'
    else:
        regex_source = f"^{re.escape(hf_template)}\\.weight$"
        replacement_source = repr(f"{gguf_template}.weight")
    return regex_source, replacement_source, is_block


@dataclass
class PatternCandidate:
    enum_name: str
    tensor_type: str
    hf_template: str
    gguf_template: str
    regex_source: str
    replacement_source: str
    is_block: bool


@dataclass
class ArchPatternReport:
    enum_name: str
    arch_known: bool
    arch_string: Optional[str]
    proposed: list = field(default_factory=list)
    skipped_types: list = field(default_factory=list)   # overlap w/ existing table
    vision_types: list = field(default_factory=list)    # excluded, manual review
    vision_hf_examples: list = field(default_factory=list)  # concrete HF prefixes seen
    moe_expert_types: list = field(default_factory=list)  # excluded, manual review
    rope_buffer_types: list = field(default_factory=list)   # excluded, never packed
    over_generic_examples: list = field(default_factory=list)  # [(type, hf_template)]
    ambiguous_types: dict = field(default_factory=dict)     # type -> [hf_template, ...]


def _collect_rows(gc_module, tm_cls, member):
    """{tensor_type: [hf_name_template, ...]} for tensor kinds `member` has,
    pulled from TensorNameMap's static class tables (no instantiation --
    those tables carry '{bid}' markers directly, which is exactly the
    genericized form _HF_TO_GGUF_PATTERNS wants)."""
    tensor_types = set(gc_module.MODEL_TENSORS.get(member, []))
    rows = {}
    for tensor, keys in tm_cls.mappings_cfg.items():
        if tensor not in tensor_types:
            continue
        rows.setdefault(tensor, []).extend(keys)
    block_cfg = dict(tm_cls.block_mappings_cfg)
    block_cfg.update(getattr(tm_cls, "arch_block_mappings_cfg", {}).get(member, {}))
    for tensor, keys in block_cfg.items():
        if tensor not in tensor_types:
            continue
        rows.setdefault(tensor, []).extend(keys)
    return rows


def derive_pattern_report(
    enum_name: str, existing_pattern_sources: list, gc_module, tm_cls
) -> ArchPatternReport:
    """Candidate _HF_TO_GGUF_PATTERNS additions for one unmapped architecture.

    Per tensor kind the arch declares (MODEL_TENSORS[arch]):
      - vision/audio kinds are excluded entirely (manual checklist instead --
        see _is_vision_or_audio);
      - a kind is SKIPPED (silently, reported separately) if ANY of
        TensorNameMap's known historical HF-name spellings for it already
        matches an existing _HF_TO_GGUF_PATTERNS entry -- the reasoning is
        that MagicQuant's generic llama-style patterns most likely already
        cover a model using the mainstream field name, so dumping the other
        ~15 legacy spellings (gpt2/bert/bloom/... conventions) nobody will
        ever hit would be pure noise;
      - rope/inv_freq buffer kinds are excluded entirely (llama.cpp's
        converters never pack them -- see _ROPE_BUFFER_TYPES);
      - an over-generic bare (non-block, dot-free) HF name is excluded
        individually (see _is_over_generic_anchor) even if its siblings for
        the same kind survive;
      - a kind with more than _MAX_SURVIVING_SPELLINGS surviving spellings
        is excluded WHOLESALE (none of them proposed) -- past that count
        TensorNameMap is most likely showing conventions from several
        unrelated architectures, not candidates for this one;
      - otherwise EVERY surviving spelling for that kind is proposed (this
        list is inherently short for a genuinely novel kind like an
        attention gate or a sandwich norm, which is exactly the case that
        matters).

    Candidates from THIS function are still per-architecture and may
    duplicate or conflict with another architecture's candidates -- see
    resolve_candidates() for the cross-architecture dedup/conflict pass that
    both build_report_markdown() and apply_changes() run before using them.

    This dedup is per literal HF-name example, never per GGUF-target stem --
    two different HF source names CAN legitimately map to the same GGUF
    target (e.g. qwen3.5's linear_attn.in_proj_z and a hypothetical arch's
    self_attn.gate_proj both legitimately produce blk.N.attn_gate.weight),
    so "some other pattern already reaches this target" must NOT suppress a
    genuinely new source-name candidate.
    """
    member = gc_module.MODEL_ARCH.__members__.get(enum_name)
    if member is None:
        return ArchPatternReport(enum_name, arch_known=False, arch_string=None)

    arch_string = gc_module.MODEL_ARCH_NAMES.get(member)
    compiled_existing = [re.compile(p) for p in existing_pattern_sources]
    rows_by_type = _collect_rows(gc_module, tm_cls, member)

    proposed, skipped, vision, vision_examples, moe_expert = [], [], [], [], []
    rope_buffer, over_generic, ambiguous = [], [], {}
    for tensor in sorted(rows_by_type, key=_tensor_type_name):
        type_name = _tensor_type_name(tensor)
        gguf_template = gc_module.TENSOR_NAMES.get(tensor, "")
        hf_templates = list(dict.fromkeys(rows_by_type[tensor]))  # de-dup, keep order

        if any(_is_vision_or_audio(type_name, t, gguf_template) for t in hf_templates):
            vision.append(type_name)
            vision_examples.extend(hf_templates)
            continue

        if _is_stacked_moe_expert(gguf_template):
            moe_expert.append(type_name)
            continue

        if type_name in _ROPE_BUFFER_TYPES:
            rope_buffer.append(type_name)
            continue

        examples = [_render_example(t) for t in hf_templates]
        if any(p.match(ex) for p in compiled_existing for ex in examples):
            skipped.append(type_name)
            continue

        # Per-template: drop over-generic bare anchors individually (the
        # REST of this kind's spellings, if any, are still candidates).
        survivors = []
        for hf_template in hf_templates:
            is_block = "{bid}" in hf_template
            if _is_over_generic_anchor(hf_template, is_block):
                over_generic.append((type_name, hf_template))
                continue
            survivors.append(hf_template)
        if not survivors:
            continue

        # Per-kind: too many surviving spellings means "ambiguous, probably
        # a cross-arch grab-bag" -- propose NONE of them, list them instead.
        if len(survivors) > _MAX_SURVIVING_SPELLINGS:
            ambiguous[type_name] = survivors
            continue

        for hf_template in survivors:
            regex_source, repl_source, is_block = _regex_and_replacement(
                hf_template, gguf_template
            )
            proposed.append(PatternCandidate(
                enum_name=enum_name, tensor_type=type_name,
                hf_template=hf_template, gguf_template=gguf_template,
                regex_source=regex_source, replacement_source=repl_source,
                is_block=is_block,
            ))

    proposed.sort(key=lambda c: (c.tensor_type, c.hf_template))
    return ArchPatternReport(
        enum_name=enum_name, arch_known=True, arch_string=arch_string,
        proposed=proposed, skipped_types=sorted(set(skipped)),
        vision_types=sorted(set(vision)),
        vision_hf_examples=sorted(set(vision_examples)),
        moe_expert_types=sorted(set(moe_expert)),
        rope_buffer_types=sorted(set(rope_buffer)),
        over_generic_examples=sorted(set(over_generic)),
        ambiguous_types={k: sorted(v) for k, v in sorted(ambiguous.items())},
    )


# ---------------------------------------------------------------------------
# Cross-architecture resolution: dedup identical candidates, quarantine
# conflicting ones. Both build_report_markdown() and apply_changes() call
# this on the SAME pattern_reports list so what gets reported and what gets
# written always agree -- the earlier version reported each architecture's
# raw proposals independently (812 lines) while apply_changes' own
# idempotent-insert check quietly collapsed identical regexes at write time
# (236 lines), so the PR body and the actual diff disagreed by 71%.
# ---------------------------------------------------------------------------

@dataclass
class ResolvedPattern:
    regex_source: str
    replacement_source: str
    tensor_type: str
    enum_names: list  # every architecture that independently proposed this


@dataclass
class PatternConflict:
    regex_source: str
    targets: list  # [(replacement_source, [enum_names, ...], tensor_kind), ...], sorted


def resolve_candidates(pattern_reports: list):
    """Collapse identical (regex, replacement) candidates from different
    architectures into one, and pull out CONFLICTS: the same regex_source
    proposed with two or more DIFFERENT replacement_source values.

    A conflict is a real correctness hazard, not a formatting nuisance --
    exactly one of those targets can be right for a given literal HF tensor
    name, and picking one by sort order (or insertion order, or any other
    arbitrary tiebreak) is the qwen3_5 failure mode: a name-only mapping
    that looks plausible and is silently wrong for whichever architecture
    didn't get its target. Every regex on either side of a conflict is
    excluded from `distinct` entirely (never auto-derived, never applied)
    and instead reported with ALL of its candidate targets so a human picks.

    Returns (distinct: list[ResolvedPattern], conflicts: list[PatternConflict]),
    both sorted for determinism.
    """
    seen: dict = {}          # (regex, repl) -> ResolvedPattern
    targets_by_regex: dict = {}  # regex -> {repl: set(enum_names)}
    kinds_by_target: dict = {}   # (regex, repl) -> set(tensor_kind)
    for r in pattern_reports:
        for pc in r.proposed:
            key = (pc.regex_source, pc.replacement_source)
            rp = seen.get(key)
            if rp is None:
                rp = ResolvedPattern(pc.regex_source, pc.replacement_source, pc.tensor_type, [])
                seen[key] = rp
            if pc.enum_name not in rp.enum_names:
                rp.enum_names.append(pc.enum_name)
            targets_by_regex.setdefault(pc.regex_source, {}).setdefault(
                pc.replacement_source, set()
            ).add(pc.enum_name)
            kinds_by_target.setdefault(
                (pc.regex_source, pc.replacement_source), set()
            ).add(pc.tensor_type)

    conflicting_regex = {rx for rx, targets in targets_by_regex.items() if len(targets) > 1}

    distinct = [rp for (rx, _), rp in seen.items() if rx not in conflicting_regex]
    for rp in distinct:
        rp.enum_names.sort()
    distinct.sort(key=lambda rp: (rp.tensor_type, rp.regex_source))

    conflicts = []
    for rx in sorted(conflicting_regex):
        targets = [
            (repl, sorted(names), "/".join(sorted(kinds_by_target[(rx, repl)])))
            for repl, names in sorted(targets_by_regex[rx].items())
        ]
        conflicts.append(PatternConflict(regex_source=rx, targets=targets))

    return distinct, conflicts


# ---------------------------------------------------------------------------
# --apply: writing the candidates into source.py
# ---------------------------------------------------------------------------

@dataclass
class ApplyResult:
    arch_lines_written: list
    pattern_lines_written: list  # list[ResolvedPattern]


def apply_changes(
    source_path: Path, arch_candidates: list, pattern_reports: list
) -> ApplyResult:
    text = source_path.read_text()
    written_arch = []

    to_insert = sorted(
        (c for c in arch_candidates if c.arch_string is not None and not c.already_present),
        key=lambda c: c.model_type,
    )
    for c in to_insert:
        text, inserted = _insert_arch_map_line(text, c.model_type, c.arch_string, c.enum_name)
        if inserted:
            written_arch.append(c)

    # Resolve BEFORE writing, not per-arch: two architectures proposing the
    # identical (regex, replacement) pair must produce exactly one line, and
    # a regex proposed with two DIFFERENT replacements must produce none --
    # see resolve_candidates()'s docstring for why either of those handled
    # any other way is a correctness or noise problem.
    distinct, _conflicts = resolve_candidates(pattern_reports)
    written_patterns = []
    for rp in distinct:
        existing = set(_existing_pattern_sources(text))
        if rp.regex_source in existing:
            continue  # idempotent: a previous --apply already added this exact regex
        m = _patterns_block(text)
        insert_at = m.start(1) + len(m.group(1)) + 1
        chunk = _SYNC_HEADER if _SYNC_HEADER not in text else ""
        credit = ", ".join(rp.enum_names)
        chunk += (
            f'    (r"{rp.regex_source}",\n'
            f'     {rp.replacement_source}),'
            f'  # {credit} / {rp.tensor_type} -- draft_upstream_sync, UNVERIFIED\n'
        )
        text = text[:insert_at] + chunk + text[insert_at:]
        written_patterns.append(rp)

    if written_arch or written_patterns:
        tmp = source_path.with_suffix(".py.tmp")
        tmp.write_text(text)
        os.replace(tmp, source_path)

    return ApplyResult(written_arch, written_patterns)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

QWEN35_INCIDENT_QUOTE = (
    "\"A safetensors source whose arch needs HF->GGUF value transforms "
    "MagicQuant has not implemented (or verified). Packing it anyway "
    "produces a GGUF that loads and runs but emits garbage -- see the "
    "qwen3_5 uniform-logits incident (PPL == vocab size, every tensor name "
    "and shape matched the reference, but 64% of tensor VALUES were "
    "wrong).\"\n"
    "-- magicquant/gguf/source.py, UnsupportedSourceArchitecture docstring"
)


def build_report_markdown(
    unmapped: list, arch_candidates: list, pattern_reports: list, gguf_version: str,
    *, full_backlog: bool = False,
) -> str:
    """The FULL report: derived diffs, every exclusion bucket explained, a
    per-architecture checklist. This is written to a file and committed to
    the PR (see build_pr_summary_markdown for the short PR-body form that
    links to it -- GitHub's PR body has a 65536-char hard cap and this report
    alone exceeds that well before the backlog is large)."""
    distinct, conflicts = resolve_candidates(pattern_reports)

    lines = []
    lines.append("# Upstream sync: draft patch")
    lines.append("")
    target_note = (
        "the FULL unmapped-architecture backlog (`--full-backlog`, one-time "
        "human-run mode -- never the automated workflow)"
        if full_backlog else
        "architectures newly appeared since the last accepted "
        "`tools/upstream_baseline.json` (`newly_appeared_architectures`)"
    )
    lines.append(
        "Auto-generated by `tools/draft_upstream_sync.py` from the installed "
        f"`gguf` package (version {gguf_version}) and "
        "`tools/check_upstream_drift.py --json`. Nothing here is verified "
        "against a real model checkpoint -- see the per-architecture checklist. "
        f"Target list for this run: {target_note}."
    )
    lines.append("")

    resolvable = [c for c in arch_candidates if c.arch_string is not None]
    unresolvable_archs = sorted({
        c.enum_name for c in arch_candidates if c.arch_string is None
    })
    new_arch_lines = [c for c in resolvable if not c.already_present]
    archs_with_patterns = sorted({name for rp in distinct for name in rp.enum_names})
    vision_archs = sorted({r.enum_name for r in pattern_reports if r.vision_types})
    moe_expert_archs = sorted({r.enum_name for r in pattern_reports if r.moe_expert_types})
    rope_archs = sorted({r.enum_name for r in pattern_reports if r.rope_buffer_types})
    over_generic_archs = sorted({r.enum_name for r in pattern_reports if r.over_generic_examples})
    ambiguous_archs = sorted({r.enum_name for r in pattern_reports if r.ambiguous_types})

    lines.append("## Summary")
    lines.append(f"- {len(unmapped)} architecture(s) considered")
    lines.append(
        f"- {len(new_arch_lines)} `arch_map` candidate line(s) derived "
        f"across {len(sorted({c.enum_name for c in new_arch_lines}))} architecture(s)"
    )
    lines.append(
        f"- {len(distinct)} distinct `_HF_TO_GGUF_PATTERNS` candidate line(s) "
        f"derived across {len(archs_with_patterns)} architecture(s) "
        "(after cross-architecture dedup -- see below)"
    )
    if conflicts:
        lines.append(
            f"- {len(conflicts)} regex conflict(s) found and EXCLUDED from the "
            "derived diff entirely (manual checklist below) -- see 'Conflicts'"
        )
    if rope_archs:
        lines.append(
            f"- {len(rope_archs)} architecture(s) have rope/inv_freq buffer "
            "kinds excluded (llama.cpp never packs these; recomputed at load)"
        )
    if over_generic_archs:
        lines.append(
            f"- {len(over_generic_archs)} architecture(s) have over-generic "
            "bare-name candidates excluded (manual checklist below)"
        )
    if ambiguous_archs:
        lines.append(
            f"- {len(ambiguous_archs)} architecture(s) have kind(s) with too "
            f"many surviving spellings (>{_MAX_SURVIVING_SPELLINGS}) excluded "
            "wholesale (manual checklist below)"
        )
    if vision_archs:
        lines.append(
            f"- {len(vision_archs)} architecture(s) have vision/audio tensor "
            "kinds excluded from the derived diff (manual checklist below)"
        )
    if moe_expert_archs:
        lines.append(
            f"- {len(moe_expert_archs)} architecture(s) have stacked-MoE-expert "
            "tensor kinds excluded from the derived diff (manual checklist below)"
        )
    if unresolvable_archs:
        lines.append(
            f"- {len(unresolvable_archs)} architecture(s) could not be resolved "
            f"against the installed gguf {gguf_version} at all (enum name not "
            "found -- likely a drift report computed against a different gguf "
            "version than this run): " + ", ".join(unresolvable_archs)
        )
    lines.append("")

    if new_arch_lines:
        lines.append("## Derived: `arch_map` additions (`magicquant/gguf/source.py`)")
        lines.append("```python")
        for c in sorted(new_arch_lines, key=lambda c: c.model_type):
            lines.append(
                f'        "{c.model_type}": "{c.arch_string}",'
                f'  # {c.enum_name} -- draft_upstream_sync, UNVERIFIED'
            )
        lines.append("```")
        lines.append("")
        lines.append(f"**Heuristic disclosure**: {ARCH_MAP_HEURISTIC_CAVEAT}")
        lines.append(
            "A `_text` variant line is always proposed alongside the base line "
            "(pattern: `qwen3_5`/`qwen3_5_text`, both already mapping to "
            "`qwen35`, is the existing precedent) because composite/multimodal "
            "HF configs commonly nest a `..._text`-suffixed `model_type` under "
            "`text_config`. Delete whichever line doesn't apply -- don't leave "
            "an unused key in arch_map."
        )
        lines.append("")

    if distinct:
        lines.append("## Derived: `_HF_TO_GGUF_PATTERNS` additions")
        lines.append("```python")
        for rp in distinct:
            lines.append(f'    (r"{rp.regex_source}",')
            lines.append(
                f'     {rp.replacement_source}),'
                f'  # {", ".join(rp.enum_names)} / {rp.tensor_type}'
            )
        lines.append("```")
        lines.append("")
        lines.append(
            "Derived from `gguf.tensor_mapping.TensorNameMap`'s static tables, "
            "restricted to tensor kinds each architecture actually declares "
            "(`MODEL_TENSORS[arch]`), and only for kinds where NONE of "
            "TensorNameMap's known historical HF-name spellings already "
            "matches an existing pattern in this file -- see 'Silently "
            "skipped' below for what was dropped, and why that's expected to "
            "be safe rather than an omission. Deduplicated across "
            "architectures: a candidate proposed identically by more than "
            "one architecture appears ONCE here, credited to all of them."
        )
        lines.append("")

    if conflicts:
        lines.append("## Conflicts (excluded -- pick one manually)")
        lines.append(
            "The same HF tensor-name pattern was independently derived with "
            "DIFFERENT GGUF targets for different architectures. Exactly one "
            "target can be right for a given literal tensor name; picking by "
            "sort order would be the qwen3_5 failure mode (a plausible-looking "
            "name mapping that is silently wrong). Neither target below was "
            "applied or included in the derived diff above."
        )
        lines.append("")
        for c in conflicts:
            lines.append(f"- `{c.regex_source}`")
            for repl, names, kind in c.targets:
                lines.append(
                    f"  - `{repl}` (kind: {kind}) -- proposed by {', '.join(names)}"
                )
        lines.append("")

    skipped_by_arch = {
        r.enum_name: r.skipped_types for r in pattern_reports if r.skipped_types
    }
    if skipped_by_arch:
        lines.append(
            "## Silently skipped (standard convention likely already covers these)"
        )
        for enum_name in sorted(skipped_by_arch):
            types = ", ".join(skipped_by_arch[enum_name])
            lines.append(f"- **{enum_name}**: {types}")
        lines.append(
            "\n  At least one of TensorNameMap's known HF spellings for each "
            "kind above already matches an existing `_HF_TO_GGUF_PATTERNS` "
            "entry, so a model using the mainstream llama-style field names "
            "needs no new pattern. If the real model uses an unusual variant "
            "spelling instead, conversion will silently keep the raw HF name "
            "and llama.cpp will reject the GGUF -- verify against the real "
            "model's tensor list if conversion fails."
        )
        lines.append("")

    lines.append("## Manual checklist (never auto-derived)")
    if vision_archs:
        for r in pattern_reports:
            if not r.vision_types:
                continue
            examples = ", ".join(r.vision_hf_examples[:6])
            lines.append(
                f"- [ ] **{r.enum_name}** vision/audio components: "
                f"{len(r.vision_types)} tensor kind(s) excluded here "
                f"({', '.join(r.vision_types)}; HF-side prefixes seen: "
                f"{examples}) -- MagicQuant does not auto-map vision/audio "
                "tensors. Decide whether source.py needs skip-list handling "
                "for these, or whether the model will only ever be "
                "converted text-only."
            )
    if moe_expert_archs:
        for r in pattern_reports:
            if not r.moe_expert_types:
                continue
            lines.append(
                f"- [ ] **{r.enum_name}** stacked MoE-expert tensors: "
                f"{len(r.moe_expert_types)} tensor kind(s) excluded here "
                f"({', '.join(r.moe_expert_types)}) -- these need the same "
                "per-expert stacking `SafetensorsSource._detect_moe_expert_tensor` "
                "already does for the existing MoE archs (see the "
                "`_HF_TO_GGUF_PATTERNS` module comment on why a naive 1:1 "
                "regex here can silently collapse all experts onto one "
                "GGUF tensor). Add dedicated stacking support, not a "
                "pattern-table line."
            )
    if over_generic_archs:
        for r in pattern_reports:
            if not r.over_generic_examples:
                continue
            examples = ", ".join(f"`{t}` ({k})" for k, t in r.over_generic_examples)
            lines.append(
                f"- [ ] **{r.enum_name}** over-generic bare-name candidate(s) "
                f"excluded: {examples} -- no layer capture and a single bare "
                "path segment is too generic for a pattern table consumed "
                "GLOBALLY across every architecture MagicQuant converts. Add "
                "a scoped pattern by hand if this tensor genuinely needs one."
            )
    if ambiguous_archs:
        for r in pattern_reports:
            if not r.ambiguous_types:
                continue
            for kind, spellings in r.ambiguous_types.items():
                lines.append(
                    f"- [ ] **{r.enum_name}** `{kind}`: "
                    f"{len(spellings)} candidate spellings, none auto-derived "
                    f"(> {_MAX_SURVIVING_SPELLINGS} surviving spellings usually "
                    "means TensorNameMap is showing conventions from several "
                    f"UNRELATED architectures): {', '.join(spellings)}. Pick "
                    "the one that matches the real model's tensor names."
                )
    lines.append(
        "- [ ] **value transforms / QK-permute**, for every architecture "
        "above with derived lines: this tool only reads `gguf`'s static NAME "
        "tables, never llama.cpp's `convert_hf_to_gguf.py` conversion "
        "classes (that would be exactly the upstream scraping this project "
        "avoids). A name-only mapping can be completely wrong at the VALUE "
        "level and still look right -- see the qwen3_5 incident quote below. "
        "Check upstream's per-arch `modify_tensors` override before trusting "
        "these."
    )
    lines.append("")

    lines.append("## Per-architecture verification checklist")
    for enum_name in unmapped:
        cands = [c for c in arch_candidates if c.enum_name == enum_name]
        if not cands:
            # derive_arch_map_candidates omits an enum entirely when it's
            # already mapped under some OTHER key -- distinct from "arch
            # string unresolvable" below, which still produces candidates
            # (with arch_string=None) rather than an empty list. Only
            # reachable with a stale --json (arch_map has moved on since it
            # was computed); a fresh run never lists an already-mapped arch
            # in `unmapped` to begin with.
            lines.append(f"### {enum_name}")
            lines.append(
                "- already mapped under a different key than this tool's "
                "guess (`arch_map`'s VALUES already cover it); nothing "
                "drafted. This only happens with a stale `--json` input -- "
                "re-run `check_upstream_drift.py --json` fresh if this "
                "looks wrong."
            )
            lines.append("")
            continue
        arch_string = next((c.arch_string for c in cands if c.arch_string), None)
        header = f"### {enum_name}" + (f" -> `{arch_string}`" if arch_string else " (unresolved)")
        lines.append(header)
        if arch_string is None:
            lines.append(
                "- [ ] not resolvable against the currently-installed gguf "
                f"{gguf_version}; re-run this tool once gguf is upgraded"
            )
            lines.append("")
            continue
        already = all(c.already_present for c in cands)
        if already:
            lines.append("- already fully mapped under the guessed key(s); nothing drafted")
            lines.append("")
            continue
        lines.append(
            "- [ ] `model_type` guess above matches the real model's "
            "`config.json` (heuristic, see disclosure above)"
        )
        lines.append(
            "- [ ] real-model conversion parity **NOT verified** -- no "
            f"{enum_name} checkpoint was converted end-to-end and diffed "
            "against upstream llama.cpp's own converter"
        )
        lines.append(
            "- [ ] keep-or-delete: if MagicQuant will never convert this "
            "architecture, delete its lines from `arch_map` and "
            "`_HF_TO_GGUF_PATTERNS` rather than leaving them unused"
        )
        lines.append("")

    lines.append("## Why merging is the control point")
    lines.append(
        "This bot never merges its own PR, and the checklists above are not "
        "decoration. The canonical cost of guessing instead of verifying:"
    )
    lines.append("")
    lines.append("> " + QWEN35_INCIDENT_QUOTE.replace("\n", "\n> "))
    lines.append("")
    lines.append(
        "Every candidate in this PR is a NAME mapping derived from a static "
        "registry table, with no execution against a real checkpoint. Merge "
        "only after checking the boxes above."
    )

    return "\n".join(lines) + "\n"


# GitHub's PR body hard-caps at 65536 chars and 422s the create/update call
# outright above it -- the full report (build_report_markdown) blew through
# that at 151,873 chars on the real backlog. REPORT_FILENAME is the file the
# full report gets committed to instead (part of the PR diff, reviewable in
# Files Changed); the PR body itself is build_pr_summary_markdown's short
# form, which links to it. _MAX_PR_BODY_CHARS leaves real headroom under the
# cap as a hard safety net independent of how good the summary's own sizing
# turns out to be.
REPORT_FILENAME = ".github/upstream-sync-report.md"
_MAX_PR_BODY_CHARS = 60_000


def _truncate_for_pr_body(body: str, limit: int = _MAX_PR_BODY_CHARS) -> str:
    if len(body) <= limit:
        return body
    cut = body[:limit]
    nl = cut.rfind("\n")
    if nl > 0:
        cut = cut[:nl]
    return (
        cut + "\n\n... [truncated for GitHub's PR-body size limit -- see "
        f"`{REPORT_FILENAME}` in this PR's Files Changed for the rest] ...\n"
    )


def build_pr_summary_markdown(
    unmapped: list, arch_candidates: list, pattern_reports: list, gguf_version: str,
    *, full_backlog: bool = False,
) -> str:
    """The PR BODY: short by construction (counts, one line per architecture,
    the control-point warning, a link to the full report file). The full
    per-architecture checklists, derived diffs, and exclusion reasoning live
    in build_report_markdown's output, committed to REPORT_FILENAME instead
    of stuffed into the body -- see the module comment on why."""
    distinct, conflicts = resolve_candidates(pattern_reports)

    lines = []
    lines.append("# Upstream sync: draft patch")
    lines.append("")
    target_note = (
        "the full unmapped-architecture backlog (`--full-backlog`)"
        if full_backlog else "architectures newly appeared since the last "
        "accepted baseline"
    )
    lines.append(
        f"Auto-generated. Target: {target_note}. Full derivation, exclusion "
        f"reasoning, and per-architecture checklists: **`{REPORT_FILENAME}`** "
        "in this PR's Files Changed."
    )
    lines.append("")

    resolvable = [c for c in arch_candidates if c.arch_string is not None]
    new_arch_lines = [c for c in resolvable if not c.already_present]
    n_arch_archs = len(sorted({c.enum_name for c in new_arch_lines}))
    pat_archs = sorted({name for rp in distinct for name in rp.enum_names})

    lines.append("## Summary")
    lines.append(f"- {len(unmapped)} architecture(s) considered")
    lines.append(f"- {len(new_arch_lines)} `arch_map` line(s) across {n_arch_archs} architecture(s)")
    lines.append(
        f"- {len(distinct)} `_HF_TO_GGUF_PATTERNS` line(s) across "
        f"{len(pat_archs)} architecture(s) (deduped)"
    )
    if conflicts:
        lines.append(f"- {len(conflicts)} conflict(s) excluded -- see report file")
    lines.append("")

    lines.append("## Per-architecture status")
    for enum_name in unmapped:
        cands = [c for c in arch_candidates if c.enum_name == enum_name]
        report = next((r for r in pattern_reports if r.enum_name == enum_name), None)
        n_pat = len([rp for rp in distinct if enum_name in rp.enum_names])
        if not cands:
            lines.append(f"- `{enum_name}`: already mapped under a different key")
            continue
        arch_string = next((c.arch_string for c in cands if c.arch_string), None)
        if arch_string is None:
            lines.append(f"- `{enum_name}`: unresolved against installed gguf")
            continue
        already = all(c.already_present for c in cands)
        n_arch = 0 if already else len({c.model_type for c in cands})
        excluded_bits = []
        if report and report.vision_types:
            excluded_bits.append("vision excluded")
        if report and report.moe_expert_types:
            excluded_bits.append("MoE-expert excluded")
        if report and (report.rope_buffer_types or report.over_generic_examples or report.ambiguous_types):
            excluded_bits.append("some kinds punted to manual checklist")
        suffix = f" ({'; '.join(excluded_bits)})" if excluded_bits else ""
        lines.append(
            f"- `{enum_name}` -> `{arch_string}`: {n_arch} arch_map line(s), "
            f"{n_pat} pattern line(s){suffix}"
        )
    lines.append("")

    lines.append("## Why merging is the control point")
    lines.append(
        "This bot never merges its own PR. Every candidate here is a NAME "
        "mapping derived from a static registry table, never executed "
        "against a real checkpoint:"
    )
    lines.append("")
    lines.append("> " + QWEN35_INCIDENT_QUOTE.replace("\n", "\n> "))
    lines.append("")
    lines.append(
        f"See `{REPORT_FILENAME}` for the per-architecture verification "
        "checklist (real-model conversion parity is NOT verified for any "
        "of these) before merging."
    )

    return _truncate_for_pr_body("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--json", metavar="PATH",
        help="drift report (tools/check_upstream_drift.py --json output); "
             "defaults to running that script fresh against the current repo",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true",
        help="write derived arch_map / _HF_TO_GGUF_PATTERNS lines into "
             "magicquant/gguf/source.py, then print the PR-body report",
    )
    mode.add_argument(
        "--check", action="store_true",
        help="exit 1 if there is derivable work, 0 otherwise; no writes",
    )
    ap.add_argument(
        "--full-backlog", action="store_true",
        help="target the WHOLE unmapped-architecture backlog "
             "(check_upstream_drift.py's unmapped_architectures) instead of "
             "just newly-appeared-since-baseline drift (newly_appeared_architectures, "
             "the default). For a deliberate, one-time human-run clearing "
             "pass ONLY -- the automated workflow must never pass this; it "
             "would redraft the entire ~70-architecture backlog every run "
             "even on a week with zero new upstream drift.",
    )
    args = ap.parse_args(argv)

    try:
        drift = _load_drift_report(args.json)
    except Exception as exc:
        print(f"cannot load drift report: {exc}", file=sys.stderr)
        return 2

    try:
        gc_module, tm_cls = _load_installed_gguf()
    except ImportError:
        print("cannot check: the `gguf` package is not installed", file=sys.stderr)
        return 2
    gguf_version = drift.get("gguf_version", "unknown")

    drift_key = "unmapped_architectures" if args.full_backlog else "newly_appeared_architectures"
    target_archs = sorted(drift.get(drift_key, []))
    source_text = SOURCE_PY.read_text()
    _, existing_pairs = _parse_arch_map(source_text)
    existing_pattern_sources = _existing_pattern_sources(source_text)

    arch_candidates = derive_arch_map_candidates(target_archs, existing_pairs, gc_module)
    pattern_reports = [
        derive_pattern_report(name, existing_pattern_sources, gc_module, tm_cls)
        for name in target_archs
    ]
    distinct, conflicts = resolve_candidates(pattern_reports)

    has_work = (
        any(c.arch_string is not None and not c.already_present for c in arch_candidates)
        or bool(distinct)
    )

    if args.check:
        n_arch = sum(
            1 for c in arch_candidates if c.arch_string is not None and not c.already_present
        )
        print(
            f"{'drift' if has_work else 'no drift'} "
            f"({'full backlog' if args.full_backlog else 'delta vs baseline'}): "
            f"{n_arch} arch_map candidate line(s), {len(distinct)} distinct "
            f"pattern candidate line(s) ({len(conflicts)} conflict(s) excluded) "
            f"across {len(target_archs)} architecture(s)",
            file=sys.stderr,
        )
        return 1 if has_work else 0

    if args.apply:
        result = apply_changes(SOURCE_PY, arch_candidates, pattern_reports)
        full_report = build_report_markdown(
            target_archs, arch_candidates, pattern_reports, gguf_version,
            full_backlog=args.full_backlog,
        )
        report_path = REPO / REPORT_FILENAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(full_report)
        summary_md = build_pr_summary_markdown(
            target_archs, arch_candidates, pattern_reports, gguf_version,
            full_backlog=args.full_backlog,
        )
        print(summary_md)
        print(
            f"applied: {len(result.arch_lines_written)} arch_map line(s), "
            f"{len(result.pattern_lines_written)} pattern line(s) written to "
            f"{SOURCE_PY}; full report written to {report_path} "
            f"({len(full_report)} chars); PR body {len(summary_md)} chars",
            file=sys.stderr,
        )
        return 0

    report_md = build_report_markdown(
        target_archs, arch_candidates, pattern_reports, gguf_version,
        full_backlog=args.full_backlog,
    )
    print(report_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
