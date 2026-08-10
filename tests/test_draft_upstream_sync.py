"""tools/draft_upstream_sync.py -- golden-standard tests.

The centerpiece is the MUSE_GLIMMER fixture: a fictional-but-realistic
architecture (never a real llama.cpp arch) with known facts baked in by this
spec --

    model_type "muse_glimmer", nested text_config.model_type "muse_glimmer_text",
    novel decoder stems self_attn.gate_proj + pre_feedforward_layernorm +
    post_feedforward_layernorm, vision prefixes model.vision_tower /
    vision_adapter / vision_projection, expected GGUF arch string
    "muse-glimmer".

Hermetic-fixture design decision (see module docstring in draft_upstream_sync.py
for why "translate registry names into model_type keys" is a disclosed
heuristic, not a guarantee): the generator's arch-string resolution and
tensor-candidate derivation are pure functions parameterized on a
"gguf-constants-like" module and a "TensorNameMap-like" class, both passed in
explicitly rather than imported at call time. Production (the CLI) passes the
REAL installed `gguf` package. This test tries the real package FIRST --
`MUSE_GLIMMER in gguf.constants.MODEL_ARCH.__members__` -- and only falls back
to a hand-recorded fixture (`_FAKE_GC` / `_FAKE_TM` below) that mimics the
exact shape of gguf.constants/gguf.tensor_mapping's static tables when the
installed gguf predates muse-glimmer (true as of gguf 0.18.0, verified while
building this fixture). This is the ONLY way to make the test both:
  - hermetic: it does not silently start failing (or silently start passing
    a previously-untested code path) when `gguf` is upgraded on some future
    CI run or dev machine;
  - deterministic: the recorded rows below are frozen, so the assertions
    about which stems are "novel" vs "already covered" cannot drift out
    from under the test as source.py's real _HF_TO_GGUF_PATTERNS evolves --
    the "existing patterns" side of that comparison is ALSO frozen (see
    _EXISTING_PATTERNS below, copied verbatim from source.py's real
    llama-style entries at the time this test was written) rather than read
    live from the file.
If some future gguf release genuinely adds MUSE_GLIMMER (vanishingly
unlikely for a fictional name, but the mechanism is real), the test
transparently switches to exercising the real registry instead -- the
assertions are about the OUTPUT shape, not about which code path produced it.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO / "tools"


def _load_module():
    """Import tools/draft_upstream_sync.py by path (tools/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "draft_upstream_sync", TOOLS_DIR / "draft_upstream_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["draft_upstream_sync"] = mod
    spec.loader.exec_module(mod)
    return mod


d = _load_module()


# ---------------------------------------------------------------------------
# The MUSE_GLIMMER fixture
# ---------------------------------------------------------------------------

# Frozen subset of magicquant/gguf/source.py's REAL _HF_TO_GGUF_PATTERNS regex
# sources (copied verbatim into _FIXTURE_SOURCE below, not read from the real
# file live -- see module docstring). Just enough to exercise the "already
# covered by the mainstream llama-style convention" dedup path for the
# standard tensor kinds muse_glimmer also has (embeddings, output, QKVO
# attention, FFN up/gate/down, attn norm). Derived FROM _FIXTURE_SOURCE
# (rather than duplicated by hand) so the two can never drift apart -- the
# apply-mechanics tests below re-extract "what's existing" straight from the
# file after each write, and a hand-duplicated list here would silently
# desync from that and make the idempotency test flaky.

_MUSE_GLIMMER = "MUSE_GLIMMER"

# MODEL_TENSORS[MUSE_GLIMMER]: standard decoder kinds (all already coverable
# via _EXISTING_PATTERNS above) + the three spec-mandated novel kinds + three
# vision kinds (one per spec-mandated prefix: vision_tower/vision_adapter/
# vision_projection).
_MUSE_GLIMMER_TENSOR_TYPES = [
    "TOKEN_EMBD", "OUTPUT", "OUTPUT_NORM",
    "ATTN_NORM", "ATTN_Q", "ATTN_K", "ATTN_V", "ATTN_OUT",
    "FFN_GATE", "FFN_UP", "FFN_DOWN",
    "ATTN_GATE", "FFN_PRE_NORM", "FFN_POST_NORM",           # novel decoder stems
    "V_ENC_TOWER", "V_MM_ADAPTER", "V_MM_PROJECTION",        # vision prefixes
]

_FAKE_GC = types.SimpleNamespace(
    MODEL_ARCH=types.SimpleNamespace(__members__={_MUSE_GLIMMER: _MUSE_GLIMMER}),
    MODEL_ARCH_NAMES={_MUSE_GLIMMER: "muse-glimmer"},
    MODEL_TENSORS={_MUSE_GLIMMER: _MUSE_GLIMMER_TENSOR_TYPES},
    TENSOR_NAMES={
        "TOKEN_EMBD": "token_embd",
        "OUTPUT": "output",
        "OUTPUT_NORM": "output_norm",
        "ATTN_NORM": "blk.{bid}.attn_norm",
        "ATTN_Q": "blk.{bid}.attn_q",
        "ATTN_K": "blk.{bid}.attn_k",
        "ATTN_V": "blk.{bid}.attn_v",
        "ATTN_OUT": "blk.{bid}.attn_output",
        "FFN_GATE": "blk.{bid}.ffn_gate",
        "FFN_UP": "blk.{bid}.ffn_up",
        "FFN_DOWN": "blk.{bid}.ffn_down",
        "ATTN_GATE": "blk.{bid}.attn_gate",
        "FFN_PRE_NORM": "blk.{bid}.ffn_norm",
        "FFN_POST_NORM": "blk.{bid}.post_ffw_norm",
        "V_ENC_TOWER": "v.blk.{bid}.tower",
        "V_MM_ADAPTER": "mm.adapter",
        "V_MM_PROJECTION": "mm.projection",
    },
)

_FAKE_TM = types.SimpleNamespace(
    mappings_cfg={
        "TOKEN_EMBD": ("model.embed_tokens",),
        "OUTPUT": ("lm_head",),
        "OUTPUT_NORM": ("model.norm",),
        "V_MM_ADAPTER": ("model.vision_adapter",),
        "V_MM_PROJECTION": ("model.vision_projection",),
    },
    block_mappings_cfg={
        "ATTN_NORM": ("model.layers.{bid}.input_layernorm",),
        "ATTN_Q": ("model.layers.{bid}.self_attn.q_proj",),
        "ATTN_K": ("model.layers.{bid}.self_attn.k_proj",),
        "ATTN_V": ("model.layers.{bid}.self_attn.v_proj",),
        "ATTN_OUT": ("model.layers.{bid}.self_attn.o_proj",),
        "FFN_GATE": ("model.layers.{bid}.mlp.gate_proj",),
        "FFN_UP": ("model.layers.{bid}.mlp.up_proj",),
        "FFN_DOWN": ("model.layers.{bid}.mlp.down_proj",),
        "ATTN_GATE": ("model.layers.{bid}.self_attn.gate_proj",),
        "FFN_PRE_NORM": ("model.layers.{bid}.pre_feedforward_layernorm",),
        "FFN_POST_NORM": ("model.layers.{bid}.post_feedforward_layernorm",),
        "V_ENC_TOWER": ("model.vision_tower.{bid}.block",),
    },
    arch_block_mappings_cfg={},
)


def _gguf_modules_for_muse_glimmer():
    """Real gguf if it happens to know muse-glimmer; recorded fixture otherwise.

    See module docstring: this is the hermetic/deterministic design decision,
    checked FIRST against the real installed package so the test would
    transparently exercise the real registry if it were ever to actually gain
    this (fictional) architecture.
    """
    try:
        real_gc, real_tm = d._load_installed_gguf()
    except ImportError:
        pytest.skip("gguf package not installed")
    if _MUSE_GLIMMER in real_gc.MODEL_ARCH.__members__:
        return real_gc, real_tm
    return _FAKE_GC, _FAKE_TM


# ---------------------------------------------------------------------------
# The GLACIER_FORGE fixture: a SECOND fictional architecture, independent of
# MUSE_GLIMMER, purpose-built to pin the hygiene rules a single-fixture suite
# cannot exercise:
#   - a stacked-MoE-expert kind (FFN_GATE_EXP) -- MUSE_GLIMMER declares none,
#     which is exactly how the original 17-test suite stayed all-green after
#     disabling the _is_stacked_moe_expert call site in derive_pattern_report:
#     nothing in that suite ever reached the code the guard protects;
#   - a rope/inv_freq buffer kind (ROPE_FREQS);
#   - an over-generic bare non-block candidate (GLACIER_BARE);
#   - a kind with more than _MAX_SURVIVING_SPELLINGS candidate spellings
#     (GLACIER_MANY_SPELLINGS);
#   - a genuine cross-architecture DUPLICATE of MUSE_GLIMMER's FFN_PRE_NORM
#     candidate (identical regex AND replacement -- resolve_candidates must
#     collapse the two reports' entries into one, credited to both archs);
#   - a genuine cross-architecture CONFLICT with MUSE_GLIMMER's ATTN_GATE
#     candidate (IDENTICAL regex, from the identical HF template
#     "self_attn.gate_proj", but a DIFFERENT GGUF target) -- this is the real
#     shape of the WAVTOKENIZER_DEC `backbone.posnet` bug the reviewer found.
# ---------------------------------------------------------------------------

_GLACIER_FORGE = "GLACIER_FORGE"

_GLACIER_TENSOR_TYPES = [
    "TOKEN_EMBD", "OUTPUT", "OUTPUT_NORM",       # standard -> silently skipped
    "FFN_PRE_NORM",                               # duplicate of MUSE's
    "GLACIER_GATE_CONFLICT",                      # conflicts with MUSE's ATTN_GATE
    "ROPE_FREQS",                                 # rope-buffer exclusion
    "GLACIER_BARE",                               # over-generic-anchor exclusion
    "GLACIER_MANY_SPELLINGS",                     # >2-spellings exclusion
    "FFN_GATE_EXP",                               # stacked-MoE-expert exclusion
]

_FAKE_GC_GLACIER = types.SimpleNamespace(
    MODEL_ARCH=types.SimpleNamespace(__members__={_GLACIER_FORGE: _GLACIER_FORGE}),
    MODEL_ARCH_NAMES={_GLACIER_FORGE: "glacier-forge"},
    MODEL_TENSORS={_GLACIER_FORGE: _GLACIER_TENSOR_TYPES},
    TENSOR_NAMES={
        "TOKEN_EMBD": "token_embd",
        "OUTPUT": "output",
        "OUTPUT_NORM": "output_norm",
        "FFN_PRE_NORM": "blk.{bid}.ffn_norm",                    # == MUSE's target
        "GLACIER_GATE_CONFLICT": "blk.{bid}.glacier_alt_gate",   # != MUSE's attn_gate
        "ROPE_FREQS": "rope_freqs",
        "GLACIER_BARE": "glacier_bare",
        "GLACIER_MANY_SPELLINGS": "blk.{bid}.glacier_many",
        "FFN_GATE_EXP": "blk.{bid}.ffn_gate_exps",   # real gguf stacked-expert target
    },
)

_FAKE_TM_GLACIER = types.SimpleNamespace(
    mappings_cfg={
        "TOKEN_EMBD": ("model.embed_tokens",),
        "OUTPUT": ("lm_head",),
        "OUTPUT_NORM": ("model.norm",),
        "GLACIER_BARE": ("glacier_bare_name",),   # non-block, no '.' -> over-generic
    },
    block_mappings_cfg={
        "FFN_PRE_NORM": ("model.layers.{bid}.pre_feedforward_layernorm",),
        "GLACIER_GATE_CONFLICT": ("model.layers.{bid}.self_attn.gate_proj",),
        "ROPE_FREQS": ("model.layers.{bid}.self_attn.rotary_emb.inv_freq",),
        "GLACIER_MANY_SPELLINGS": (
            "model.layers.{bid}.glacier.alpha",
            "model.layers.{bid}.glacier.beta",
            "model.layers.{bid}.glacier.gamma",
        ),
        "FFN_GATE_EXP": ("model.layers.{bid}.mlp.experts.gate_proj",),
    },
    arch_block_mappings_cfg={},
)


def _gguf_modules_for_glacier_forge():
    return _FAKE_GC_GLACIER, _FAKE_TM_GLACIER


# ---------------------------------------------------------------------------
# arch_map candidates
# ---------------------------------------------------------------------------

def test_muse_glimmer_arch_map_candidates():
    gc_module, _ = _gguf_modules_for_muse_glimmer()
    candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)

    rendered = {f'"{c.model_type}": "{c.arch_string}"' for c in candidates}
    assert '"muse_glimmer": "muse-glimmer"' in rendered
    assert '"muse_glimmer_text": "muse-glimmer"' in rendered
    assert all(c.arch_string == "muse-glimmer" for c in candidates)
    assert all(not c.already_present for c in candidates)


def test_already_mapped_arch_excluded_even_under_different_key():
    """An arch already reachable via SOME key in arch_map is excluded, even
    when the candidate's own guessed key differs from the real one -- this is
    the qwen3_5-style case: enum QWEN35's value 'qwen35' is already mapped,
    but under keys 'qwen3_5'/'qwen3_5_text', not a lower()'d 'qwen35'."""
    gc_module, _ = _gguf_modules_for_muse_glimmer()
    existing_pairs = [("some_other_key", "muse-glimmer", 0)]
    candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], existing_pairs, gc_module)
    assert candidates == []


def test_no_drift_arch_map_is_noop():
    gc_module, _ = _gguf_modules_for_muse_glimmer()
    assert d.derive_arch_map_candidates([], [], gc_module) == []


def test_report_distinguishes_already_mapped_from_unresolvable():
    """Regression: an enum that produces ZERO arch_map candidates (because
    derive_arch_map_candidates found it already mapped under a different key
    -- only reachable with a stale --json, see that function's docstring)
    must not be reported as "unresolvable against the installed gguf", which
    would be a different and misleading claim."""
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    existing_pairs = [("some_other_key", "muse-glimmer", 0)]
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], existing_pairs, gc_module)
    assert arch_candidates == []  # precondition for the case under test

    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, [], gc_module, tm_cls)]
    md = d.build_report_markdown([_MUSE_GLIMMER], arch_candidates, pattern_reports, gguf_version="test")
    assert "already mapped under a different key" in md
    assert "not resolvable against the currently-installed gguf" not in md


# ---------------------------------------------------------------------------
# _HF_TO_GGUF_PATTERNS candidates
# ---------------------------------------------------------------------------

def test_muse_glimmer_novel_stems_are_proposed():
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    report = d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)

    assert report.arch_known
    assert report.arch_string == "muse-glimmer"

    proposed_regex = {pc.regex_source for pc in report.proposed}
    assert r"^model\.layers\.(\d+)\.self_attn\.gate_proj\.weight$" in proposed_regex
    assert r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$" in proposed_regex
    assert r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$" in proposed_regex

    # Replacement side must target the real GGUF names for these kinds.
    by_regex = {pc.regex_source: pc for pc in report.proposed}
    gate = by_regex[r"^model\.layers\.(\d+)\.self_attn\.gate_proj\.weight$"]
    assert gate.replacement_source == 'lambda m: f"blk.{m.group(1)}.attn_gate.weight"'
    pre_norm = by_regex[r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$"]
    assert pre_norm.replacement_source == 'lambda m: f"blk.{m.group(1)}.ffn_norm.weight"'
    post_norm = by_regex[r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$"]
    assert post_norm.replacement_source == 'lambda m: f"blk.{m.group(1)}.post_ffw_norm.weight"'


def test_muse_glimmer_standard_stems_are_not_reproposed():
    """Kinds already coverable by the mainstream llama-style convention
    (embeddings, output, QKVO attention, FFN up/gate/down, attn norm) are
    silently skipped, not re-derived -- this is the noise-control rule: dump
    the full candidate list only for genuinely novel kinds."""
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    report = d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)

    proposed_types = {pc.tensor_type for pc in report.proposed}
    for standard in (
        "TOKEN_EMBD", "OUTPUT", "OUTPUT_NORM", "ATTN_Q", "ATTN_K", "ATTN_V",
        "ATTN_OUT", "FFN_GATE", "FFN_UP", "FFN_DOWN", "ATTN_NORM",
    ):
        assert standard not in proposed_types, (
            f"{standard} should have been silently skipped (already covered)"
        )
    assert set(report.skipped_types) == {
        "TOKEN_EMBD", "OUTPUT", "OUTPUT_NORM", "ATTN_Q", "ATTN_K", "ATTN_V",
        "ATTN_OUT", "FFN_GATE", "FFN_UP", "FFN_DOWN", "ATTN_NORM",
    }


def test_muse_glimmer_vision_prefixes_excluded_and_reported():
    """The three spec-mandated vision prefixes never leak into the derived
    pattern diff -- they go to the manual checklist (vision_types /
    vision_hf_examples) instead."""
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    report = d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)

    for pc in report.proposed:
        assert "vision" not in pc.hf_template.lower()

    assert set(report.vision_types) == {"V_ENC_TOWER", "V_MM_ADAPTER", "V_MM_PROJECTION"}
    examples = " ".join(report.vision_hf_examples)
    assert "model.vision_tower" in examples
    assert "model.vision_adapter" in examples
    assert "model.vision_projection" in examples

    # And the markdown report surfaces them for a human to see.
    md = d.build_report_markdown(
        [_MUSE_GLIMMER],
        d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module),
        [report],
        gguf_version="test",
    )
    assert "model.vision_tower" in md
    assert "model.vision_adapter" in md
    assert "model.vision_projection" in md
    assert "MUSE_GLIMMER" in md


def test_no_drift_pattern_report_is_noop():
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    report = d.derive_pattern_report("SOME_UNKNOWN_ENUM_NAME_XYZ", [], gc_module, tm_cls)
    assert report.arch_known is False
    assert report.proposed == []


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_generator_is_deterministic():
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()

    def _run():
        arch = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
        pat = d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)
        md = d.build_report_markdown([_MUSE_GLIMMER], arch, [pat], gguf_version="test")
        return (
            [(c.model_type, c.arch_string) for c in arch],
            [(pc.regex_source, pc.replacement_source) for pc in pat.proposed],
            md,
        )

    first = _run()
    second = _run()
    assert first == second


# ---------------------------------------------------------------------------
# --apply mechanics: alphabetical placement + idempotency, against a small
# throwaway copy of the real source.py so file-writing is exercised for
# real without touching the actual repo file.
# ---------------------------------------------------------------------------

_FIXTURE_SOURCE = '''\
def _build_gguf_metadata_from_config(config):
    arch_map = {
        "arctic": "arctic", "baichuan": "baichuan", "bloom": "bloom",
        "gemma3": "gemma3", "glm4": "glm4", "gpt2": "gpt2",
        "qwen3_5": "qwen35", "qwen3_5_text": "qwen35",
        "starcoder": "starcoder", "starcoder2": "starcoder2",
    }
    arch = arch_map.get(config.get("model_type"))
    return arch


_HF_TO_GGUF_PATTERNS = [
    # Embeddings
    (r"^model\\.embed_tokens\\.weight$",              "token_embd.weight"),
    # Output head
    (r"^lm_head\\.weight$",                          "output.weight"),
    # Final norm
    (r"^model\\.norm\\.weight$",                       "output_norm.weight"),
    # Per-layer attention
    (r"^model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_q.weight"),
    (r"^model\\.layers\\.(\\d+)\\.self_attn\\.k_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_k.weight"),
    (r"^model\\.layers\\.(\\d+)\\.self_attn\\.v_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_v.weight"),
    (r"^model\\.layers\\.(\\d+)\\.self_attn\\.o_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_output.weight"),
    # Per-layer FFN
    (r"^model\\.layers\\.(\\d+)\\.mlp\\.up_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_up.weight"),
    (r"^model\\.layers\\.(\\d+)\\.mlp\\.gate_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_gate.weight"),
    (r"^model\\.layers\\.(\\d+)\\.mlp\\.down_proj\\.weight$",
     lambda m: f"blk.{m.group(1)}.ffn_down.weight"),
    # Layer norms
    (r"^model\\.layers\\.(\\d+)\\.input_layernorm\\.weight$",
     lambda m: f"blk.{m.group(1)}.attn_norm.weight"),
]

_HF_TO_GGUF_COMPILED = [(__import__("re").compile(p), r) for p, r in _HF_TO_GGUF_PATTERNS]
'''


@pytest.fixture()
def fixture_source_path(tmp_path):
    p = tmp_path / "source.py"
    p.write_text(_FIXTURE_SOURCE)
    return p


# Derived from _FIXTURE_SOURCE itself (see comment above) so the "what
# already exists" list used by the non-apply pattern-report tests can never
# desync from what the apply-mechanics tests' fixture file actually contains.
_EXISTING_PATTERNS = d._existing_pattern_sources(_FIXTURE_SOURCE)


def test_apply_inserts_arch_map_line_alphabetically(fixture_source_path):
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, [], gc_module, tm_cls)]

    result = d.apply_changes(fixture_source_path, arch_candidates, pattern_reports)
    assert len(result.arch_lines_written) == 2  # muse_glimmer + muse_glimmer_text

    text = fixture_source_path.read_text()
    assert '"muse_glimmer": "muse-glimmer",' in text
    assert '"muse_glimmer_text": "muse-glimmer",' in text
    # alphabetically: "gpt2" < "muse_glimmer" < "muse_glimmer_text" < "qwen3_5"
    gpt2_pos = text.index('"gpt2"')
    muse_pos = text.index('"muse_glimmer"')
    muse_text_pos = text.index('"muse_glimmer_text"')
    qwen_pos = text.index('"qwen3_5"')
    assert gpt2_pos < muse_pos < muse_text_pos < qwen_pos

    # File must still be valid, parseable Python.
    compile(text, str(fixture_source_path), "exec")


def test_apply_inserts_pattern_lines(fixture_source_path):
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)]
    result = d.apply_changes(fixture_source_path, [], pattern_reports)
    assert result.pattern_lines_written  # at least the 3 novel stems

    text = fixture_source_path.read_text()
    assert r"self_attn\.gate_proj" in text
    assert "attn_gate" in text
    compile(text, str(fixture_source_path), "exec")


def test_apply_is_idempotent(fixture_source_path):
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)]

    first = d.apply_changes(fixture_source_path, copy.deepcopy(arch_candidates), pattern_reports)
    assert first.arch_lines_written
    text_after_first = fixture_source_path.read_text()

    # Re-derive candidates against the NOW-MODIFIED file (mirrors how the CLI
    # re-parses source.py fresh on every invocation) and apply again.
    _, existing_pairs = d._parse_arch_map(text_after_first)
    existing_patterns = d._existing_pattern_sources(text_after_first)
    arch_candidates_2 = d.derive_arch_map_candidates([_MUSE_GLIMMER], existing_pairs, gc_module)
    pattern_reports_2 = [
        d.derive_pattern_report(_MUSE_GLIMMER, existing_patterns, gc_module, tm_cls)
    ]
    second = d.apply_changes(fixture_source_path, arch_candidates_2, pattern_reports_2)
    assert second.arch_lines_written == []
    assert second.pattern_lines_written == []

    text_after_second = fixture_source_path.read_text()
    assert text_after_first == text_after_second


def test_apply_no_drift_leaves_file_untouched(fixture_source_path):
    original = fixture_source_path.read_text()
    result = d.apply_changes(fixture_source_path, [], [])
    assert result.arch_lines_written == []
    assert result.pattern_lines_written == []
    assert fixture_source_path.read_text() == original


# ---------------------------------------------------------------------------
# Report content: control-point language
# ---------------------------------------------------------------------------

def test_report_quotes_qwen35_incident_and_control_points():
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)]
    md = d.build_report_markdown([_MUSE_GLIMMER], arch_candidates, pattern_reports, gguf_version="test")

    assert "qwen3_5 uniform-logits incident" in md
    assert "64% of tensor VALUES were wrong" in md
    assert "NOT verified" in md
    assert "keep-or-delete" in md


# ---------------------------------------------------------------------------
# Regex/replacement rendering sanity (unit-level, no fixture needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hf_template,gguf_template,expect_regex,expect_is_block",
    [
        (
            "model.layers.{bid}.self_attn.gate_proj", "blk.{bid}.attn_gate",
            r"^model\.layers\.(\d+)\.self_attn\.gate_proj\.weight$", True,
        ),
        ("lm_head", "output", r"^lm_head\.weight$", False),
    ],
)
def test_regex_and_replacement_rendering(hf_template, gguf_template, expect_regex, expect_is_block):
    regex_source, repl_source, is_block = d._regex_and_replacement(hf_template, gguf_template)
    assert regex_source == expect_regex
    assert is_block == expect_is_block
    if is_block:
        assert repl_source == 'lambda m: f"blk.{m.group(1)}.attn_gate.weight"'
    else:
        assert repl_source == repr("output.weight")


def test_is_stacked_moe_expert_excludes_plural_exps_only():
    assert d._is_stacked_moe_expert("blk.{bid}.ffn_gate_exps") is True
    assert d._is_stacked_moe_expert("blk.{bid}.ffn_gate_chexps") is True
    assert d._is_stacked_moe_expert("blk.{bid}.ffn_gate_shexp") is False
    assert d._is_stacked_moe_expert("blk.{bid}.exp_probs_b") is False


# ---------------------------------------------------------------------------
# Candidate hygiene (spec fixes item 4): rope-buffer exclusion, over-generic
# anchors, the >2-spellings ambiguity threshold -- all via GLACIER_FORGE.
# ---------------------------------------------------------------------------

def test_rope_buffer_kind_excluded_entirely():
    gc_module, tm_cls = _gguf_modules_for_glacier_forge()
    report = d.derive_pattern_report(_GLACIER_FORGE, [], gc_module, tm_cls)

    assert "ROPE_FREQS" not in {pc.tensor_type for pc in report.proposed}
    assert "ROPE_FREQS" in report.rope_buffer_types


def test_over_generic_bare_anchor_excluded():
    gc_module, tm_cls = _gguf_modules_for_glacier_forge()
    report = d.derive_pattern_report(_GLACIER_FORGE, [], gc_module, tm_cls)

    assert "GLACIER_BARE" not in {pc.tensor_type for pc in report.proposed}
    assert ("GLACIER_BARE", "glacier_bare_name") in report.over_generic_examples
    # Sanity: a real over-generic example from the spec itself.
    assert d._is_over_generic_anchor("classifier", is_block=False) is True
    assert d._is_over_generic_anchor("dense", is_block=False) is True
    # Block-scoped (has layer capture) is never over-generic regardless of dots.
    assert d._is_over_generic_anchor("dense", is_block=True) is False
    # A dotted (nested) non-block name is not bare.
    assert d._is_over_generic_anchor("model.norm", is_block=False) is False


def test_ambiguous_spellings_threshold_excludes_whole_kind():
    gc_module, tm_cls = _gguf_modules_for_glacier_forge()
    report = d.derive_pattern_report(_GLACIER_FORGE, [], gc_module, tm_cls)

    assert "GLACIER_MANY_SPELLINGS" not in {pc.tensor_type for pc in report.proposed}
    assert "GLACIER_MANY_SPELLINGS" in report.ambiguous_types
    assert len(report.ambiguous_types["GLACIER_MANY_SPELLINGS"]) == 3
    assert len(report.ambiguous_types["GLACIER_MANY_SPELLINGS"]) > d._MAX_SURVIVING_SPELLINGS


# ---------------------------------------------------------------------------
# MoE-expert exclusion, mutation-verified (spec fixes item 6): the ORIGINAL
# 17-test suite stayed all-green with the _is_stacked_moe_expert call site
# deleted from derive_pattern_report, because MUSE_GLIMMER declares no _exps
# kind at all -- the exclusion was untested, not merely unexercised-by-luck.
# GLACIER_FORGE declares FFN_GATE_EXP specifically so this gap can't recur.
# ---------------------------------------------------------------------------

def test_moe_expert_exclusion_is_actually_wired(monkeypatch):
    gc_module, tm_cls = _gguf_modules_for_glacier_forge()

    # 1) With the real guard active, the stacked-expert kind never reaches
    #    `proposed` -- it's excluded and separately bucketed.
    report = d.derive_pattern_report(_GLACIER_FORGE, [], gc_module, tm_cls)
    assert "FFN_GATE_EXP" not in {pc.tensor_type for pc in report.proposed}
    assert "FFN_GATE_EXP" in report.moe_expert_types

    # 2) Mutation check: simulate the call-site guard being disabled/deleted.
    #    If this assertion still passed unchanged, the fixture would provide
    #    NO coverage of the exclusion -- which is exactly what happened with
    #    MUSE_GLIMMER-only (it has no _exps kind, so removing the guard
    #    couldn't have changed its test results). Here it must flip.
    monkeypatch.setattr(d, "_is_stacked_moe_expert", lambda gguf_template: False)
    mutated = d.derive_pattern_report(_GLACIER_FORGE, [], gc_module, tm_cls)
    assert "FFN_GATE_EXP" in {pc.tensor_type for pc in mutated.proposed}, (
        "the mutation (guard disabled) did not change the result -- the "
        "exclusion is not actually load-bearing for this fixture"
    )


# ---------------------------------------------------------------------------
# Cross-architecture resolution (spec fixes item 4a/4b): dedup identical
# candidates, quarantine conflicting ones. Exercised across the two
# INDEPENDENT fixtures (MUSE_GLIMMER, GLACIER_FORGE) together.
# ---------------------------------------------------------------------------

def _muse_and_glacier_reports():
    muse_gc, muse_tm = _gguf_modules_for_muse_glimmer()
    glacier_gc, glacier_tm = _gguf_modules_for_glacier_forge()
    muse_report = d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, muse_gc, muse_tm)
    glacier_report = d.derive_pattern_report(_GLACIER_FORGE, [], glacier_gc, glacier_tm)
    return muse_report, glacier_report


def test_resolve_candidates_dedupes_identical_cross_arch_candidates():
    muse_report, glacier_report = _muse_and_glacier_reports()
    distinct, _conflicts = d.resolve_candidates([muse_report, glacier_report])

    ffn_pre_norm_regex = r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$"
    matches = [rp for rp in distinct if rp.regex_source == ffn_pre_norm_regex]
    assert len(matches) == 1, "identical (regex, replacement) must collapse to ONE entry"
    assert matches[0].enum_names == sorted([_MUSE_GLIMMER, _GLACIER_FORGE])


def test_resolve_candidates_quarantines_conflicting_targets():
    muse_report, glacier_report = _muse_and_glacier_reports()
    distinct, conflicts = d.resolve_candidates([muse_report, glacier_report])

    gate_regex = r"^model\.layers\.(\d+)\.self_attn\.gate_proj\.weight$"
    # The conflicting regex must NOT appear in `distinct` at all -- neither
    # target wins by sort order or any other implicit tiebreak.
    assert gate_regex not in {rp.regex_source for rp in distinct}

    matching_conflicts = [c for c in conflicts if c.regex_source == gate_regex]
    assert len(matching_conflicts) == 1
    targets = {repl: (names, kind) for repl, names, kind in matching_conflicts[0].targets}
    assert 'lambda m: f"blk.{m.group(1)}.attn_gate.weight"' in targets
    assert 'lambda m: f"blk.{m.group(1)}.glacier_alt_gate.weight"' in targets
    names, kind = targets['lambda m: f"blk.{m.group(1)}.attn_gate.weight"']
    assert names == [_MUSE_GLIMMER]
    assert kind  # the disambiguating tensor kind must be carried on the conflict
    names, kind = targets['lambda m: f"blk.{m.group(1)}.glacier_alt_gate.weight"']
    assert names == [_GLACIER_FORGE]
    assert kind


def test_apply_never_writes_a_conflicting_regex(tmp_path):
    """End-to-end: apply_changes on both fixtures together must write the
    deduped FFN_PRE_NORM line exactly ONCE and must never write either side
    of the self_attn.gate_proj conflict."""
    p = tmp_path / "source.py"
    p.write_text(_FIXTURE_SOURCE)
    muse_report, glacier_report = _muse_and_glacier_reports()

    d.apply_changes(p, [], [muse_report, glacier_report])
    text = p.read_text()

    assert text.count(r"pre_feedforward_layernorm") == 1
    assert "self_attn.gate_proj" not in text.replace("\\", "")  # neither side landed
    compile(text, str(p), "exec")


def test_conflict_report_shows_both_targets():
    muse_report, glacier_report = _muse_and_glacier_reports()
    arch_candidates = []
    md = d.build_report_markdown(
        [_MUSE_GLIMMER, _GLACIER_FORGE], arch_candidates,
        [muse_report, glacier_report], gguf_version="test",
    )
    assert "## Conflicts" in md
    assert "attn_gate" in md
    assert "glacier_alt_gate" in md
    assert _MUSE_GLIMMER in md
    assert _GLACIER_FORGE in md


# ---------------------------------------------------------------------------
# arch_map lines carry the same UNVERIFIED marker pattern lines do (spec
# fixes item 5 -- CHANGELOG previously claimed this was already true).
# ---------------------------------------------------------------------------

def test_applied_arch_map_line_carries_unverified_marker(tmp_path):
    p = tmp_path / "source.py"
    p.write_text(_FIXTURE_SOURCE)
    gc_module, _ = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)

    d.apply_changes(p, arch_candidates, [])
    text = p.read_text()
    assert '"muse_glimmer": "muse-glimmer",  # MUSE_GLIMMER -- draft_upstream_sync, UNVERIFIED' in text
    compile(text, str(p), "exec")


def test_full_report_arch_map_block_shows_unverified_marker():
    gc_module, _ = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
    md = d.build_report_markdown([_MUSE_GLIMMER], arch_candidates, [], gguf_version="test")
    assert "draft_upstream_sync, UNVERIFIED" in md
    # Specifically on an arch_map line, not just somewhere else in the report.
    arch_block = md.split("## Derived: `arch_map`")[1].split("## ")[0]
    assert "UNVERIFIED" in arch_block


# ---------------------------------------------------------------------------
# DELTA vs BACKLOG (spec fixes item 1): default targets
# newly_appeared_architectures; --full-backlog opts into unmapped_architectures.
# Exercised at the main() CLI level with a constructed drift.json -- read-only
# (--check), so it never touches the real magicquant/gguf/source.py.
# ---------------------------------------------------------------------------

def _write_drift_json(tmp_path, unmapped, newly_appeared):
    drift = {
        "unmapped_architectures": unmapped,
        "newly_appeared_architectures": newly_appeared,
        "gguf_version": "test",
    }
    p = tmp_path / "drift.json"
    p.write_text(json.dumps(drift))
    return p


def test_default_mode_targets_the_delta_not_the_backlog(tmp_path, capsys):
    # Real, currently-unmapped architecture names so main()'s internal
    # (always-real) gguf lookups resolve them -- the CLI has no injection
    # point for gc_module/tm_cls, unlike the pure derive_* functions above.
    drift_path = _write_drift_json(tmp_path, ["AFMOE", "BERT", "EXAONE4"], ["EXAONE4"])

    exit_delta = d.main(["--json", str(drift_path), "--check"])
    err_delta = capsys.readouterr().err
    assert "across 1 architecture(s)" in err_delta
    assert "delta vs baseline" in err_delta

    exit_backlog = d.main(["--json", str(drift_path), "--check", "--full-backlog"])
    err_backlog = capsys.readouterr().err
    assert "across 3 architecture(s)" in err_backlog
    assert "full backlog" in err_backlog

    # Both are legitimate exit codes (0 or 1) depending on real repo state;
    # what's pinned here is which TARGET LIST each mode consulted, not
    # whether that list happens to have derivable work today.
    assert exit_delta in (0, 1)
    assert exit_backlog in (0, 1)


def test_default_mode_is_true_noop_when_only_backlog_has_drift(tmp_path):
    """The exact regression the reviewer found: --apply/--check without
    --full-backlog must ignore a non-empty unmapped_architectures backlog
    when newly_appeared_architectures is empty."""
    drift_path = _write_drift_json(tmp_path, ["AFMOE", "BERT"], [])
    assert d.main(["--json", str(drift_path), "--check"]) == 0


# ---------------------------------------------------------------------------
# PR body size (spec fixes item 2): short summary, hard truncation.
# ---------------------------------------------------------------------------

def test_pr_summary_is_short_and_links_to_report_file():
    gc_module, tm_cls = _gguf_modules_for_muse_glimmer()
    arch_candidates = d.derive_arch_map_candidates([_MUSE_GLIMMER], [], gc_module)
    pattern_reports = [d.derive_pattern_report(_MUSE_GLIMMER, _EXISTING_PATTERNS, gc_module, tm_cls)]

    summary = d.build_pr_summary_markdown(
        [_MUSE_GLIMMER], arch_candidates, pattern_reports, gguf_version="test",
    )
    full = d.build_report_markdown(
        [_MUSE_GLIMMER], arch_candidates, pattern_reports, gguf_version="test",
    )
    assert len(summary) < len(full)
    assert len(summary) < 20_000  # nowhere near GitHub's 65536 cap
    assert d.REPORT_FILENAME in summary
    assert "qwen3_5 uniform-logits incident" in summary  # control-point warning kept
    # The full per-architecture checklist prose lives in the report file,
    # not duplicated into the short body.
    assert "real-model conversion parity **NOT verified**" not in summary


def test_truncate_for_pr_body_hard_caps_regardless_of_content():
    oversized = "x" * 200_000
    truncated = d._truncate_for_pr_body(oversized, limit=1000)
    assert len(truncated) <= 1200  # limit + the truncation notice
    assert "truncated" in truncated
    assert d.REPORT_FILENAME in truncated


def test_truncate_for_pr_body_is_noop_under_limit():
    body = "short body\n"
    assert d._truncate_for_pr_body(body, limit=1000) == body
