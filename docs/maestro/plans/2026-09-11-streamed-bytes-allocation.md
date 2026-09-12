# Streamed-Bytes-Aware Allocation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use maestro:subagent-driven-development (recommended) or maestro:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking; if your harness has a native task system, mirror one task per plan task there as well — the plan file remains the durable record.

**Goal:** Give MagicQuant a per-tensor *stream weight* and a bisected Lagrangian streamed-bytes term in the v2 allocator (plus an opt-in v1 speed proxy), so MoE quants stop spending bits on routed experts that are read 3% of the time while the trunk that is read every token sits at BF16.

**Architecture:** A new pure module `magicquant/v2/bandwidth.py` derives weights from group + name + GGUF hparams, accounts streamed bytes and pure loss, and bisects λ against a streamed-bytes budget by repeatedly calling the *unmodified* `allocate()`. `search.py` threads the weights and λ into `_build_units` (defaults keep every existing caller byte-identical), recomputes every published loss as pure κ·werr, and emits an always-present `bandwidth` block. v1 gets `predict_stream_gb` and `--stream-tps`. Nothing touches `allocate.py`, `Allocation.to_json()`, or the distortion-table cache.

**Tech Stack:** Python 3.10, dataclasses, pytest; repo venv at `/server/programming/MagicQuant/.venv` (editable install of the main checkout — see Global Constraints).

**Spec:** `/server/ai/docs/specs/magicquant-bandwidth.md` (revision 3). Section numbers below refer to it.

**Scope.** Tasks 1–8 implement spec milestones M0–M4 (M0 → Tasks 1+2, M1 → Task 3, M2 → Tasks 4+5+6, M3 → Task 7, M4 → Task 8). **M5** (the GPU validation campaign on real models, gates G1–G4) is run separately by the coordinator under `/server/ai/CLAUDE.md` bench discipline after this branch merges; **M6** (the Foundry seam) lives in `/server/programming/Foundry` and is not part of this plan.

**Deliberate refinements of the spec (all sound, recorded so reviewers do not re-litigate them):** (a) `solve_bandwidth` takes the unit builder as a callable instead of importing `search._build_units` — `search.py` imports `bandwidth.py`, so the reverse import would cycle; (b) `_allocate_frontier_and_anchors` returns a 5-tuple (adds `anchor_stats`) and takes `weights_source`, and `_build_and_verify_anchors` consumes `anchor_stats=` — so the anchor loop needs no table/kappa/weights of its own; (c) the reported per-group weight is the **modal** weight over a group's tensors (`exceptions` carries the rest) rather than a byte-weighted mean, which would report a value no tensor has.

## Global Constraints

- Work only in `/server/programming/MagicQuant-bandwidth` (branch `feat/bandwidth-allocation`, base `bf01f41`). **Run every command from that directory**: the venv's editable finder points at `/server/programming/MagicQuant`; the worktree's `magicquant/` shadows it only when the cwd is the worktree. First command of every task: `cd /server/programming/MagicQuant-bandwidth && /server/programming/MagicQuant/.venv/bin/python -c "import magicquant; print(magicquant.__file__)"` must print a path under `MagicQuant-bandwidth`.
- Test command: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/ -q` — **record the count on your own base commit first** (run it once before editing anything) and compare every later run to that; one 2026-09-11 run reported 1269 passed / 20 skipped. Lint (blocking in CI): `/server/programming/MagicQuant/.venv/bin/ruff check --select F magicquant/ tools/ tests/`. Never bare `python`/`pytest`.
- Python 3.10 compatible (no `tomllib`, no `match`, no PEP 604 `X | Y` in runtime annotations without `from __future__ import annotations`).
- **Do not modify** `magicquant/v2/allocate.py`, `Allocation.to_json()`, `magicquant/v2/sensitivity.py`'s cache key or `TABLE_VERSION`, `predict_size`, the `use_bytes_tps` branch, `stream_aware`, `_speed_aware_pick`, or group V's patterns.
- `tests/test_refactor_regression.py` must pass **without** regenerating `tests/fixtures/refactor_regression_seed42.json`.
- Four pinned key sets in `tests/test_v2_search_characterization.py` change deliberately, each by adding one key: the results pin at `238-244` (add `"bandwidth"` after `"final_model"`), the per-anchor pins at `286` and `590` (add `"streamed_bytes"`), and the frontier pin at `330` (add `"lambda"`). `Allocation.to_json()`'s pin at `268-271` and every VALUE pin in that file (assignment, `total_bytes == 6160`, `predicted_loss ≈ 0.07`, `report_fit_affine`) stay as is — that is the λ=0 byte-identity proof.
- Weights are floats in `[0,1]`, always finite; `lam` is a float in loss per streamed GiB; budgets in bytes are `int(gib * 1024**3)`.
- Commit subjects: `type(scope): imperative summary`, scope = module directory (`gguf`, `v2`, `evolution`, `cli`, `docs`), ASCII only, one concern per commit. Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o
  ```
- CHANGELOG entries are written **once, in Task 8** (so Tasks 1–7 never touch `CHANGELOG.md` and can be reviewed independently).

---

### Task 1: redesign.md section 11 + CLAUDE.md test count

**Dispatch:** INDEPENDENT

**Files:**
- Modify: `docs/redesign.md` (append after §10, which ends at the "Why not make it the default immediately" subsection)
- Modify: `CLAUDE.md:28` (the `~1090 tests` figure)

**Interfaces:**
- Produces: nothing consumed by code; Task 8 cross-links §11 from the README.

- [ ] **Step 1: Capture the failing check**

Run: `grep -n "^## 11\." docs/redesign.md`
Expected: no output (section does not exist).

- [ ] **Step 2: Append section 11**

Append this to the end of `docs/redesign.md` (keep §10's heading style; this is the §10 precedent for revising a design decision):

````markdown
## 11. Streamed-bytes addendum: pricing bytes by how often they are read

**Status (2026-09-11):** opt-in (`--budget-bw-gb` / `--bandwidth-weight`),
default off and byte-identical. Revises one item of §9: "speed-aware objectives
beyond byte-pricing" is no longer a non-goal for the single term below.

### The failure

§4.3 says "the byte cost is priced directly by the budget constraint." That is
true when every stored byte is read once per decode token — a dense model. It
is false for a routed MoE: a trunk byte is read every token, a routed-expert
byte `n_used/n_expert` of the time (8/256 on Qwen3.6-35B-A3B and Ornith), an
embedding row almost never. MagicQuant's own shipped MoE quants show the
consequence — per-group accounting from the GGUF headers, `w` = fraction of a
group's bytes touched per token:

| model | quant | trunk bpw | experts bpw | trunk share of traffic | weights GiB/token |
|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B | MagicQuant Q4_K_M (shipped) | 11.86 (head BF16) | 4.25 | 84% | 3.24 |
| Qwen3.6-35B-A3B | MagicQuant ROCmFPX MQ-Q4 (shipped) | 12.69 (head BF16, DeltaNet 16) | 4.50 | 84% | 3.47 |
| Qwen3.6-35B-A3B | unsloth MXFP4_MOE (stock) | 8.81 | 4.71 | 78% | 2.60 |
| Qwen3.8-Flash-Next | unsloth UD-IQ4_XS | 9.08 | 3.94 | 80% | 5.44 |
| gemma-4-26B-A4B | unsloth UD-Q4_K_XL | 4.68 | 4.50 | 55% | 1.65 |
| Qwen3.8-27B (dense) | MagicQuant Q4_K_M | 4.61 | — | 100% | 13.97 |

Same architecture, MagicQuant vs stock: +33% bytes per token. Decode at batch 1
is memory-bandwidth-bound, so that is roughly a 25% decode-speed penalty for a
file that is no smaller. v1's `stream_aware` cannot fix it (a sampling bias
with no cost term — the shipped Ornith run had it on and still chose BF16 for
H/K/O); v2's MCKP cannot either (bytes are priced by storage only).

### The fix: a stream weight and one Lagrangian term

Each tensor gets an architectural weight `w_t ∈ [0,1]` — 1.0 for anything read
in full every token, `expert_used_count / expert_count` for routed experts,
0.0 for row-gathered embeddings (vocab, per-layer, MTP embedding tables) —
read off the GGUF hparams, never measured. The allocator's per-choice loss becomes

```
loss_eff(t, s) = κ_g · ε(t, s) + λ · w_t · bytes(t, s) / 2^30      (λ in loss per streamed GiB)
```

and `allocate()` is unchanged: `Choice.loss` was already opaque to the hull,
greedy and polish. With `--budget-bw-gb B`, λ is found by log-space bisection
so that the realised `Σ w_t · bytes` meets `B` while the storage budget stays a
hard constraint; the smallest feasible λ wins, ties broken by pure κ·ε loss.
With `--bandwidth-weight λ` the term is applied at a fixed λ. Every published
loss (`allocation.predicted_loss`, anchors, the reporting fit) is recomputed as
pure κ·ε so `report_fit_affine` keeps its meaning; `frontier.json`'s point
losses are the effective loss and the file records `lambda`.

### Where this bites: w = 0 groups

The λ term subtracts `λ·w_t/2^30` from every hull-edge slope, so a `w=0` group
(embeddings) is the one place a bandwidth budget does *not* penalise storage.
Combined with the single-group-probe κ_E failure in §10, a large λ can pour
freed storage into embeddings. The bisection returns the smallest feasible λ,
`--floor E=Q6_K` remains the measured guardrail, and `v2_results.json` reports
per-group bpw at λ=0 and at the chosen λ so the shift is visible.

### v1

`--stream-tps` swaps the `use_bytes_tps` proxy's stored-size ratio for a
stream-weighted one (`predict_stream_gb`), leaving the remap and clamp
untouched. It is the fallback route for the validation campaign in
`docs/validation.md`.

### Validation

`docs/validation.md` — "Streamed-bytes campaign" — re-quantizes the shipped
Ornith-1.5-35B-A3B at equal payload size under three gates (streamed bytes
lower; KL vs BF16 not worse, re-measured in-session; measured decode faster)
before anything is republished.
````

- [ ] **Step 3: Fix the stale test count in CLAUDE.md**

Change line 28 of `CLAUDE.md` from `# main suite (~1090 tests, ~15s)` to `# main suite (~1270 tests, ~15s)`.

- [ ] **Step 4: Verify**

Run: `grep -n "^## 11\." docs/redesign.md && grep -n "1270" CLAUDE.md`
Expected: both lines print.

- [ ] **Step 5: Commit**

```bash
git add docs/redesign.md CLAUDE.md
git commit -m "docs: record streamed-bytes addendum (redesign section 11)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 2: Classify PLE and hyper-connection tensors

**Dispatch:** INDEPENDENT

**Files:**
- Modify: `magicquant/gguf/tensor_groups.py:38-71` (`GROUP_PATTERNS`)
- Test: `tests/test_tensor_groups.py` (`test_classify` parametrize list at 30-56; `_KNOWN_ARCH_NAMES` at 95-117)

**Interfaces:**
- Produces: `hc_*` names → group `O`; `per_layer_token_embd.weight` → `E` (already true; pinned).

- [ ] **Step 1: Write the failing tests**

Add to the `@pytest.mark.parametrize("name,expected", [...])` list in `tests/test_tensor_groups.py`:

```python
    # --- qwen4exp hyper-connection mixers: small per-token residual projections, O ---
    ("blk.3.hc_attn_up.weight", "O"),
    ("blk.3.hc_attn_down.weight", "O"),
    ("blk.3.hc_ffn_inject.weight", "O"),
    ("output_hc_down.weight", "O"),
    # --- per-layer embedding table (row-gathered): E. Regression pin — this
    # already classifies to E by substring; the explicit pattern documents it. ---
    ("per_layer_token_embd.weight", "E"),
```

And append to `_KNOWN_ARCH_NAMES`:

```python
    "per_layer_token_embd.weight",
    "blk.3.hc_attn_up.weight",
    "blk.3.hc_ffn_inject.weight",
    "output_hc_down.weight",
```

- [ ] **Step 2: Run to verify the hc_* cases fail**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_tensor_groups.py -q`
Expected: the four `hc_*` cases FAIL with `-> UNKNOWN`; the PLE case passes (it is a pin).

- [ ] **Step 3: Add the patterns**

In `GROUP_PATTERNS`, change the `E` and `O` entries to:

```python
        # per_layer_token_embd: the per-layer (PLE / n-gram) embedding table
        # (gemma-3n, qwen4exp). Row-gathered like token_embd, so group E.
        # Explicit for documentation: the generic pattern already matches it.
        'E': [r'per_layer_token_embd\.weight', r'token_embd\.weight'],
```

```python
        # attn_gate: Qwen3.5 gated attention -- a per-head gate multiplied into
        # the attention output, so it shares O's sensitivity band.
        # hc_*: qwen4exp hyper-connection residual mixers (blk.N.hc_attn_up/
        # down/inject, blk.N.hc_ffn_inject, output_hc_up/down). Small, on the
        # per-token path, quantisable; O is the closest band. Must come before
        # U/D so 'hc_ffn_inject' is not caught by the generic ffn patterns.
        'O': [r'attn_output\.weight', r'attn_gate\.weight',
              r'hc_(attn|ffn)_(up|down|inject)\.weight',
              r'^output_hc_(up|down)\.weight'],
```

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_tensor_groups.py -q`
Expected: all pass, including `test_known_arch_names_never_flagged_unclassified`.

- [ ] **Step 5: Full gates and commit**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/ -q` (expect your recorded baseline + the new cases, same skips) and `/server/programming/MagicQuant/.venv/bin/ruff check --select F magicquant/ tools/ tests/`.

```bash
git add magicquant/gguf/tensor_groups.py tests/test_tensor_groups.py
git commit -m "feat(gguf): classify PLE and hyper-connection tensors" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 3: `magicquant/v2/bandwidth.py` — stream weights and accounting

**Dispatch:** INDEPENDENT

**Files:**
- Create: `magicquant/v2/bandwidth.py`
- Create: `tests/test_v2_bandwidth.py`
- Modify: `tests/test_v2_audit_regressions.py:15-25` (`_float_entry` gains `metadata=None`)

**Interfaces:**
- Produces (consumed by Tasks 4, 5, 7):
  - `KNOWN_GROUPS: frozenset[str]`
  - `expert_ratio(metadata: Dict[str, Any], *, log=None) -> Optional[float]`
  - `stream_weights_by_group(metadata, overrides=None, *, log=None) -> Dict[str, float]`
  - `stream_weights(table_tensors, metadata, overrides=None, *, log=None) -> Dict[str, float]` (per tensor name)
  - `streamed_bytes(assignment, table_tensors, w_stream) -> int`
  - `pure_loss(assignment, table_tensors, kappa) -> float`
  - `bpw_by_group(assignment, table_tensors) -> Dict[str, float]`
  - `group_view(w_stream, table_tensors, *, log=None) -> Tuple[Dict[str, float], Dict[str, float]]` (observed modal weight per group, exceptions); no `"default"` key inside — the report adds it as a sibling

- [ ] **Step 1: Write the failing tests**

Create `tests/test_v2_bandwidth.py`:

```python
"""Stream weights + streamed-bytes accounting (spec: /server/ai/docs/specs/magicquant-bandwidth.md §2.2)."""
import logging
import math

import numpy as np
import pytest

from magicquant.v2 import bandwidth as bw

QWEN_MD = {
    "general.architecture": "qwen35moe",
    "qwen35moe.expert_count": 256,
    "qwen35moe.expert_used_count": 8,
}


def _entry(group, shape, fixed=False, choices=None):
    return {"group": group, "shape": list(shape), "n_elems": int(np.prod(shape)),
            "fixed": fixed, "wnorm": None, "choices": choices or {}}


def test_by_group_from_hparams():
    w = bw.stream_weights_by_group(QWEN_MD)
    assert w["X"] == pytest.approx(8 / 256)
    assert w["E"] == 0.0 and w["V"] == 0.0 and w["UNKNOWN"] == 1.0
    for g in "QKOUDSRHN":
        assert w[g] == 1.0
    assert w["default"] == 1.0
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in w.values())


def test_override_wins():
    w = bw.stream_weights_by_group(QWEN_MD, {"H": 0.5})
    assert w["H"] == 0.5


def test_missing_hparams_prices_experts_hot_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group({"general.architecture": "qwen35moe"}, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "qwen35moe.expert_count" in msg and "qwen35moe.expert_used_count" in msg


def test_zero_expert_count_never_divides(caplog):
    md = dict(QWEN_MD, **{"qwen35moe.expert_count": 0})
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights_by_group(md, log=logging.getLogger("t"))
    assert w["X"] == 1.0
    assert any("expert_count" in r.getMessage() for r in caplog.records)


def test_per_tensor_rules(caplog):
    tensors = {
        "token_embd.weight": _entry("E", (100, 64)),
        "per_layer_token_embd.weight": _entry("E", (100, 16)),
        "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64)),   # gathered by NAME inside H
        "blk.0.nextn.eh_proj.weight": _entry("H", (64, 128)),
        "output.weight": _entry("H", (100, 64)),
        "blk.0.ffn_down_exps.weight": _entry("X", (4, 64, 32)),
        "blk.0.attn_q.weight": _entry("Q", (64, 64)),
        "blk.0.per_layer_proj.weight": _entry("UNKNOWN", (64, 16)),   # PLE projection: read every token
        "blk.0.ffn_gate_inp.weight": _entry("R", (4, 64), fixed=True),
    }
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights(tensors, QWEN_MD, log=logging.getLogger("t"))
    assert w["token_embd.weight"] == 0.0
    assert w["per_layer_token_embd.weight"] == 0.0
    assert w["blk.0.nextn.embed_tokens.weight"] == 0.0
    assert w["blk.0.nextn.eh_proj.weight"] == 1.0
    assert w["output.weight"] == 1.0
    assert w["blk.0.ffn_down_exps.weight"] == pytest.approx(8 / 256)
    assert w["blk.0.attn_q.weight"] == 1.0
    assert w["blk.0.per_layer_proj.weight"] == 1.0
    assert w["blk.0.ffn_gate_inp.weight"] == 1.0
    assert any("UNKNOWN" in r.getMessage() for r in caplog.records)


def test_shape_sanity_warnings(caplog):
    tensors = {"blk.0.ffn_up_exps.weight": _entry("X", (64, 32)),   # 2-D X
               "blk.0.ffn_up.weight": _entry("U", (4, 64, 32))}      # 3-D non-X
    with caplog.at_level(logging.WARNING):
        bw.stream_weights(tensors, QWEN_MD, log=logging.getLogger("t"))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("blk.0.ffn_up_exps.weight" in m for m in msgs)
    assert any("blk.0.ffn_up.weight" in m for m in msgs)


def _table():
    return {
        "a": _entry("U", (8, 8), choices={"BF16": {"actual": "BF16", "bytes": 128, "werr": 0.0},
                                          "Q4_K_M": {"actual": "Q4_K_M", "bytes": 36, "werr": 1.0}}),
        "x": _entry("X", (2, 8, 8), choices={"BF16": {"actual": "BF16", "bytes": 256, "werr": 0.0},
                                             "Q4_K_M": {"actual": "Q4_K_M", "bytes": 72, "werr": 0.5}}),
        "n": _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}}),
    }


def test_streamed_bytes_and_pure_loss_by_hand():
    t = _table()
    w = {"a": 1.0, "x": 0.25, "n": 1.0}
    assign = {"a": "Q4_K_M", "x": "BF16", "n": "F32"}
    assert bw.streamed_bytes(assign, t, w) == 36 + int(0.25 * 256) + 32
    assert bw.pure_loss(assign, t, {"U": 2.0, "X": 3.0}) == pytest.approx(2.0 * 1.0 + 0.0)
    assert bw.streamed_bytes({"a": "Q4_K_M"}, t, {}) == 36          # missing weight -> 1.0


def test_bpw_by_group_and_group_view(caplog):
    t = _table()
    assign = {"a": "Q4_K_M", "x": "Q4_K_M", "n": "F32"}
    b = bw.bpw_by_group(assign, t)
    assert b["U"] == pytest.approx(36 * 8 / 64)
    assert b["X"] == pytest.approx(72 * 8 / 128)
    by_group, exceptions = bw.group_view({"a": 1.0, "x": 0.25, "n": 1.0}, t)
    assert by_group == {"U": 1.0, "X": 0.25, "N": 1.0}
    assert exceptions == {}
    # a second N tensor at a different weight: modal value wins, the odd one is an exception, a WARNING fires
    t2 = dict(t)
    t2["n2"] = _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}})
    t2["n3"] = _entry("N", (8,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 32, "werr": 0.0}})
    with caplog.at_level(logging.WARNING):
        by_group, exceptions = bw.group_view({"a": 1.0, "x": 0.25, "n": 1.0, "n2": 1.0, "n3": 0.0}, t2, log=logging.getLogger("t"))
    assert by_group["N"] == 1.0 and exceptions == {"n3": 0.0}
    assert any("distinct weights" in r.getMessage() for r in caplog.records)


def test_group_view_ignores_gathered_by_name_disagreement(caplog):
    # rule-2 tensors (MTP embedding inside H) are an intended exception: no WARNING
    t = {"output.weight": _entry("H", (100, 64)),
         "blk.0.nextn.embed_tokens.weight": _entry("H", (100, 64))}
    with caplog.at_level(logging.WARNING):
        by_group, exceptions = bw.group_view({"output.weight": 1.0, "blk.0.nextn.embed_tokens.weight": 0.0}, t, log=logging.getLogger("t"))
    assert by_group["H"] == 1.0 and exceptions == {"blk.0.nextn.embed_tokens.weight": 0.0}
    assert not any("distinct weights" in r.getMessage() for r in caplog.records)


def test_float_entry_stub_source_gets_expert_ratio(monkeypatch, tmp_path):
    from tests.test_v2_audit_regressions import _float_entry
    weights = np.full((2, 4, 32), 0.5, dtype=np.float32)
    entry = _float_entry(monkeypatch, tmp_path, weights, metadata=QWEN_MD)
    w = bw.stream_weights({"blk.0.ffn_down_exps.weight": entry}, QWEN_MD)
    assert w["blk.0.ffn_down_exps.weight"] == pytest.approx(8 / 256)


def test_float_entry_default_metadata_prices_hot(monkeypatch, tmp_path, caplog):
    from tests.test_v2_audit_regressions import _float_entry
    weights = np.full((2, 4, 32), 0.5, dtype=np.float32)
    entry = _float_entry(monkeypatch, tmp_path, weights)
    with caplog.at_level(logging.WARNING):
        w = bw.stream_weights({"blk.0.ffn_down_exps.weight": entry},
                              {"general.architecture": "llama"}, log=logging.getLogger("t"))
    assert w["blk.0.ffn_down_exps.weight"] == 1.0
    assert any("expert_count" in r.getMessage() for r in caplog.records)
```

And change `_float_entry` in `tests/test_v2_audit_regressions.py` to:

```python
def _float_entry(monkeypatch, tmp_path, weights, *, sample_rows=None, imatrix=None, metadata=None):
    name = "blk.0.ffn_down_exps.weight" if weights.ndim == 3 else "blk.0.ffn_down.weight"
    source = StubSource([(name, weights, weights.shape)], metadata=metadata)
```
(the rest unchanged — `StubSource` already accepts `metadata=`, `tests/test_writer.py:33`).

- [ ] **Step 2: Run to verify it fails**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_bandwidth.py -q`
Expected: FAIL with `ImportError: cannot import name 'bandwidth'`.

- [ ] **Step 3: Write the module**

Create `magicquant/v2/bandwidth.py`:

```python
"""Stream weights and streamed-bytes accounting for the v2 allocator.

Spec: /server/ai/docs/specs/magicquant-bandwidth.md (sections 2.2-2.3);
design record: docs/redesign.md section 11.

A tensor's *stream weight* w_t in [0, 1] is the fraction of its stored bytes
a single decode token reads: 1.0 for the trunk (read in full every token),
expert_used_count / expert_count for routed experts, 0.0 for row-gathered
embedding tables. Weights are architectural -- derived from the group, the
name and the GGUF hparams -- never measured, and never written into the cached
distortion table (they are computed after the table is loaded).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Mapping, Optional, Tuple

KNOWN_GROUPS = frozenset({"E", "H", "X", "R", "Q", "K", "O", "S", "U", "D", "N", "V", "UNKNOWN"})

# Row-gathered embedding tables, matched on the NAME (any prefix), so the MTP
# module's own copy (blk.N.nextn.embed_tokens.weight, group H) is caught too.
_GATHERED_NAME = re.compile(r"(^|\.)(token_embd|embed_tokens|per_layer_token_embd)\.weight$")


def _warn(log, msg: str, *args) -> None:
    # Pre-format: callers may pass a stdlib logger (tests) or magicquant's
    # structlog logger (search.py), and structlog does not %-format positional args.
    if log is not None:
        log.warning(msg % args if args else msg)


def expert_ratio(metadata: Mapping[str, Any], *, log=None) -> Optional[float]:
    """expert_used_count / expert_count from the GGUF hparams, or None (with a
    warning) when either is missing or expert_count is zero. Never divides by
    zero; never returns a non-finite value."""
    arch = metadata.get("general.architecture")
    key_n = f"{arch}.expert_count"
    key_k = f"{arch}.expert_used_count"
    n = metadata.get(key_n) if arch else None
    k = metadata.get(key_k) if arch else None
    try:
        n = int(n) if n is not None else None
        k = int(k) if k is not None else None
    except (TypeError, ValueError):
        n, k = None, None
    if not n or k is None or n <= 0:
        _warn(log, "stream weights: %s / %s missing or zero (%r / %r); "
                   "routed experts priced at w=1.0 (today's pricing)", key_n, key_k, n, k)
        return None
    return min(1.0, max(0.0, k / n))


def stream_weights_by_group(metadata: Mapping[str, Any], overrides: Optional[Mapping[str, float]] = None,
                            *, log=None) -> Dict[str, float]:
    """One weight per group letter (plus "default"), the v1 entry point.
    Rule order: override > X ratio > E/V zero > everything else 1.0."""
    overrides = dict(overrides or {})
    for g in overrides:
        if g not in KNOWN_GROUPS:
            _warn(log, "stream weights: override for unknown group %r", g)
    ratio = expert_ratio(metadata, log=log)
    w: Dict[str, float] = {g: 1.0 for g in KNOWN_GROUPS}
    w["X"] = 1.0 if ratio is None else ratio
    w["E"] = 0.0
    w["V"] = 0.0
    for g, v in overrides.items():
        w[g] = float(v)
    w["default"] = 1.0
    for g, v in w.items():
        if not (math.isfinite(v) and 0.0 <= v <= 1.0):
            raise ValueError(f"stream weight for group {g!r} is not a finite value in [0,1]: {v!r}")
    return w


def stream_weights(table_tensors: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any],
                   overrides: Optional[Mapping[str, float]] = None, *, log=None) -> Dict[str, float]:
    """Per-tensor weights over a distortion table's ``tensors`` mapping.
    Rule order (first match wins): override by group > gathered by name >
    X ratio > E > V > UNKNOWN (1.0, warned) > everything else 1.0."""
    by_group = stream_weights_by_group(metadata, overrides, log=log)
    overrides = dict(overrides or {})
    out: Dict[str, float] = {}
    unknown = 0
    for name, entry in table_tensors.items():
        group = entry.get("group", "UNKNOWN")
        shape = list(entry.get("shape") or [])
        if group in overrides:
            w = float(overrides[group])
        elif _GATHERED_NAME.search(name):
            w = 0.0
        elif group == "X":
            w = by_group["X"]
        elif group in ("E", "V"):
            w = 0.0
        elif group == "UNKNOWN":
            w = 1.0
            unknown += 1
        else:
            w = 1.0
        if group == "X" and len(shape) != 3:
            _warn(log, "stream weights: group-X tensor %s is not 3-D (shape %r)", name, shape)
        if group != "X" and len(shape) == 3:
            _warn(log, "stream weights: 3-D tensor %s is in group %s, not X", name, group)
        if not (math.isfinite(w) and 0.0 <= w <= 1.0):
            raise ValueError(f"stream weight for {name!r} is not a finite value in [0,1]: {w!r}")
        out[name] = w
    if unknown:
        _warn(log, "stream weights: %d UNKNOWN-group tensor(s) priced at w=1.0", unknown)
    return out


def streamed_bytes(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
                   w_stream: Mapping[str, float]) -> int:
    """Sum over ALL assigned tensors (fixed included) of w_t * bytes(t, chosen).
    A tensor missing from w_stream is priced at 1.0."""
    total = 0.0
    for name, scheme in assignment.items():
        b = int(table_tensors[name]["choices"][scheme]["bytes"])
        total += w_stream.get(name, 1.0) * b
    return int(round(total))


def pure_loss(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]],
              kappa: Mapping[str, float]) -> float:
    """Sum of kappa_g * werr(t, chosen) over non-fixed tensors -- the quality
    objective with no lambda term (fixed tensors contribute 0.0, as today)."""
    total = 0.0
    for name, scheme in assignment.items():
        entry = table_tensors[name]
        if entry.get("fixed"):
            continue
        werr = entry["choices"][scheme].get("werr")
        if werr is None:
            continue
        total += float(kappa.get(entry["group"], 1.0)) * float(werr)
    return total


def bpw_by_group(assignment: Mapping[str, str], table_tensors: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Mean bits per weight per group for an assignment (bytes*8 / n_elems)."""
    bytes_g: Dict[str, int] = defaultdict(int)
    elems_g: Dict[str, int] = defaultdict(int)
    for name, scheme in assignment.items():
        entry = table_tensors[name]
        bytes_g[entry["group"]] += int(entry["choices"][scheme]["bytes"])
        elems_g[entry["group"]] += int(entry.get("n_elems") or 0)
    return {g: (bytes_g[g] * 8.0 / elems_g[g]) for g in bytes_g if elems_g[g] > 0}


def group_view(w_stream: Mapping[str, float], table_tensors: Mapping[str, Mapping[str, Any]],
               *, log=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(observed_by_group, exceptions) for reporting: the modal weight over
    each group's tensors (a group with more than one distinct weight logs a
    WARNING, ignoring rule-2 gathered-by-name tensors, whose disagreement is
    intended); exceptions lists every tensor whose weight differs from its
    group's modal value."""
    per_group: Dict[str, Counter] = defaultdict(Counter)
    disagree_check: Dict[str, set] = defaultdict(set)
    for name, entry in table_tensors.items():
        g = entry.get("group", "UNKNOWN")
        w = w_stream.get(name, 1.0)
        per_group[g][w] += 1
        if not _GATHERED_NAME.search(name):        # rule-2 tensors are an intended, silent exception
            disagree_check[g].add(w)
    by_group: Dict[str, float] = {}
    for g, counts in per_group.items():
        if len(disagree_check[g]) > 1:
            _warn(log, "stream weights: group %s has %d distinct weights %r", g, len(disagree_check[g]), sorted(disagree_check[g]))
        by_group[g] = counts.most_common(1)[0][0]
    exceptions = {name: w_stream.get(name, 1.0) for name, entry in table_tensors.items()
                  if w_stream.get(name, 1.0) != by_group[entry.get("group", "UNKNOWN")]}
    return by_group, exceptions
```

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_bandwidth.py tests/test_v2_audit_regressions.py -q`
Expected: all pass.

- [ ] **Step 5: Full gates and commit**

Run the full suite and ruff (Global Constraints).

```bash
git add magicquant/v2/bandwidth.py tests/test_v2_bandwidth.py tests/test_v2_audit_regressions.py
git commit -m "feat(v2): stream weights and streamed-bytes accounting" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 4: `BandwidthInfeasibleError`, `BandwidthSolution`, `solve_bandwidth`

**Dispatch:** DEPENDS-ON Task 3

**Files:**
- Modify: `magicquant/v2/outcome.py` (after `BudgetInfeasibleError`, line 33)
- Modify: `magicquant/v2/__init__.py:12-16` (export)
- Modify: `magicquant/v2/bandwidth.py` (append)
- Test: `tests/test_v2_bandwidth.py` (append)

**Interfaces:**
- Consumes: Task 3's `streamed_bytes`, `pure_loss`, `bpw_by_group`, `group_view`; `magicquant.v2.allocate.allocate`, `Unit`, `Choice`.
- Produces (consumed by Task 5):
  - `class BandwidthInfeasibleError(RuntimeError)` with `.budget_bytes`, `.min_bytes` — in `outcome.py`, exported from `magicquant.v2`.
  - `LAM_MAX = 1e9`
  - `@dataclass BandwidthSolution` with fields `mode: str, lam: float, chosen: Allocation, w_stream: Dict[str,float], by_group: Dict[str,float], exceptions: Dict[str,float], weights_source: Dict[str,Any], streamed_bytes: int, streamed_bytes_lambda0: int, streamed_bytes_min: Optional[int], budget_bw_bytes: Optional[int], nonmonotone_probes: int, predicted_loss_pure: float, total_loss_with_lambda: float, bpw_lambda0: Dict[str,float], bpw_chosen: Dict[str,float]` and `to_json() -> Dict` (everything except `chosen` and `w_stream`, with `"stream_weights": {"by_group":..., "exceptions":...}` and `"bpw_by_group": {"lambda0":..., "chosen":...}`).
  - `_count_inversions(probes: Sequence[Tuple[float, int]]) -> int` — pairs `i<j` with `lam_i < lam_j` and `s_i < s_j` (streamed bytes rose with λ).
  - `solve_bandwidth(build_units, table, kappa, floors, w_stream, budget_bytes, budget_bw_bytes, *, lam_fixed=None, weights_source=None, log=None) -> BandwidthSolution` where `build_units(table, kappa, floors, w_stream, lam) -> List[Unit]` is supplied by the caller (Task 5 passes `search._build_units`; tests pass a local builder) — this keeps `bandwidth.py` free of any import of `search.py`. Raises `ValueError` if both `lam_fixed` and `budget_bw_bytes` are given; `BandwidthInfeasibleError` when `s_min > budget_bw_bytes`; `RuntimeError` when the bisection finds no feasible λ although `s_min <= budget`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_v2_bandwidth.py`:

```python
from magicquant.v2.allocate import Choice, Unit, allocate  # noqa: E402
from magicquant.v2 import BandwidthInfeasibleError  # noqa: E402


def _moe_table():
    """X carries ~95% of bytes at w=0.03; U and H are hot (w=1); N is fixed."""
    def ch(bf, q6, q6w, q4, q4w):
        return {"BF16": {"actual": "BF16", "bytes": bf, "werr": 0.0},
                "Q6_K": {"actual": "Q6_K", "bytes": q6, "werr": q6w},
                "Q4_K_M": {"actual": "Q4_K_M", "bytes": q4, "werr": q4w}}
    t = {}
    for i in range(4):
        t[f"blk.{i}.ffn_down_exps.weight"] = _entry("X", (8, 64, 64), choices=ch(65536, 26880, 0.02, 18432, 0.08))
        t[f"blk.{i}.attn_q.weight"] = _entry("U", (64, 64), choices=ch(8192, 3360, 0.05, 2304, 0.20))
    t["output.weight"] = _entry("H", (128, 64), choices=ch(16384, 6720, 0.05, 4608, 0.25))
    t["output_norm.weight"] = _entry("N", (64,), fixed=True, choices={"F32": {"actual": "F32", "bytes": 256, "werr": 0.0}})
    return t


def _build_units(table, kappa, floors, w_stream, lam):
    units = []
    for name, entry in table["tensors"].items():
        if entry.get("fixed"):
            (s, c), = entry["choices"].items()
            units.append(Unit(name=name, group=entry["group"], choices=[Choice(s, c["actual"], int(c["bytes"]), 0.0)]))
            continue
        k = kappa.get(entry["group"], 1.0)
        chs = []
        for s, c in entry["choices"].items():
            loss = k * float(c["werr"])
            if lam:
                loss += lam * w_stream.get(name, 1.0) * int(c["bytes"]) / 2**30
            chs.append(Choice(s, c["actual"], int(c["bytes"]), loss))
        units.append(Unit(name=name, group=entry["group"], choices=chs))
    return units


def _solve(budget_bytes, budget_bw_bytes=None, lam_fixed=None):
    table = {"tensors": _moe_table()}
    w = bw.stream_weights(table["tensors"], {"general.architecture": "m", "m.expert_count": 100, "m.expert_used_count": 3})
    return bw.solve_bandwidth(_build_units, table, {"X": 1.0, "U": 1.0, "H": 1.0}, {}, w,
                              budget_bytes, budget_bw_bytes, lam_fixed=lam_fixed), table, w


def test_mode_off_is_lambda_zero():
    sol, table, w = _solve(budget_bytes=200_000)
    assert sol.mode == "off" and sol.lam == 0.0
    assert sol.streamed_bytes == sol.streamed_bytes_lambda0 == bw.streamed_bytes(sol.chosen.assignment, table["tensors"], w)
    assert sol.streamed_bytes_min is None and sol.budget_bw_bytes is None
    assert sol.predicted_loss_pure == pytest.approx(sol.chosen.total_loss)


def _s_min(budget_bytes=200_000):
    table = {"tensors": _moe_table()}
    w = bw.stream_weights(table["tensors"], {"general.architecture": "m", "m.expert_count": 100, "m.expert_used_count": 3})
    a = allocate(_build_units(table, {"X": 1.0, "U": 1.0, "H": 1.0}, {}, w, bw.LAM_MAX), budget_bytes)
    return bw.streamed_bytes(a.assignment, table["tensors"], w)


def test_budget_mode_moves_bits_from_experts_to_trunk():
    sol0, table, w = _solve(budget_bytes=200_000)
    target = (sol0.streamed_bytes_lambda0 + _s_min()) // 2          # strictly inside (s_min, s0)
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=target)
    assert sol.mode == "budget" and sol.lam > 0.0
    assert sol.streamed_bytes <= sol.budget_bw_bytes
    assert sol.chosen.total_bytes <= 200_000
    assert sol.streamed_bytes < sol.streamed_bytes_lambda0
    # direction: hot groups (U, H) LOSE bpw; cold X GAINS or holds
    assert sol.bpw_chosen["U"] + sol.bpw_chosen["H"] < sol.bpw_lambda0["U"] + sol.bpw_lambda0["H"]
    assert sol.bpw_chosen["X"] >= sol.bpw_lambda0["X"]
    assert sol.predicted_loss_pure < sol.total_loss_with_lambda
    assert sol.predicted_loss_pure == pytest.approx(bw.pure_loss(sol.chosen.assignment, table["tensors"], {"X": 1.0, "U": 1.0, "H": 1.0}))


def test_both_modes_is_an_error():
    with pytest.raises(ValueError):
        _solve(budget_bytes=200_000, budget_bw_bytes=60_000, lam_fixed=0.01)


def test_count_inversions():
    assert bw._count_inversions([(1e-3, 100), (1e-2, 90), (1e-1, 80)]) == 0
    assert bw._count_inversions([(1e-3, 100), (1e-2, 90), (1e-1, 95), (1.0, 80)]) == 1
    assert bw._count_inversions([]) == 0


def test_budget_already_met_returns_lambda_zero():
    sol0, _, _ = _solve(budget_bytes=200_000)
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=sol0.streamed_bytes_lambda0 * 2)
    assert sol.mode == "budget" and sol.lam == 0.0


def test_infeasible_budget_raises_and_never_returns_lam_max():
    with pytest.raises(BandwidthInfeasibleError) as ei:
        _solve(budget_bytes=200_000, budget_bw_bytes=1)
    assert ei.value.budget_bytes == 1 and ei.value.min_bytes > 1


def test_weight_mode_reports_given_lambda():
    sol, _, _ = _solve(budget_bytes=200_000, lam_fixed=0.05)
    assert sol.mode == "weight" and sol.lam == 0.05


def test_determinism():
    a, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    b, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    assert a.chosen.assignment == b.chosen.assignment and a.lam == b.lam


def test_to_json_shape():
    sol, _, _ = _solve(budget_bytes=200_000, budget_bw_bytes=60_000)
    j = sol.to_json()
    assert set(j) == {"mode", "lambda", "stream_weights", "weights_source", "streamed_bytes",
                      "streamed_bytes_lambda0", "streamed_bytes_min", "budget_bw_bytes",
                      "nonmonotone_probes", "predicted_loss_pure", "total_loss_with_lambda", "bpw_by_group"}
    assert set(j["stream_weights"]) == {"observed_by_group", "default", "exceptions"}
    assert j["stream_weights"]["default"] == 1.0
    assert set(j["bpw_by_group"]) == {"lambda0", "chosen"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_bandwidth.py -q`
Expected: FAIL with `ImportError: cannot import name 'BandwidthInfeasibleError'`.

- [ ] **Step 3: Implement**

Append to `magicquant/v2/outcome.py` (after `BudgetInfeasibleError`):

```python
class BandwidthInfeasibleError(RuntimeError):
    """The requested streamed-bytes budget (--budget-bw-gb) is below the
    smallest streamed-bytes figure any allocation under the storage budget
    can reach. Carries ``min_bytes`` so callers can report what IS achievable."""

    def __init__(self, budget_bytes: int, min_bytes: int):
        self.budget_bytes = budget_bytes
        self.min_bytes = min_bytes
        super().__init__(
            f"Streamed-bytes budget {budget_bytes / 1024**3:.2f} GiB/token is "
            f"infeasible: the least any allocation streams under the storage "
            f"budget is {min_bytes / 1024**3:.2f} GiB/token. Raise --budget-bw-gb, "
            "raise --budget-gb, or enable more aggressive schemes."
        )
```

In `magicquant/v2/__init__.py`, add `BandwidthInfeasibleError,` to the `from magicquant.v2.outcome import (...)` list.

Append to `magicquant/v2/bandwidth.py`:

```python
from dataclasses import dataclass, field  # noqa: E402  (keep imports at top in the real file)
from typing import Callable, List, Sequence, Tuple  # noqa: E402

from magicquant.v2.allocate import Allocation, Unit, allocate  # noqa: E402
from magicquant.v2.outcome import BandwidthInfeasibleError  # noqa: E402

LAM_MAX = 1e9          # feasibility probe only -- never returned as the chosen lambda
LAM_MIN = 1e-6
BISECTION_STEPS = 40


def _count_inversions(probes) -> int:
    """Number of probe pairs (i < j) where lambda rose but streamed bytes rose
    too -- evidence that S(lambda) is not monotone (allocate.py drops a unit
    permanently once its next edge exceeds the remaining budget)."""
    n = 0
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            if probes[i][0] < probes[j][0] and probes[i][1] < probes[j][1]:
                n += 1
    return n

BuildUnits = Callable[[Dict[str, Any], Mapping[str, float], Mapping[str, str], Mapping[str, float], float], List[Unit]]


@dataclass
class BandwidthSolution:
    mode: str                                   # "off" | "weight" | "budget"
    lam: float
    chosen: Allocation
    w_stream: Dict[str, float]
    by_group: Dict[str, float]
    exceptions: Dict[str, float]
    weights_source: Dict[str, Any]
    streamed_bytes: int
    streamed_bytes_lambda0: int
    streamed_bytes_min: Optional[int]
    budget_bw_bytes: Optional[int]
    nonmonotone_probes: int
    predicted_loss_pure: float
    total_loss_with_lambda: float
    bpw_lambda0: Dict[str, float] = field(default_factory=dict)
    bpw_chosen: Dict[str, float] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "lambda": self.lam,
            "stream_weights": {"observed_by_group": self.by_group, "default": 1.0, "exceptions": self.exceptions},
            "weights_source": self.weights_source,
            "streamed_bytes": self.streamed_bytes,
            "streamed_bytes_lambda0": self.streamed_bytes_lambda0,
            "streamed_bytes_min": self.streamed_bytes_min,
            "budget_bw_bytes": self.budget_bw_bytes,
            "nonmonotone_probes": self.nonmonotone_probes,
            "predicted_loss_pure": self.predicted_loss_pure,
            "total_loss_with_lambda": self.total_loss_with_lambda,
            "bpw_by_group": {"lambda0": self.bpw_lambda0, "chosen": self.bpw_chosen},
        }


def solve_bandwidth(build_units: BuildUnits, table: Dict[str, Any], kappa: Mapping[str, float],
                    floors: Mapping[str, str], w_stream: Mapping[str, float],
                    budget_bytes: int, budget_bw_bytes: Optional[int], *,
                    lam_fixed: Optional[float] = None,
                    weights_source: Optional[Dict[str, Any]] = None, log=None) -> BandwidthSolution:
    """Choose lambda and the allocation. The storage budget is always hard
    (enforced inside allocate()); lambda only steers which bytes are spent.
    Spec section 2.3."""
    tensors = table["tensors"]
    kappa = dict(kappa)
    if lam_fixed is not None and budget_bw_bytes is not None:
        raise ValueError("bandwidth_weight and budget_bw_gb are mutually exclusive")
    probes: List[Tuple[float, int]] = []

    def solve(lam: float):
        alloc = allocate(build_units(table, kappa, floors, w_stream, lam), budget_bytes)
        s = streamed_bytes(alloc.assignment, tensors, w_stream)
        probes.append((lam, s))
        return alloc, s

    def finish(mode, lam, alloc, s, s0, s_min, nonmono, a0):
        return BandwidthSolution(
            mode=mode, lam=lam, chosen=alloc, w_stream=dict(w_stream),
            by_group=group_view(w_stream, tensors, log=log)[0],
            exceptions=group_view(w_stream, tensors)[1],
            weights_source=dict(weights_source or {}),
            streamed_bytes=s, streamed_bytes_lambda0=s0, streamed_bytes_min=s_min,
            budget_bw_bytes=budget_bw_bytes if mode == "budget" else None,
            nonmonotone_probes=nonmono,
            predicted_loss_pure=pure_loss(alloc.assignment, tensors, kappa),
            total_loss_with_lambda=alloc.total_loss,
            bpw_lambda0=bpw_by_group(a0.assignment, tensors),
            bpw_chosen=bpw_by_group(alloc.assignment, tensors),
        )

    a0, s0 = solve(0.0)
    if lam_fixed is not None:
        a, s = solve(float(lam_fixed))
        return finish("weight", float(lam_fixed), a, s, s0, None, 0, a0)
    if budget_bw_bytes is None:
        return finish("off", 0.0, a0, s0, s0, None, 0, a0)
    B = int(budget_bw_bytes)
    if s0 <= B:
        return finish("budget", 0.0, a0, s0, s0, None, 0, a0)
    a_max, s_min = solve(LAM_MAX)
    if s_min > B:
        raise BandwidthInfeasibleError(B, s_min)
    lo, hi = LAM_MIN, LAM_MAX
    best_alloc: Optional[Allocation] = None
    best_lam = 0.0
    best_s = 0
    best_pure = math.inf
    for _ in range(BISECTION_STEPS):
        mid = math.sqrt(lo * hi)
        a, s = solve(mid)
        if s <= B:
            hi = mid
            pl = pure_loss(a.assignment, tensors, kappa)
            if best_alloc is None or pl < best_pure:
                best_alloc, best_lam, best_s, best_pure = a, mid, s, pl
        else:
            lo = mid
    nonmono = _count_inversions(probes)
    if best_alloc is None:
        raise RuntimeError(
            f"bandwidth: bisection found no feasible lambda although the streamed-bytes "
            f"floor {s_min / 1024**3:.3f} GiB <= budget {B / 1024**3:.3f} GiB; "
            f"S(lambda) is non-monotone ({nonmono} inversions over {len(probes)} probes)"
        )
    assert best_s <= B and best_alloc.total_bytes <= budget_bytes
    if nonmono and log is not None:
        log.warning("bandwidth: S(lambda) non-monotone (%d inversions over %d probes)", nonmono, len(probes))
    return finish("budget", best_lam, best_alloc, best_s, s0, s_min, nonmono, a0)
```

(Put the imports at the top of the module with the others; the `# noqa` markers above are only because this plan shows an append.)

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_bandwidth.py -q`
Expected: all pass. If `test_budget_mode_moves_bits_from_experts_to_trunk` fails on the direction assertions, the λ sign is wrong — the term must be *added* to loss (spec §4 M2 (c)).

- [ ] **Step 5: Full gates and commit**

```bash
git add magicquant/v2/outcome.py magicquant/v2/__init__.py magicquant/v2/bandwidth.py tests/test_v2_bandwidth.py
git commit -m "feat(v2): bisected streamed-bytes solver and BandwidthInfeasibleError" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 5: Wire the solver into `search.py` and recompute published losses

**Dispatch:** DEPENDS-ON Task 4

**Files:**
- Modify: `magicquant/v2/search.py` — `V2Config` (49-83), `_build_units` (167-217), `_allocate_frontier_and_anchors` (361-395), `_build_and_verify_anchors` (~398-470; the anchor `entry` dict at 444-450), `_assemble_results` (510-579), `run_budget_search` (646-692)
- Test: `tests/test_v2_bandwidth.py` (append), `tests/test_v2_search_characterization.py:474-480` and `:330` (pins)

**Interfaces:**
- Consumes: Task 4's `solve_bandwidth`, `BandwidthSolution`; Task 3's `stream_weights`, `pure_loss`, `streamed_bytes`.
- Produces (consumed by Task 6): `V2Config.bandwidth_weight: float = 0.0`, `V2Config.budget_bw_gb: Optional[float] = None`, `V2Config.stream_weights: Dict[str, float]`; `results["bandwidth"]`; `frontier.json["lambda"]`.

- [ ] **Step 1: Write the failing tests**

Update the four pins in `tests/test_v2_search_characterization.py`: add `"bandwidth",` to the results set at 238-244 (after `"final_model"`), `"streamed_bytes"` to both per-anchor sets at 286 and 590, and `"lambda"` to the frontier set at 330. Do not touch the `Allocation.to_json()` pin at 268-271 or any value pin. Also extend `_install_stubs` (135-195) with a fifth stub so the happy path exercises the real metadata code: `import magicquant.gguf.source as source_mod` at the top of the file and, beside the other four `monkeypatch.setattr` calls, `monkeypatch.setattr(source_mod, "open_model_source", lambda path, *a, **kw: _FakeSource())` where `_FakeSource` has `get_metadata()` returning `{"general.architecture": "llama"}` and a no-op `close()`. Append to `tests/test_v2_bandwidth.py` a test that drives `run_budget_search` through the characterization harness — import `_patch_pipeline`-style helpers from `tests.test_v2_search_characterization` (use whatever fixture/helper that file exposes to build `happy_run`; read lines 100-232 of it first) with `budget_bw_gb` set below the λ=0 streamed figure, and assert: `results["bandwidth"]["mode"] == "budget"`, `results["allocation"]["predicted_loss"] == results["bandwidth"]["predicted_loss_pure"]`, every `results["anchors"][i]` has `"streamed_bytes"`, and `frontier.json["lambda"] == results["bandwidth"]["lambda"]`; that `results["allocation"]["predicted_loss"]` equals `pure_loss(assignment, table["tensors"], results["kappa"])` recomputed in the test and is `< results["bandwidth"]["total_loss_with_lambda"]`; and (the (h) invariant) that at a λ small enough to leave the assignment unchanged (`bandwidth_weight=1e-9`, assert the assignment equals the λ=0 one) `results["report_fit_affine"]` is identical to the λ=0 run's. Also assert that with no bandwidth flags `results["bandwidth"]["mode"] == "off"`, `results["allocation"]["predicted_loss"] == pytest.approx(0.07)` (the existing pin value), and, in a separate test that monkeypatches `magicquant.gguf.source.open_model_source` to raise `ValueError`, that the run still completes with `results["bandwidth"]["weights_source"]["error"]` set and every `observed_by_group` value 1.0.

- [ ] **Step 2: Run to verify it fails**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_search_characterization.py tests/test_v2_bandwidth.py -q`
Expected: the pin tests FAIL (`bandwidth`/`lambda` missing), the new test FAILS (`V2Config` has no `budget_bw_gb`).

- [ ] **Step 3: Implement**

`V2Config` — add after `enable_iq`:
```python
    # Streamed-bytes term (docs/redesign.md section 11). Both off = byte-
    # identical to today. bandwidth_weight is a fixed lambda (loss per
    # streamed GiB); budget_bw_gb bisects lambda to a streamed-bytes budget.
    bandwidth_weight: float = 0.0
    budget_bw_gb: Optional[float] = None
    stream_weights: Dict[str, float] = field(default_factory=dict)  # group -> w override
```

`_build_units` — new signature and choosable-branch loss:
```python
def _build_units(
    table: Dict[str, Any],
    kappa: Dict[str, float],
    floors: Dict[str, str],
    w_stream: Optional[Dict[str, float]] = None,
    lam: float = 0.0,
) -> List[Unit]:
```
and replace the `choices.append(Choice(scheme, c["actual"], int(c["bytes"]), k * float(c["werr"])))` call with:
```python
                loss = k * float(c["werr"])
                if lam:
                    loss += lam * (w_stream or {}).get(name, 1.0) * int(c["bytes"]) / 2**30
                choices.append(Choice(scheme, c["actual"], int(c["bytes"]), loss))
```
(the fixed branch is unchanged).

`_allocate_frontier_and_anchors` — new signature `(table, kappa, cfg, budget_bytes, w_stream, budget_bw_bytes, weights_source)` returning `Tuple[Allocation, List[Allocation], List[Dict[str, Any]], BandwidthSolution, List[Dict[str, Any]]]` (chosen, anchor_allocs, failures, solution, anchor_stats):
```python
    solution = solve_bandwidth(
        _build_units, table, kappa, cfg.floors, w_stream, budget_bytes, budget_bw_bytes,
        lam_fixed=(cfg.bandwidth_weight or None), weights_source=weights_source, log=log,
    )
    chosen = solution.chosen
    units = _build_units(table, kappa, cfg.floors, w_stream, solution.lam)
    log.info("v2 allocation solved", stage="allocate", size_gb=round(chosen.total_bytes / 1024**3, 3),
             predicted_loss=solution.predicted_loss_pure, frontier_points=len(chosen.frontier),
             bandwidth_mode=solution.mode, lam=solution.lam,
             streamed_gb=round(solution.streamed_bytes / 1024**3, 3))
    anchor_allocs = [chosen]; failures = []
    for i in range(1, max(1, cfg.anchors)):
        ...  # unchanged loop body, allocate(units, int(budget_bytes * factor))
    anchor_stats = [
        {"predicted_loss_pure": pure_loss(a.assignment, table["tensors"], kappa),
         "streamed_bytes": streamed_bytes(a.assignment, table["tensors"], w_stream)}
        for a in anchor_allocs
    ]
    return chosen, anchor_allocs, failures, solution, anchor_stats
```
`_build_and_verify_anchors(..., out_dir, anchor_stats=None)`: in the `entry = {...}` dict, `"predicted_loss": anchor_stats[idx]["predicted_loss_pure"] if anchor_stats else alloc.total_loss`, and add `"streamed_bytes": anchor_stats[idx]["streamed_bytes"] if anchor_stats else None`.

`_assemble_results(..., report_fit, *, bandwidth: Optional[BandwidthSolution] = None)`: after building `results`, do
```python
    if bandwidth is not None:
        results["allocation"]["predicted_loss"] = bandwidth.predicted_loss_pure
        results["bandwidth"] = bandwidth.to_json()
    else:
        results["bandwidth"] = {"mode": "off", "lambda": 0.0}
```
and add `"lambda": bandwidth.lam if bandwidth is not None else 0.0,` to the `frontier.json` dict at 554-557. Emit one log line in the file's structlog style: `log.info("bandwidth", mode=..., lam=..., streamed_gb=round(.., 3), streamed_gb_lambda0=..., streamed_gb_min=..., storage_gb=..., budget_gb=...)`.

`run_budget_search`: after `table = _build_distortion_table(...)` — the import is
function-local (so tests can monkeypatch `magicquant.gguf.source.open_model_source`)
and the read is fail-soft (the characterization suite's `SOURCE = "src.gguf"` does
not exist on disk; `open_model_source` raises `ValueError` on it):
```python
    from magicquant.gguf.source import open_model_source
    metadata: Dict[str, Any] = {}
    meta_error: Optional[str] = None
    try:
        src = open_model_source(cfg.source_model_path)
        try:
            metadata = dict(src.get_metadata())
        finally:
            src.close()
    except Exception as exc:  # unreadable/absent source: price every byte as streamed
        meta_error = f"{type(exc).__name__}: {exc}"
        log.warning("bandwidth: could not read source metadata; all stream weights = 1.0", error=meta_error)
    w_stream = stream_weights(table["tensors"], metadata, cfg.stream_weights, log=log)
    arch = metadata.get("general.architecture")
    weights_source = {
        "arch": arch,
        "expert_count": metadata.get(f"{arch}.expert_count"),
        "expert_used_count": metadata.get(f"{arch}.expert_used_count"),
        "overrides": dict(cfg.stream_weights),
        "unknown_tensors": sum(1 for e in table["tensors"].values() if e.get("group") == "UNKNOWN"),
    }
    if meta_error:
        weights_source["error"] = meta_error
    budget_bw_bytes = int(cfg.budget_bw_gb * 1024**3) if cfg.budget_bw_gb is not None else None
```
then unpack the 5-tuple, pass `anchor_stats=anchor_stats` to `_build_and_verify_anchors`, and `bandwidth=solution` to `_assemble_results`. Import at the top of `search.py`: `from magicquant.v2.bandwidth import BandwidthSolution, pure_loss, solve_bandwidth, stream_weights, streamed_bytes`. Note: the characterization suite does NOT monkeypatch `open_model_source` (its `_install_stubs` at 137-206 patches only `LlamaCppTools`, `create_hybrid_gguf`, `compute_distortion_table`, `ensure_imatrix`), which is exactly why the read above is fail-soft. For the MoE-shaped budget-mode test, monkeypatch `magicquant.gguf.source.open_model_source` to return a `StubSource([], metadata=QWEN_MD)` (from `tests.test_writer`) so real weights are derived.

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_v2_search_characterization.py tests/test_v2_bandwidth.py tests/test_allocation_and_corpus_guards.py -q`
Expected: all pass; in the characterization suite **only** the two key-set pins changed.

- [ ] **Step 5: Full gates and commit**

```bash
git add magicquant/v2/search.py tests/test_v2_bandwidth.py tests/test_v2_search_characterization.py
git commit -m "feat(v2): streamed-bytes budget in the allocation pipeline" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 6: CLI and settings wiring for v2

**Dispatch:** DEPENDS-ON Task 5

**Files:**
- Modify: `magicquant/config.py` (after `budget_gb`, line 74)
- Modify: `magicquant/__main__.py` — `_maybe` block (~163), `_run_v2_search` (215-262), argparse near `--floor` (1012-1019)
- Test: `tests/test_config_routing.py` (append)

**Interfaces:**
- Consumes: Task 5's `V2Config` fields.
- Produces: `MagicQuantSettings.budget_bw_gb: Optional[float] = None` (env `MAGICQUANT_BUDGET_BW_GB`); flags `--budget-bw-gb`, `--bandwidth-weight`, `--stream-weight G=W` (repeatable).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_routing.py` (mirror `test_cmd_search_v2_real_parser_warns_ignored_v1_flags_and_still_runs` at ~366): (a) `--algo v2 --budget-gb 5 --budget-bw-gb 2 --stream-weight H=0.5 --stream-weight X=0.1` → the captured `cfg` has `budget_bw_gb == 2.0`, `stream_weights == {"H": 0.5, "X": 0.1}`, `bandwidth_weight == 0.0`; (b) `--bandwidth-weight 0.05` → `cfg.bandwidth_weight == 0.05`; (c) both flags → `SystemExit`; (d) `--stream-weight X=abc` → `SystemExit`; (e) `--stream-weight X=2` → `SystemExit`; (f) `MAGICQUANT_BUDGET_BW_GB=1.5` in env with no flag → `cfg.budget_bw_gb == 1.5`.

- [ ] **Step 2: Run to verify it fails**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_config_routing.py -q -k "bw or bandwidth or stream_weight"`
Expected: FAIL (unrecognized arguments).

- [ ] **Step 3: Implement**

`config.py`, after `budget_gb`:
```python
    # Streamed-bytes budget in GiB/token for --algo v2 (docs/redesign.md
    # section 11). None = off.
    budget_bw_gb: Optional[float] = None
```
`__main__.py`: `_maybe("budget_bw_gb", "budget_bw_gb")` after the `budget_gb` line. In `_run_v2_search`, after the `--budget-gb` guard:
```python
    bandwidth_weight = getattr(args, "bandwidth_weight", None)
    if settings.budget_bw_gb is not None and bandwidth_weight is not None:
        raise SystemExit("--budget-bw-gb and --bandwidth-weight are mutually exclusive")

    stream_weights = {}
    for spec in (getattr(args, "stream_weight", None) or []):
        if "=" not in spec:
            raise SystemExit(f"--stream-weight expects GROUP=W, got {spec!r}")
        g, v = spec.split("=", 1)
        try:
            w = float(v)
        except ValueError:
            raise SystemExit(f"--stream-weight expects a float W, got {v!r}")
        if not (0.0 <= w <= 1.0):
            raise SystemExit(f"--stream-weight W must be in [0,1], got {w!r}")
        stream_weights[g.strip()] = w
```
and in the `V2Config(...)` call: `bandwidth_weight=float(bandwidth_weight or 0.0), budget_bw_gb=settings.budget_bw_gb, stream_weights=stream_weights,`. Argparse, next to `--floor`:
```python
    search_parser.add_argument("--budget-bw-gb", dest="budget_bw_gb", type=float, default=None,
        help="[v2] streamed-bytes budget in GiB per decode token: bisect the "
             "bandwidth term so the trunk+active-expert bytes read per token fit "
             "(default: MAGICQUANT_BUDGET_BW_GB or off). Mutually exclusive with --bandwidth-weight")
    search_parser.add_argument("--bandwidth-weight", dest="bandwidth_weight", type=float, default=None,
        help="[v2] fixed lambda for the streamed-bytes term, in loss per streamed GiB "
             "(useful range ~0.003-0.1). Mutually exclusive with --budget-bw-gb")
    search_parser.add_argument("--stream-weight", action="append", default=None, metavar="GROUP=W",
        help="[v2] override a group's stream weight in [0,1], repeatable (e.g. --stream-weight H=0.5)")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_config_routing.py -q`

- [ ] **Step 5: Full gates and commit**

```bash
git add magicquant/config.py magicquant/__main__.py tests/test_config_routing.py
git commit -m "feat(cli): --budget-bw-gb, --bandwidth-weight and --stream-weight for v2" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 7: v1 opt-in stream-weighted speed proxy (`--stream-tps`)

**Dispatch:** DEPENDS-ON Task 6

**Files:**
- Modify: `magicquant/evolution/predictor.py` (`__init__` 57-67, add `predict_stream_gb` after `predict_size` 170-187, `score_hybrid` 395-445)
- Modify: `magicquant/evolution/survival.py` (`__init__` ~175 and `_predict_population` 734-757)
- Modify: `magicquant/orchestrator.py` (`~576`, `~1172`, `1572`, `~2654`, `~2784`, `3450`)
- Modify: `magicquant/config.py` (after `use_bytes_tps`, line 61), `magicquant/__main__.py` (`_maybe` ~159, `_V2_IGNORED_V1_FLAGS` 180-197, the two orchestrator forwarding sites `run_measured_search(... use_bytes_tps=settings.use_bytes_tps ...)` at ~320 and `run_full_search(...)` at ~341, argparse after `--bytes-tps` 912-919)
- Test: `tests/test_tps_objective.py` (append), `tests/test_config_routing.py` (append)

**Interfaces:**
- Consumes: Task 3's `stream_weights_by_group`.
- Produces: `PredictiveScorer(stream_weights=...)`, `PredictiveScorer.predict_stream_gb`, `PredictiveScorer.baseline_stream_gb`, `score_hybrid(..., use_stream_tps=False)`, `MagicQuantSettings.use_stream_tps`, `--stream-tps`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tps_objective.py`:
```python
def _stream_scorer(**kw):
    return PredictiveScorer({"U": 1.0, "X": 1.0, "H": 1.0},
                            parameter_counts={"U": 1_000, "X": 30_000, "H": 500},
                            baseline_size_gb=10.0, **kw)


def test_predict_stream_gb_equals_predict_size_without_weights():
    s = _stream_scorer()
    cfg = {"U": "Q8_0", "X": "Q4_K_M", "H": "Q6_K"}
    assert s.predict_stream_gb(cfg) == pytest.approx(s.predict_size(cfg), abs=1e-9)
    assert s.baseline_stream_gb == pytest.approx(s.baseline_size_gb)


def test_predict_stream_gb_discounts_cold_experts():
    s = _stream_scorer(stream_weights={"X": 0.03})
    cfg = {"U": "Q8_0", "X": "Q4_K_M", "H": "Q6_K"}
    assert s.predict_stream_gb(cfg) < 0.2 * s.predict_size(cfg)


def test_use_stream_tps_changes_score_only_with_weights():
    cfg = {"U": "Q8_0", "X": "Q4_K_M", "H": "Q6_K"}
    a = _stream_scorer().score_hybrid(cfg, use_bytes_tps=True)["tps_score"]
    b = _stream_scorer().score_hybrid(cfg, use_stream_tps=True)["tps_score"]
    assert a == pytest.approx(b)
    c = _stream_scorer(stream_weights={"X": 0.03}).score_hybrid(cfg, use_stream_tps=True)["tps_score"]
    assert c != pytest.approx(a)
    assert 0.0 <= c <= 1.0


def test_survivor_with_only_use_stream_tps_reaches_score_hybrid():
    # the survival.py:739 gate: use_stream_tps alone must take the tunable path
    s = _stream_scorer(stream_weights={"X": 0.03})
    pop = [{"config": {"U": "Q8_0", "X": "Q4_K_M", "H": "Q6_K"}}]
    default = EvolutionarySurvivor(s)._predict_population([dict(pop[0])])[0]["tps_score"]
    streamed = EvolutionarySurvivor(s, use_stream_tps=True)._predict_population([dict(pop[0])])[0]["tps_score"]
    assert streamed != pytest.approx(default)


def test_use_stream_tps_wins_over_use_bytes_tps_with_warning(caplog):
    import logging
    cfg = {"U": "Q8_0", "X": "Q4_K_M", "H": "Q6_K"}
    s = _stream_scorer(stream_weights={"X": 0.03})
    with caplog.at_level(logging.WARNING, logger="magicquant.evolution.predictor"):
        both = s.score_hybrid(cfg, use_bytes_tps=True, use_stream_tps=True)["tps_score"]
    assert both == pytest.approx(s.score_hybrid(cfg, use_stream_tps=True)["tps_score"])
    assert any("use_stream_tps" in r.getMessage() for r in caplog.records)
```
Append to `tests/test_config_routing.py`: `--algo v2 --stream-tps` names `--stream-tps` in the ignored-flags WARNING (copy the existing test's shape).

- [ ] **Step 2: Run to verify it fails**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_tps_objective.py -q -k stream`
Expected: FAIL (`unexpected keyword argument 'stream_weights'`). (If `EvolutionarySurvivor(s)` needs more constructor arguments than the predictor, copy the construction the existing `test_predict_population_*` tests in this file use — read lines 225-260 first.)

- [ ] **Step 3: Implement**

`predictor.py`: add `stream_weights: Optional[Dict[str, float]] = None` to `__init__` and store `self.stream_weights = dict(stream_weights or {})`. Add:
```python
    @property
    def baseline_stream_gb(self) -> float:
        """BF16 baseline size weighted by how often each group is read per
        decode token (literal 16 bpw; writer-compat rewrites never enter the
        baseline). Equals baseline_size_gb when no weights are set."""
        if not self.stream_weights or not self.parameter_counts or not self.baseline_size_gb:
            return self.baseline_size_gb
        total = sum(self.parameter_counts.values())
        if total <= 0:
            return self.baseline_size_gb
        weighted = sum(self.stream_weights.get(g, 1.0) * p for g, p in self.parameter_counts.items())
        return self.baseline_size_gb * weighted / total

    def predict_stream_gb(self, group_schemes: Dict[str, str]) -> float:
        """predict_size with each group's bytes scaled by its stream weight."""
        if not self.baseline_size_gb or not self.parameter_counts:
            return self.predict_size(group_schemes)
        total_weighted_bits = 0.0
        total_params = sum(self.parameter_counts.values())
        for group, scheme in group_schemes.items():
            params_in_group = self.parameter_counts.get(group, 0)
            bits = self._bpw_for(group, scheme)
            total_weighted_bits += self.stream_weights.get(group, 1.0) * params_in_group * bits
        if total_params > 0:
            return self.baseline_size_gb * ((total_weighted_bits / total_params) / 16.0)
        return self.predict_size(group_schemes)
```
`score_hybrid(..., use_bytes_tps=False, use_stream_tps=False)`: before the `if use_bytes_tps:` branch:
```python
        if use_stream_tps and use_bytes_tps:
            log.warning("score_hybrid: use_stream_tps overrides use_bytes_tps")
        if use_stream_tps:
            speedup = self.baseline_stream_gb / max(self.predict_stream_gb(group_schemes), self._BYTES_TPS_EPS)
            tps_score = (speedup - 1.0) / (self._BYTES_TPS_MAX_SPEEDUP - 1.0)
            tps_score = min(1.0, max(0.0, tps_score))
        elif use_bytes_tps:
            ...  # unchanged
```
(`predictor.py` has no module logger. Use **stdlib** logging — `import logging` and `log = logging.getLogger(__name__)` at the top, the pattern `magicquant/gguf/tensor_groups.py:24` uses — NOT `magicquant.logging.get_logger`, which is structlog on a `PrintLoggerFactory` and never reaches `caplog`.)
`survival.py`: `use_stream_tps: bool = False` in `__init__` stored as `self.use_stream_tps`; in `_predict_population`, `tunable = self.objective_weights is not None or self.use_bytes_tps or self.use_stream_tps` and pass `use_stream_tps=self.use_stream_tps` in the tunable call.
`orchestrator.py`: thread `use_stream_tps` exactly where `use_bytes_tps` is threaded (`576`, `765` docstring, `1172`, `2654`, `2682`, `2784`); at `3450`, after `self._effective_bpw = ...`: 
```python
                try:
                    from magicquant.v2.bandwidth import stream_weights_by_group
                    self._stream_weights = stream_weights_by_group(src.get_metadata(), log=log)
                except Exception as exc:  # never fatal: today's pricing
                    log.warning("stream weights unavailable (%s); v1 prices every byte as streamed", exc)
                    self._stream_weights = None
```
(initialise `self._stream_weights = None` where `self._effective_bpw` is initialised) and pass `stream_weights=self._stream_weights` in the `PredictiveScorer(...)` call at `1572`.
`config.py`: `use_stream_tps: bool = False` after `use_bytes_tps`. `__main__.py`: `_maybe("use_stream_tps", "use_stream_tps")`; `use_stream_tps=settings.use_stream_tps,` immediately after the `use_bytes_tps=settings.use_bytes_tps,` line at BOTH ~320 (`run_measured_search`) and ~341 (`run_full_search`); `("--stream-tps", "use_stream_tps")` in `_V2_IGNORED_V1_FLAGS`; argparse after `--bytes-tps`:
```python
    search_parser.add_argument("--stream-tps", dest="use_stream_tps", action="store_true", default=None,
        help="Score speed from predicted STREAMED bytes per decode token (trunk read every "
             "token, routed experts n_used/n_expert of the time) instead of stored size; "
             "MoE-correct variant of --bytes-tps (default: MAGICQUANT_USE_STREAM_TPS or off)")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/server/programming/MagicQuant/.venv/bin/python -m pytest tests/test_tps_objective.py tests/test_config_routing.py tests/test_refactor_regression.py tests/test_stream_aware.py -q`
Expected: all pass; `test_refactor_regression.py` passes without touching its fixture.

- [ ] **Step 5: Full gates and commit**

```bash
git add magicquant/evolution/predictor.py magicquant/evolution/survival.py magicquant/orchestrator.py magicquant/config.py magicquant/__main__.py tests/test_tps_objective.py tests/test_config_routing.py
git commit -m "feat(evolution): opt-in stream-weighted speed proxy (--stream-tps)" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```

---

### Task 8: README, validation.md header, CHANGELOG

**Dispatch:** DEPENDS-ON Task 1, Task 2, Task 3, Task 4, Task 5, Task 6, Task 7

**Files:**
- Modify: `README.md` (new section "Streamed bytes and MoE" near the existing v2 section; check `grep -n "v2" README.md`)
- Modify: `docs/validation.md:1-5` (status header)
- Modify: `docs/redesign.md` §11 (add a "See README" cross-link line)
- Modify: `CHANGELOG.md` (`[Unreleased]`)

**Interfaces:** none.

- [ ] **Step 1: Capture the failing check**

Run: `grep -n "Streamed bytes and MoE" README.md; grep -n "streamed-bytes" CHANGELOG.md`
Expected: no output.

- [ ] **Step 2: Write the docs**

README section: the §1.1 table from `docs/redesign.md` §11, the two-command recipe
```
magicquant search <model> --algo v2 --budget-gb 19.57 --budget-bw-gb 2.1 --probe-mode cumulative --floor E=Q6_K --use-imatrix
magicquant search <model> --stream-tps --speed-weight 0.3        # v1 route
```
, what the `bandwidth` block reports, and a link to `docs/redesign.md#11`. `docs/validation.md`: change `**Status: COMPLETE.**` to `**Status: COMPLETE for the 2026-07 v1/v2 matched-size study; a streamed-bytes campaign (docs/redesign.md section 11) is open and will be appended below.**`. CHANGELOG under `[Unreleased]`, new subsection `### Added (2026-09-11 streamed-bytes allocation)` with one bullet per commit (Tasks 1–7), each ending with `Files:` and `Validation:` lines (the full-suite count from Task 7's run and "ruff --select F clean"), and stating explicitly that `v2_results.json` gained the top-level `bandwidth` key and `frontier.json` gained `lambda`, both pins updated deliberately.

- [ ] **Step 3: Verify and commit**

Run: `grep -n "Streamed bytes and MoE" README.md && grep -n "streamed-bytes" CHANGELOG.md && /server/programming/MagicQuant/.venv/bin/python -m pytest tests/ -q`

```bash
git add README.md docs/validation.md docs/redesign.md CHANGELOG.md
git commit -m "docs: streamed-bytes allocation guide and changelog" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01G3vPPyNDc16w6eXVT6s11o"
```
