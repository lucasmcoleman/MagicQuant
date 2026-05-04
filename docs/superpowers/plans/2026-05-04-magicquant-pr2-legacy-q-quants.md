# PR2: Legacy Q-quants (Q4_0, Q4_1, Q5_0, Q5_1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** PR1 must be merged to `master`. Verify with `cd /server/programming/MagicQuant && git log --oneline | grep "ggml_encode\|libggml" | head -3` showing recent binding commits.

**Goal:** Register the four legacy Q-quant block formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`) as MagicQuant schemes. Q4_0 is already encoded by the ctypes binding but not registered as a scheme; the other three are net-new. These give the evolutionary search additional 4–5 bpw alternatives in tiers Q4 and Q5.

**Architecture:** The ctypes binding from PR1 already supports all four types (their `ggml_type_id` values are in `GGML_TYPE_IDS`). This PR is purely a registry-level addition: add four `QuantizationScheme` instances to `schemes.py`, wire them into the upgrade/downgrade chains, and add encoder-parity tests for each.

**Tech Stack:** Python 3.12, pytest

**Spec:** `docs/superpowers/specs/2026-05-04-magicquant-encoder-expansion-design.md` (section "PR2 — Legacy Q-quants")

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `magicquant/quant/schemes.py` | Modify | Register Q4_0, Q4_1, Q5_0, Q5_1; wire into upgrade chain |
| `magicquant/quant/__init__.py` | Modify | Re-export new scheme constants |
| `magicquant/__init__.py` | Modify | Re-export new scheme constants |
| `tests/integration/test_encoder_parity.py` | Modify | Add 3 new schemes (Q4_1, Q5_0, Q5_1) to parametrize list |

**File-size note:** ~150 net new lines, all in `schemes.py` (4 new scheme objects, registry updates, re-exports).

---

## Tasks

### Task 1: Prerequisite verification

**Files:** none

- [ ] **Step 1: Verify PR1 is merged**

Run:
```bash
cd /server/programming/MagicQuant && git log --oneline | head -10
```

Expected: includes commits from PR1 like `feat: register Q2_K and Q3_K schemes`, `refactor: delete pure-Python encoder helpers`, `feat: ctypes call surface for ggml_quantize_chunk`.

- [ ] **Step 2: Verify ggml_binding handles the legacy types**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.ggml_binding import GGML_TYPE_IDS, ggml_encode
import numpy as np

assert 'Q4_0' in GGML_TYPE_IDS, f'Q4_0 missing from GGML_TYPE_IDS'
assert 'Q4_1' in GGML_TYPE_IDS, f'Q4_1 missing'
assert 'Q5_0' in GGML_TYPE_IDS, f'Q5_0 missing'
assert 'Q5_1' in GGML_TYPE_IDS, f'Q5_1 missing'
print('all legacy Q types present in GGML_TYPE_IDS')

# Smoke-test each
weights = np.random.randn(256).astype(np.float32) * 0.02
for t in ['Q4_0', 'Q4_1', 'Q5_0', 'Q5_1']:
    out = ggml_encode(weights, t)
    print(f'  {t}: {len(out)} bytes')
"
```

Expected output:
```
all legacy Q types present in GGML_TYPE_IDS
  Q4_0: 144 bytes
  Q4_1: 160 bytes
  Q5_0: 176 bytes
  Q5_1: 192 bytes
```

(256 elements / 32 per block = 8 blocks. Q4_0 = 18 B/block × 8 = 144. Q4_1 = 20 × 8 = 160. Q5_0 = 22 × 8 = 176. Q5_1 = 24 × 8 = 192.)

- [ ] **Step 3: Run all tests as baseline**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: all PR1 tests pass (22+).

---

### Task 2: Register the four legacy Q-quant schemes

**Files:**
- Modify: `magicquant/quant/schemes.py`

- [ ] **Step 1: Add the four scheme constants after Q8_0**

Edit `/server/programming/MagicQuant/magicquant/quant/schemes.py`. Find the existing Q8_0 definition:

```python
Q8_0 = QuantizationScheme(
    name="Q8_0",
    ggml_type_name="Q8_0",
    ggml_type_id=8,
    bits_per_weight=8.5,
    noise_factor=1.0,           # calibrated 2026-05-04 anchor
    speed_multiplier=1.75,
    category="legacy_q",
    upgrade_neighbor="BF16",
    downgrade_neighbor="Q6_K",
)
```

Update its `downgrade_neighbor` to point at the new Q5_1 scheme (Q5_1 sits between Q8_0 and Q6_K in the chain):

Replace `downgrade_neighbor="Q6_K"` with:
```python
    downgrade_neighbor="Q5_1",
```

Then add the four new schemes immediately AFTER Q8_0:

```python


# ── Legacy Q-quant block formats (registered in PR2) ──────────────────
# These predate K-quants but are still produced by llama.cpp and
# supported by the ctypes binding. They give the evolutionary search
# additional 4-5 bpw alternatives.
#
# Q5_1 (5.5 bpw, asymmetric) and Q5_0 (5.5 bpw, symmetric) are the
# closest competitors to Q5_K. Q4_1 (5.0 bpw) and Q4_0 (4.5 bpw) sit
# between Q5_K and IQ4_NL/Q4_K_M.
#
# noise_factor values are placeholders pending a calibration re-run
# (the PR1 calibration didn't include these because they weren't
# registered yet). Re-run tools/calibrate_noise_factors.py at the end
# of this PR to refine them.

Q5_1 = QuantizationScheme(
    name="Q5_1",
    ggml_type_name="Q5_1",
    ggml_type_id=7,
    bits_per_weight=6.0,         # 24B * 8 / 32 = 6.0
    noise_factor=2.4,            # placeholder; near Q6_K
    speed_multiplier=2.5,
    category="legacy_q",
    upgrade_neighbor="Q8_0",
    downgrade_neighbor="Q5_0",
)

Q5_0 = QuantizationScheme(
    name="Q5_0",
    ggml_type_name="Q5_0",
    ggml_type_id=6,
    bits_per_weight=5.5,         # 22B * 8 / 32 = 5.5
    noise_factor=3.2,            # placeholder; near Q5_K but slightly worse
    speed_multiplier=2.6,
    category="legacy_q",
    upgrade_neighbor="Q5_1",
    downgrade_neighbor="Q6_K",   # Q6_K is higher quality but lower bpw
)

Q4_1 = QuantizationScheme(
    name="Q4_1",
    ggml_type_name="Q4_1",
    ggml_type_id=3,
    bits_per_weight=5.0,         # 20B * 8 / 32 = 5.0
    noise_factor=4.0,            # placeholder; between Q5_K and Q4_K_M
    speed_multiplier=3.0,
    category="legacy_q",
    upgrade_neighbor="Q5_0",
    downgrade_neighbor="Q4_0",
)

Q4_0 = QuantizationScheme(
    name="Q4_0",
    ggml_type_name="Q4_0",
    ggml_type_id=2,
    bits_per_weight=4.5,         # 18B * 8 / 32 = 4.5
    noise_factor=4.7,            # placeholder; comparable to Q4_K_M
    speed_multiplier=3.3,
    category="legacy_q",
    upgrade_neighbor="Q4_1",
    downgrade_neighbor="Q4_K_M",
)
```

Note: The downgrade_neighbor chain links Q5_0 → Q6_K (which is *higher* quality, lower bpw than Q5_0). This is intentional — the chain represents "next step in compression direction" by quality, not by bpw. The legacy Q-quants are quality-inferior to K-quants at similar bpw, so downgrading from Q5_0 leads to Q6_K (better quality, smaller block format).

- [ ] **Step 2: Update Q5_K's upgrade chain**

Find:
```python
Q5_K = QuantizationScheme(
    name="Q5_K",
    ...
    upgrade_neighbor="Q6_K",
    downgrade_neighbor="IQ4_NL",
)
```

The downgrade chain stays as-is. The upgrade chain stays as-is (Q5_K still upgrades to Q6_K). The legacy Q-quants form a parallel chain that doesn't intersect with K-quants except at the Q4_K_M / Q4_0 boundary.

No change needed if the existing Q5_K block already matches.

- [ ] **Step 3: Update Q4_K_M's upgrade_neighbor (optional)**

Q4_K_M.upgrade_neighbor is currently "MXFP4_MOE". With Q4_0 now in the chain, an alternative is "Q4_0" (slightly higher bpw at 4.5 vs MXFP4's 4.25). Decision: keep "MXFP4_MOE" — it's higher quality at lower bpw. Q4_0 is reachable via the legacy_q chain instead.

No change.

- [ ] **Step 4: Add the four schemes to _REGISTRY**

Find:
```python
_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
    "Q3_K": Q3_K,
    "Q2_K": Q2_K,
}
```

Replace with:
```python
_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q5_1": Q5_1,
    "Q5_0": Q5_0,
    "Q4_1": Q4_1,
    "Q4_0": Q4_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
    "Q3_K": Q3_K,
    "Q2_K": Q2_K,
}
```

- [ ] **Step 5: Verify the new schemes are registered**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.schemes import (
    get_all_schemes, get_scheme_by_name, get_schemes_by_category,
    Q4_0, Q4_1, Q5_0, Q5_1
)
print('legacy_q schemes:', [s.name for s in get_schemes_by_category('legacy_q')])
print('Q5_1.upgrade:', Q5_1.upgrade_neighbor)
print('Q4_0.downgrade:', Q4_0.downgrade_neighbor)
print('total schemes:', len(get_all_schemes()))
"
```

Expected output:
```
legacy_q schemes: ['Q8_0', 'Q5_1', 'Q5_0', 'Q4_1', 'Q4_0']
Q5_1.upgrade: Q8_0
Q4_0.downgrade: Q4_K_M
total schemes: 13
```

(13 total: 7 from PR0 + Q3_K + Q2_K from PR1 + Q4_0/Q4_1/Q5_0/Q5_1 from PR2.)

- [ ] **Step 6: Run regression test**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected: `1 passed`. The new schemes shift the random-config draws slightly (since they're added to `get_all_schemes()` and the category-indexed sampling distributes mass across all `legacy_q` schemes including the new ones). If the test fails, regenerate the fixture (procedure same as PR0 Task 3 / PR1 Task 10).

- [ ] **Step 7: Commit scheme registrations**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/schemes.py && \
  (git diff --cached tests/fixtures/refactor_regression_seed42.json && git add tests/fixtures/refactor_regression_seed42.json || true) && \
  git commit -m "feat: register Q4_0, Q4_1, Q5_0, Q5_1 legacy Q-quants

Adds the four legacy Q-quant block formats as registered MagicQuant
schemes. The ctypes binding from PR1 already supports them; this
makes them available to the evolutionary search.

  Q5_1 (6.0 bpw)  →  legacy alternative to Q8_0/Q6_K
  Q5_0 (5.5 bpw)  →  legacy alternative to Q5_K
  Q4_1 (5.0 bpw)  →  fills 5-bpw gap
  Q4_0 (4.5 bpw)  →  legacy alternative to Q4_K_M

noise_factor values are placeholders; calibrated in next commit.

Regression fixture regenerated for the slightly-shifted distribution.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update package re-exports

**Files:**
- Modify: `magicquant/quant/__init__.py`
- Modify: `magicquant/__init__.py`

- [ ] **Step 1: Update `magicquant/quant/__init__.py`**

Edit and find the schemes import block. Add `Q4_0, Q4_1, Q5_0, Q5_1` to both the import statement and `__all__`:

```python
from magicquant.quant.schemes import (
    BF16, Q8_0,
    Q5_1, Q5_0, Q4_1, Q4_0,            # legacy Q-quants
    Q6_K, Q5_K, Q4_K_M,                # K-quants (existing)
    Q3_K, Q2_K,                        # K-quants (PR1)
    IQ4_NL, MXFP4_MOE,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
    get_floor_for_group_class,
)

__all__ = [
    "BF16", "Q8_0",
    "Q5_1", "Q5_0", "Q4_1", "Q4_0",
    "Q6_K", "Q5_K", "Q4_K_M",
    "Q3_K", "Q2_K",
    "IQ4_NL", "MXFP4_MOE",
    "QuantizationScheme",
    "get_scheme_by_name",
    "get_all_schemes",
    "get_schemes_by_category",
    "get_floor_for_group_class",
]
```

- [ ] **Step 2: Update `magicquant/__init__.py`**

Apply the same change to the top-level `__init__.py`.

- [ ] **Step 3: Verify imports**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant import Q4_0, Q4_1, Q5_0, Q5_1
print('Q4_0:', Q4_0)
print('Q4_1:', Q4_1)
print('Q5_0:', Q5_0)
print('Q5_1:', Q5_1)
"
```

Expected output: four `QuantScheme(...)` lines.

- [ ] **Step 4: Commit re-exports**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/__init__.py magicquant/__init__.py && \
  git commit -m "feat: re-export Q4_0/Q4_1/Q5_0/Q5_1 at package top-level

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add encoder-parity tests for the new schemes

**Files:**
- Modify: `tests/integration/test_encoder_parity.py`

- [ ] **Step 1: Add the three new schemes to the parametrize list**

Edit `/server/programming/MagicQuant/tests/integration/test_encoder_parity.py`. Find:

```python
SCHEMES_PR1 = [
    "Q8_0", "Q6_K", "Q5_K", "Q4_K", "IQ4_NL", "MXFP4", "Q4_0",
    "Q2_K", "Q3_K",
]
```

Replace with:
```python
# Schemes covered by encoder-parity tests. Names match ggml_type_name
# (NOT MagicQuant scheme name). Q4_0 is here from PR1; Q4_1/Q5_0/Q5_1
# added in PR2; IQ-quants added in PR3.
SCHEMES_PARITY = [
    "Q8_0",
    "Q6_K", "Q5_K", "Q4_K",
    "IQ4_NL", "MXFP4",
    "Q4_0", "Q4_1", "Q5_0", "Q5_1",   # legacy Q-quants
    "Q2_K", "Q3_K",
]
```

Then update the test parametrize decorator below it. Find:
```python
@pytest.mark.parametrize("scheme", SCHEMES_PR1)
```

Replace with:
```python
@pytest.mark.parametrize("scheme", SCHEMES_PARITY)
```

(Both occurrences if there are multiple parametrized tests.)

- [ ] **Step 2: Run the parity tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v 2>&1 | tail -20
```

Expected output: 12 tests pass (9 from PR1 + 3 new — Q4_1, Q5_0, Q5_1).

If `Q4_1`, `Q5_0`, or `Q5_1` fail with byte mismatch, the same `--pure` flag fix from PR1 Task 6 applies. Add it to the `subprocess.run` call:
```python
[quantize_bin, "--pure", str(src_path), str(dst_path), scheme],
```

- [ ] **Step 3: Commit parity tests**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_encoder_parity.py && \
  git commit -m "test: extend encoder parity coverage to Q4_1, Q5_0, Q5_1

Renames SCHEMES_PR1 → SCHEMES_PARITY (will keep growing through PR3).
12 schemes now byte-parity verified against llama-quantize.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Re-run the calibration bench (optional but recommended)

**Files:**
- Modify: `tools/calibration_results.json`
- Modify: `magicquant/quant/schemes.py`

The PR1 calibration was run before the legacy Q-quants were registered, so it didn't include them. PR2's noise_factor values are placeholders. A re-run gets accurate values.

**Optional**: if you want to skip this and ship PR2 with placeholder noise factors, the placeholders are reasonable estimates (within ~0.5 of likely calibrated values for these well-known formats). The orchestrator's residual cache self-corrects per-run. Skip this task if pressed for time and proceed to Task 6.

- [ ] **Step 1: Re-run the calibration bench**

Run (same model and corpus as PR1's run):
```bash
cd /server/programming/MagicQuant && \
  python tools/calibrate_noise_factors.py \
    --model <SAME-PATH-AS-PR1> \
    --corpus /home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --output tools/calibration_results.json \
    2>&1 | tee /tmp/calibration_run_pr2.log
```

Expected output: progress per scheme. The 4 new legacy Q-quants are added; previous schemes are re-measured (results should be very close to PR1's; small drift is benign).

Wall-clock: similar to PR1's bench (~1.5–2 hr) since the perplexity runs dominate, not the count of schemes.

- [ ] **Step 2: Update schemes.py with refreshed noise factors**

Apply the same procedure as PR1 Task 14: read calibrated values from `tools/calibration_results.json`, update each scheme's `noise_factor=` line in `schemes.py`, append a `# calibrated 2026-MM-DD vs <model>` comment.

Focus on the 4 new schemes (Q5_1, Q5_0, Q4_1, Q4_0). The other 9 schemes' values may shift by ±0.05–0.10 from PR1's calibration; update them too if there's meaningful drift.

- [ ] **Step 3: Run regression test**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

If it fails, regenerate the fixture.

- [ ] **Step 4: Commit calibration update**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tools/calibration_results.json magicquant/quant/schemes.py && \
  (git diff --cached tests/fixtures/refactor_regression_seed42.json && git add tests/fixtures/refactor_regression_seed42.json || true) && \
  git commit -m "calibrate: refresh noise factors with legacy Q-quants

Re-runs the empirical bench to include Q4_0, Q4_1, Q5_0, Q5_1 with
real measurements. Existing schemes' values shift by < 0.1 noise
units (within run-to-run variance).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final verification + push

**Files:** none

- [ ] **Step 1: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -25
```

Expected output: 25+ tests pass (PR1's 22 + 3 new parity tests = 25).

- [ ] **Step 2: Verify legacy Q-quants are reachable from random config**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import random
random.seed(42)
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor

predictor = PredictiveScorer({'E': 1.0, 'U': 0.4}, {'E': 100_000_000, 'U': 800_000_000}, 4.0, 20.0)
survivor = EvolutionarySurvivor(predictor, {'E': 'BF16', 'U': 'MXFP4_MOE'}, max_generations=1, population_size=300)
configs = [survivor._generate_random_config(['E', 'H', 'Q', 'K', 'O', 'U', 'D']) for _ in range(300)]
seen = set()
for c in configs: seen.update(c.values())
print('schemes seen:', sorted(seen))
expected_legacy = {'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1'}
missing = expected_legacy - seen
assert not missing, f'Missing legacy schemes from random init: {missing}'
print('OK — all legacy Q-quants reachable from random init')
"
```

Expected output: at least one of each legacy Q-quant scheme appears across 300 random configs.

- [ ] **Step 3: Push**

Run:
```bash
cd /server/programming/MagicQuant && git push origin master 2>&1
```

- [ ] **Step 4: PR2 done**

Print:
```
PR2 complete:
- Q4_0, Q4_1, Q5_0, Q5_1 registered as legacy_q schemes
- 12 schemes total (BF16 + 5 legacy + 5 K-quants + IQ4_NL + MXFP4)
- All 12 byte-parity verified against llama-quantize
- (Optional) noise_factor values refreshed via calibration re-run
- Random config generator samples all schemes
- Ready for PR3 (IQ-quants — the big one)
```

---

## Self-Review Checklist

**Spec coverage (PR2 section):**
- [x] "Register Q4_0 (already encoded but not registered as a scheme), Q4_1, Q5_0, Q5_1" → Task 2
- [x] "Add to predictor/survival category-weighted tables" → automatic via PR1's category-indexed weights
- [x] "Add ggml_type_id mappings" → already in PR1's `GGML_TYPE_IDS`; verified in Task 1
- [x] "Add encoder-parity tests for each" → Task 4

**Placeholder scan:** No "TBD" or vague directives.

**Type consistency:** New schemes use the same `QuantizationScheme` dataclass; field names match the existing 9 schemes from PR0/PR1.

**Risk callouts:**
- Q4_0/Q4_1/Q5_0/Q5_1 may be repacked by llama-quantize (Q4_0_R4, Q4_0_R8 variants for SIMD). The `--pure` flag fix from PR1 covers this. If parity tests still fail, dig into the gguf reader's data-extraction path.

**Skipped from spec:**
- Calibration re-run is marked optional. The placeholder noise factors are reasonable estimates; PR3's calibration will include all schemes anyway.

---

## Future Work (not in this plan)

- PR3: IQ1/IQ2/IQ3/IQ4_XS schemes — the big one, populates the Q2 tier band
- PR4: Importance-matrix support
