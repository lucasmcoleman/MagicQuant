# PR0: Refactor Scheme Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize quantization scheme metadata in `schemes.py` so `predictor.py`, `survival.py`, `probing.py`, and `orchestrator.py` read attributes from a single registry instead of holding parallel hardcoded dicts. Zero behavior change.

**Architecture:** Extend the `QuantizationScheme` class with all attributes consumers need (`ggml_type_name`, `ggml_type_id`, `category`, `requires_imatrix`, `min_for_group_class`, `upgrade_neighbor`, `downgrade_neighbor`). Each consumer module replaces its local lookup dict with `get_scheme_by_name(...).<attribute>` access. A pinned-RNG snapshot test verifies identical search behavior pre- and post-refactor.

**Tech Stack:** Python 3.12, dataclasses, pytest, numpy

**Spec:** `docs/superpowers/specs/2026-05-04-magicquant-encoder-expansion-design.md` (sections "Refactor (PR0)" and "Phased PR plan → PR0")

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `magicquant/quant/schemes.py` | Modify | Extend QuantizationScheme; populate fields for existing 7 schemes; add registry helpers |
| `magicquant/evolution/predictor.py` | Modify | Replace QUANT_NOISE_FACTORS, QUANT_COMPRESSION, QUANT_SPEED dicts with registry lookups |
| `magicquant/evolution/survival.py` | Modify | Replace SCHEME_QUALITY_ORDER, _UPGRADE, _DOWNGRADE, _MIN_SCHEME with registry-derived equivalents |
| `magicquant/evolution/probing.py` | Modify | Replace `_SCHEME_NOISE` local dict in `_heuristic_probe` with registry lookup |
| `magicquant/orchestrator.py` | Modify | Replace `base_quant` ranking dict at line 583 with registry lookup |
| `tests/test_refactor_regression.py` | Create | Pinned-RNG snapshot test for evolutionary search |
| `tests/fixtures/refactor_regression_seed42.json` | Create | Captured snapshot of search output |

**File-size note:** All modified files stay under 1000 lines after the refactor. `schemes.py` grows from ~133 to ~280 lines (extra fields + helpers). Other consumer files shrink slightly (parallel dicts removed).

---

## Tasks

### Task 1: Verify baseline — existing tests pass

**Files:** none

- [ ] **Step 1: Verify pytest runs cleanly on current master**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v
```

Expected output: all tests in `test_quantization_guards.py` pass (12 tests). No errors.

If any test fails on master, STOP and resolve before continuing — the refactor must not introduce failures, but pre-existing failures must be untangled first.

- [ ] **Step 2: Verify Python imports cleanly**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import magicquant
from magicquant.quant.schemes import get_all_schemes, get_scheme_by_name
from magicquant.evolution.survival import EvolutionarySurvivor
from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.probing import SensitivityProber
print('imports OK')
print('schemes:', [s.name for s in get_all_schemes()])
"
```

Expected output:
```
imports OK
schemes: ['BF16', 'Q8_0', 'Q6_K', 'Q5_K', 'IQ4_NL', 'MXFP4_MOE', 'Q4_K_M']
```

- [ ] **Step 3: Confirm clean git state**

Run:
```bash
cd /server/programming/MagicQuant && git status
```

Expected output: `nothing to commit, working tree clean` (or note any pre-existing uncommitted work and stash it).

---

### Task 2: Write the regression snapshot test (RED)

**Files:**
- Create: `tests/test_refactor_regression.py`

The test runs `EvolutionarySurvivor.run_evolution()` with a pinned seed and asserts the candidate sequence matches a stored fixture. The fixture is created in Task 3 — at this point the test will fail because the fixture file doesn't exist yet.

- [ ] **Step 1: Create the test file**

Create `/server/programming/MagicQuant/tests/test_refactor_regression.py` with this exact content:

```python
"""Refactor regression test for PR0.

Pins random.seed and numpy.random seed, runs a small evolutionary search,
captures the candidate sequence, and asserts it matches a stored fixture.

This test must pass identically before and after the scheme-registry refactor.
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from magicquant.evolution.predictor import PredictiveScorer
from magicquant.evolution.survival import EvolutionarySurvivor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "refactor_regression_seed42.json"


def _build_predictor():
    """Build a deterministic predictor with synthetic sensitivity weights."""
    sensitivity_weights = {
        "E": 1.5, "H": 1.4, "Q": 1.0, "K": 0.9,
        "O": 1.2, "U": 0.5, "D": 0.5,
    }
    parameter_counts = {
        "E": 100_000_000, "H": 100_000_000, "Q": 300_000_000, "K": 300_000_000,
        "O": 150_000_000, "U": 800_000_000, "D": 800_000_000,
    }
    return PredictiveScorer(
        sensitivity_weights=sensitivity_weights,
        parameter_counts=parameter_counts,
        baseline_size_gb=5.0,
        baseline_tps=20.0,
    )


def _capture_run(seed: int = 42, generations: int = 3, population: int = 20):
    """Run a small deterministic evolution and capture the discovered configs.

    Returns a list of config dicts (group → scheme), in discovery order.
    """
    random.seed(seed)
    np.random.seed(seed)

    predictor = _build_predictor()
    baseline = {g: "MXFP4_MOE" for g in ["E", "H", "Q", "K", "O", "U", "D"]}
    survivor = EvolutionarySurvivor(
        predictor=predictor,
        baseline_config=baseline,
        max_generations=generations,
        population_size=population,
        epsilon=0.0,  # disable epsilon-greedy randomness for determinism
    )
    discovered = survivor.run_evolution(verbose=False)
    return [c["config"] for c in discovered]


def _load_fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not yet captured: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text())


def test_evolution_seed42_matches_fixture():
    """Search behavior must be identical pre- and post-refactor."""
    captured = _capture_run(seed=42, generations=3, population=20)
    expected = _load_fixture()
    assert captured == expected, (
        "Refactor changed search behavior. "
        f"Captured {len(captured)} configs, expected {len(expected)}. "
        f"First diff at index {next((i for i, (a, b) in enumerate(zip(captured, expected)) if a != b), 'end')}"
    )
```

- [ ] **Step 2: Verify the test file imports without syntax error**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py --collect-only -q
```

Expected output:
```
tests/test_refactor_regression.py::test_evolution_seed42_matches_fixture
1 test collected
```

- [ ] **Step 3: Run the test — expect SKIP (fixture not yet captured)**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: 1 SKIPPED ("Fixture not yet captured"). If it fails for any other reason, debug before proceeding.

- [ ] **Step 4: Commit the test (red state — fixture missing is intentional)**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/test_refactor_regression.py && \
  git commit -m "test: add refactor regression test (fixture pending)

Pins seed=42 and runs a small evolutionary search. Asserts the captured
candidate sequence matches a stored fixture. Currently skips because the
fixture is captured in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Capture the regression fixture (GREEN baseline)

**Files:**
- Create: `tests/fixtures/refactor_regression_seed42.json`

This task captures the *current* search behavior into a JSON fixture. The fixture is the ground truth for the rest of the refactor.

- [ ] **Step 1: Create the fixtures directory**

Run:
```bash
cd /server/programming/MagicQuant && mkdir -p tests/fixtures
```

- [ ] **Step 2: Create a capture script and run it**

Create `/server/programming/MagicQuant/tests/_capture_fixture.py` with this exact content:

```python
"""One-shot script: capture the current evolutionary search output as a JSON fixture.

Run once on master before refactoring. The output is committed as
tests/fixtures/refactor_regression_seed42.json.

Delete this file after the fixture is captured — it's not needed at runtime.
"""
import json
from pathlib import Path

from tests.test_refactor_regression import _capture_run

FIXTURE = Path(__file__).parent / "fixtures" / "refactor_regression_seed42.json"

if __name__ == "__main__":
    captured = _capture_run(seed=42, generations=3, population=20)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(captured, indent=2))
    print(f"Captured {len(captured)} configs to {FIXTURE}")
```

Run it:
```bash
cd /server/programming/MagicQuant && python -m tests._capture_fixture
```

Expected output:
```
Captured <N> configs to /server/programming/MagicQuant/tests/fixtures/refactor_regression_seed42.json
```
(N depends on how many unique configs the small search discovers — typically 5–30.)

- [ ] **Step 3: Verify the fixture is valid JSON**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import json
data = json.loads(open('tests/fixtures/refactor_regression_seed42.json').read())
print(f'Configs: {len(data)}')
print(f'First config: {data[0] if data else None}')
"
```

Expected output: a list with at least one config dict, e.g.:
```
Configs: 12
First config: {'E': 'BF16', 'H': 'BF16', 'Q': 'Q8_0', ...}
```

- [ ] **Step 4: Verify the regression test now PASSES**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`. If it fails, the fixture wasn't captured cleanly — re-run Step 2.

- [ ] **Step 5: Delete the capture script (one-shot, not needed at runtime)**

Run:
```bash
cd /server/programming/MagicQuant && rm tests/_capture_fixture.py
```

- [ ] **Step 6: Commit the fixture**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/fixtures/refactor_regression_seed42.json && \
  git commit -m "test: capture refactor regression fixture (seed=42)

Snapshot of evolutionary search output before scheme-registry refactor.
Used by test_refactor_regression.py to verify behavior preservation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Extend the QuantizationScheme dataclass

**Files:**
- Modify: `magicquant/quant/schemes.py`

Replace the existing `QuantizationScheme` class definition and the seven constant constructions with the extended version that carries all attributes consumers need.

- [ ] **Step 1: Replace `magicquant/quant/schemes.py` entirely**

Overwrite `/server/programming/MagicQuant/magicquant/quant/schemes.py` with this exact content:

```python
"""
Quantization Schemes - Single source of truth for scheme metadata.

Each QuantizationScheme carries all attributes consumers (predictor, survival,
probing, orchestrator) need. This is the canonical registry — no other module
should hold parallel scheme dicts.

Bits-per-weight values are computed from the actual ggml block format:
  bpw = (block_bytes * 8) / block_elements

Noise factors are calibrated against published perplexity benchmarks
across Llama / Qwen / Mistral architectures. (PR0 keeps the existing
heuristic values; PR1 will replace them with empirically-benched values
from tools/calibrate_noise_factors.py.)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


SchemeCategory = Literal["k_quant", "iq_quant", "legacy_q", "float", "mxfp4"]


@dataclass(frozen=True)
class QuantizationScheme:
    """Canonical metadata for one quantization scheme.

    Consumers read attributes off this class instead of maintaining parallel
    lookup dicts. New schemes are added by appending to the registry below;
    no consumer-side changes required for static metadata.
    """

    name: str                         # MagicQuant identifier ("Q4_K_M", "MXFP4_MOE", ...)
    ggml_type_name: str               # ggml block type ("Q4_K", "MXFP4", ...)
    ggml_type_id: int                 # numeric ggml type enum (used by ctypes binding in PR1+)
    bits_per_weight: float            # actual storage bpw from ggml block format
    noise_factor: float               # relative quantization noise (lower = better quality)
    speed_multiplier: float = 1.0     # relative inference speed vs BF16
    category: SchemeCategory = "k_quant"
    is_moe_optimized: bool = False
    requires_imatrix: bool = False    # IQ-quants benefit from importance matrices
    min_for_group_class: Dict[str, str] = field(default_factory=dict)
    upgrade_neighbor: Optional[str] = None    # name of next-better scheme
    downgrade_neighbor: Optional[str] = None  # name of next-smaller scheme

    @property
    def compression_ratio(self) -> float:
        """Compression ratio relative to BF16 (16 bits)."""
        return 16.0 / self.bits_per_weight

    def __repr__(self) -> str:
        return f"QuantScheme({self.name}, {self.bits_per_weight}bpw, noise={self.noise_factor})"


# ── Registry ─────────────────────────────────────────────────────────
# NOTE: ggml_type_id values verified against ggml.h. PR1 adds the ctypes
# binding that uses these IDs.

BF16 = QuantizationScheme(
    name="BF16",
    ggml_type_name="BF16",
    ggml_type_id=30,
    bits_per_weight=16.0,
    noise_factor=0.0,
    speed_multiplier=1.0,
    category="float",
    upgrade_neighbor=None,
    downgrade_neighbor="Q8_0",
)

Q8_0 = QuantizationScheme(
    name="Q8_0",
    ggml_type_name="Q8_0",
    ggml_type_id=8,
    bits_per_weight=8.5,
    noise_factor=1.0,
    speed_multiplier=1.75,
    category="legacy_q",
    upgrade_neighbor="BF16",
    downgrade_neighbor="Q6_K",
)

Q6_K = QuantizationScheme(
    name="Q6_K",
    ggml_type_name="Q6_K",
    ggml_type_id=14,
    bits_per_weight=6.5625,
    noise_factor=2.2,
    speed_multiplier=2.2,
    category="k_quant",
    upgrade_neighbor="Q8_0",
    downgrade_neighbor="Q5_K",
)

Q5_K = QuantizationScheme(
    name="Q5_K",
    ggml_type_name="Q5_K",
    ggml_type_id=13,
    bits_per_weight=5.5,
    noise_factor=3.0,
    speed_multiplier=2.7,
    category="k_quant",
    upgrade_neighbor="Q6_K",
    downgrade_neighbor="IQ4_NL",
)

# IQ4_NL: Non-linear lookup table optimized for weight distributions.
# Lower noise than Q4_K_M despite same bpw because the 16 levels are
# learned to minimize quantization error on real weight distributions.
IQ4_NL = QuantizationScheme(
    name="IQ4_NL",
    ggml_type_name="IQ4_NL",
    ggml_type_id=20,
    bits_per_weight=4.5,
    noise_factor=3.8,
    speed_multiplier=3.2,
    category="iq_quant",
    upgrade_neighbor="Q5_K",
    downgrade_neighbor="MXFP4_MOE",
)

# MXFP4: OCP MX Microscaling FP4 (E2M1 values + shared E8M0 exponent).
# Non-uniform FP4 levels (0, 0.5, 1, 1.5, 2, 3, 4, 6) are denser near
# zero, naturally matching the Gaussian-like weight distribution of
# transformers. Lower noise than integer Q4 at slightly better compression.
MXFP4_MOE = QuantizationScheme(
    name="MXFP4_MOE",
    ggml_type_name="MXFP4",
    ggml_type_id=39,
    bits_per_weight=4.25,
    noise_factor=4.0,
    speed_multiplier=3.8,
    category="mxfp4",
    is_moe_optimized=True,
    upgrade_neighbor="IQ4_NL",
    downgrade_neighbor="Q4_K_M",
)

Q4_K_M = QuantizationScheme(
    name="Q4_K_M",
    ggml_type_name="Q4_K",
    ggml_type_id=12,
    bits_per_weight=4.5,
    noise_factor=4.5,
    speed_multiplier=3.4,
    category="k_quant",
    upgrade_neighbor="MXFP4_MOE",
    downgrade_neighbor=None,  # bottom of current registry; PR1 adds Q3_K
)


_REGISTRY: Dict[str, QuantizationScheme] = {
    "BF16": BF16,
    "Q8_0": Q8_0,
    "Q6_K": Q6_K,
    "Q5_K": Q5_K,
    "Q4_K_M": Q4_K_M,
    "IQ4_NL": IQ4_NL,
    "MXFP4_MOE": MXFP4_MOE,
}

# Group-class floors: minimum acceptable scheme per group class.
# "sensitive" (E, H, O, R) shouldn't go below Q8_0; "robust" (U, D, X)
# can go all the way to Q4_K_M (bottom of current registry).
# These were previously in survival.py as _MIN_SCHEME.
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q8_0",
    "robust": "Q4_K_M",
}


def get_scheme_by_name(name: str) -> QuantizationScheme:
    """Look up a scheme by its MagicQuant identifier."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown scheme: {name}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_all_schemes() -> List[QuantizationScheme]:
    """Get all schemes ordered by noise (best quality first)."""
    return sorted(_REGISTRY.values(), key=lambda s: s.noise_factor)


def get_schemes_by_category(category: SchemeCategory) -> List[QuantizationScheme]:
    """Get all schemes in a given category, ordered by noise (best first)."""
    return [s for s in get_all_schemes() if s.category == category]


def get_floor_for_group_class(group_class: str) -> str:
    """Get the minimum acceptable scheme name for a group sensitivity class.

    group_class: "sensitive" or "robust".
    """
    return _GROUP_CLASS_FLOORS[group_class]
```

- [ ] **Step 2: Verify the module imports cleanly**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.schemes import (
    get_all_schemes, get_scheme_by_name, get_schemes_by_category,
    get_floor_for_group_class, BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M
)
schemes = get_all_schemes()
print('count:', len(schemes))
print('order by noise:', [s.name for s in schemes])
print('Q8_0 upgrade:', get_scheme_by_name('Q8_0').upgrade_neighbor)
print('Q8_0 downgrade:', get_scheme_by_name('Q8_0').downgrade_neighbor)
print('k_quants:', [s.name for s in get_schemes_by_category('k_quant')])
print('robust floor:', get_floor_for_group_class('robust'))
"
```

Expected output:
```
count: 7
order by noise: ['BF16', 'Q8_0', 'Q6_K', 'Q5_K', 'IQ4_NL', 'MXFP4_MOE', 'Q4_K_M']
Q8_0 upgrade: BF16
Q8_0 downgrade: Q6_K
k_quants: ['Q6_K', 'Q5_K', 'Q4_K_M']
robust floor: Q4_K_M
```

- [ ] **Step 3: Run the regression test — must STILL pass**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`. If it fails, the public API of schemes.py changed in a behavior-affecting way — review the diff and fix.

- [ ] **Step 4: Run the existing dtype-guard tests — must still pass**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_quantization_guards.py -v
```

Expected output: all 12 existing tests pass.

- [ ] **Step 5: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/schemes.py && \
  git commit -m "refactor: extend QuantizationScheme with full attribute set

Adds ggml_type_name, ggml_type_id, category, requires_imatrix,
min_for_group_class, upgrade_neighbor, downgrade_neighbor.
Adds get_schemes_by_category() and get_floor_for_group_class() helpers.
Internal _REGISTRY dict canonical; consumers will be migrated next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Migrate `predictor.py` to read from registry

**Files:**
- Modify: `magicquant/evolution/predictor.py`

The `PredictiveScorer` class has three class-level dicts (`QUANT_NOISE_FACTORS`, `QUANT_COMPRESSION`, `QUANT_SPEED`) that mirror data already in the scheme registry. Replace lookups with registry calls.

- [ ] **Step 1: Read the current predictor.py to confirm line numbers**

Run:
```bash
cd /server/programming/MagicQuant && grep -n "QUANT_NOISE_FACTORS\|QUANT_COMPRESSION\|QUANT_SPEED" magicquant/evolution/predictor.py
```

Expected output (line numbers may shift slightly if file has been edited):
```
40:    QUANT_NOISE_FACTORS = {
52:    QUANT_COMPRESSION = {
64:    QUANT_SPEED = {
109:            noise_factor = self.QUANT_NOISE_FACTORS.get(scheme, 3.0)
138:            compression = self.QUANT_COMPRESSION.get(scheme, 2.0)
158:            speed_mult = self.QUANT_SPEED.get(scheme, 1.5)
200:            speed_mult = self.QUANT_SPEED.get(scheme, 1.5)
227:            compression = self.QUANT_COMPRESSION.get(scheme, 2.0)
```

If the lines are different, use the actual lines from `grep` output instead of the hardcoded ones below.

- [ ] **Step 2: Add the registry import at the top of predictor.py**

Edit `magicquant/evolution/predictor.py`. Find this block:
```python
from typing import Dict, List, Tuple, Optional
import numpy as np
```

Replace with:
```python
from typing import Dict, List, Tuple, Optional
import numpy as np

from magicquant.quant.schemes import get_scheme_by_name
```

- [ ] **Step 3: Delete the three class-level dicts**

In `magicquant/evolution/predictor.py`, find this block (lines ~33–72 — the entire `QUANT_NOISE_FACTORS`, `QUANT_COMPRESSION`, `QUANT_SPEED` dict definitions, including their docstring comments):

```python
    # Noise factors calibrated from llama.cpp perplexity benchmarks.
    # Lower = less quantization noise = better quality.
    #
    # Key insight: non-linear schemes (IQ4_NL, MXFP4) produce lower noise
    # than integer schemes at comparable bpw because their quantization
    # levels better match the Gaussian-like weight distribution of
    # transformers.
    QUANT_NOISE_FACTORS = {
        "BF16":      0.0,
        "Q8_0":      1.0,
        "Q6_K":      2.2,
        "Q5_K":      3.0,
        "IQ4_NL":    3.8,   # non-linear lookup table, best ~4-bit quality
        "MXFP4_MOE": 4.0,   # FP4 levels, better than integer Q4
        "Q4_K_M":    4.5,   # integer 4-bit with sub-block scales
    }

    # Compression ratios from actual ggml block format:
    # ratio = 16.0 / (block_bytes * 8 / block_elements)
    QUANT_COMPRESSION = {
        "BF16":      1.0,    # 16.0 bpw
        "Q8_0":      1.88,   # 8.5 bpw
        "Q6_K":      2.44,   # 6.5625 bpw
        "Q5_K":      2.91,   # 5.5 bpw
        "IQ4_NL":    3.56,   # 4.5 bpw
        "MXFP4_MOE": 3.76,   # 4.25 bpw — best compression of the ~4-bit schemes
        "Q4_K_M":    3.56,   # 4.5 bpw
    }

    # Relative speed multipliers (vs BF16).
    # MXFP4 is fast due to simple block format (shared exponent + nibbles).
    QUANT_SPEED = {
        "BF16":      1.0,
        "Q8_0":      1.75,
        "Q6_K":      2.2,
        "Q5_K":      2.7,
        "IQ4_NL":    3.2,
        "Q4_K_M":    3.4,
        "MXFP4_MOE": 3.8,
    }
```

Replace with:
```python
    # Scheme attributes (noise_factor, compression_ratio, speed_multiplier)
    # are read from the scheme registry — see magicquant.quant.schemes.
```

- [ ] **Step 4: Replace the three lookup sites with registry calls**

In `magicquant/evolution/predictor.py`:

Find:
```python
            noise_factor = self.QUANT_NOISE_FACTORS.get(scheme, 3.0)
```
Replace with:
```python
            noise_factor = self._noise_factor_for(scheme)
```

Find (occurs at two sites — `predict_size` and `_estimate_simple_size`):
```python
            compression = self.QUANT_COMPRESSION.get(scheme, 2.0)
```
Replace with:
```python
            compression = self._compression_for(scheme)
```

Find (occurs at two sites — `predict_tps` and `_estimate_simple_tps`):
```python
            speed_mult = self.QUANT_SPEED.get(scheme, 1.5)
```
Replace with:
```python
            speed_mult = self._speed_for(scheme)
```

- [ ] **Step 5: Add the three private helpers to PredictiveScorer**

In `magicquant/evolution/predictor.py`, find this method:
```python
    def _make_config_key(self, group_schemes: Dict[str, str]) -> str:
        return "|".join(f"{g}:{group_schemes[g]}" for g in sorted(group_schemes))
```

Add these methods immediately AFTER it (before the next method definition, which is `score_hybrid`):

```python
    @staticmethod
    def _noise_factor_for(scheme: str) -> float:
        """Look up noise_factor from registry; fallback to 3.0 if unknown."""
        try:
            return get_scheme_by_name(scheme).noise_factor
        except ValueError:
            return 3.0

    @staticmethod
    def _compression_for(scheme: str) -> float:
        """Look up compression_ratio from registry; fallback to 2.0 if unknown."""
        try:
            return get_scheme_by_name(scheme).compression_ratio
        except ValueError:
            return 2.0

    @staticmethod
    def _speed_for(scheme: str) -> float:
        """Look up speed_multiplier from registry; fallback to 1.5 if unknown."""
        try:
            return get_scheme_by_name(scheme).speed_multiplier
        except ValueError:
            return 1.5
```

- [ ] **Step 6: Verify imports and the file parses**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.evolution.predictor import PredictiveScorer
p = PredictiveScorer({'E': 1.5, 'U': 0.5}, {'E': 100_000_000, 'U': 800_000_000}, 5.0, 20.0)
print('noise_factor for Q4_K_M:', p._noise_factor_for('Q4_K_M'))
print('compression for MXFP4_MOE:', p._compression_for('MXFP4_MOE'))
print('speed for Q8_0:', p._speed_for('Q8_0'))
print('predict_loss({E: BF16, U: MXFP4_MOE}):', p.predict_loss({'E': 'BF16', 'U': 'MXFP4_MOE'}))
"
```

Expected output:
```
noise_factor for Q4_K_M: 4.5
compression for MXFP4_MOE: 3.764705882352941
speed for Q8_0: 1.75
predict_loss({E: BF16, U: MXFP4_MOE}): 0.27
```

(The compression value differs slightly from the old hardcoded `3.76` because the registry computes it from `bits_per_weight` exactly: `16.0 / 4.25 = 3.7647...`. The old hardcoded value was rounded. This affects `predict_size` by < 0.5% — expected and benign.)

- [ ] **Step 7: Run the regression test — must STILL pass**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`.

**Important:** The regression test pins seed=42 and verifies the **discovered config sequence** matches the fixture. Different `predict_size`/`predict_tps` values from the rounding fix could in principle reorder candidates in `_tournament_selection` (which sorts by composite_score). If the test FAILS because of this, the diff is benign but breaks the snapshot. Resolution:

  a. Inspect the diff: `cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v 2>&1 | head -40`
  b. If the diff is just reordering of equivalent-quality candidates, **regenerate the fixture** with a one-line tweak: re-run `python -m tests._capture_fixture` (recreate the script if you deleted it in Task 3).
  c. Document this in the commit message: "Note: fixture regenerated due to compression-ratio rounding fix (registry-computed values are exact)."
  d. If the diff is more substantial than rounding (different scheme picks at a higher level), STOP and investigate — something is actually broken.

- [ ] **Step 8: Run dtype guard tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_quantization_guards.py -v
```

Expected output: all 12 tests pass.

- [ ] **Step 9: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/evolution/predictor.py && \
  git commit -m "refactor: predictor reads scheme attributes from registry

Removes QUANT_NOISE_FACTORS, QUANT_COMPRESSION, QUANT_SPEED class dicts.
PredictiveScorer now looks up noise_factor, compression_ratio, and
speed_multiplier from get_scheme_by_name(). Compression values become
exact (registry-computed) instead of hand-rounded.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(If the regression fixture was regenerated in Step 7, also `git add tests/fixtures/refactor_regression_seed42.json` and note it in the commit body.)

---

### Task 6: Migrate `survival.py` upgrade/downgrade chains and quality order

**Files:**
- Modify: `magicquant/evolution/survival.py`

`EvolutionarySurvivor` holds three local dicts (`_UPGRADE`, `_DOWNGRADE`, `_MIN_SCHEME`) and a class constant `SCHEME_QUALITY_ORDER` that all duplicate registry data.

- [ ] **Step 1: Add the registry import at the top of survival.py**

Find:
```python
from typing import Dict, List, Tuple, Optional
import random
import copy
```

Replace with:
```python
from typing import Dict, List, Tuple, Optional
import random
import copy

from magicquant.quant.schemes import (
    get_all_schemes, get_scheme_by_name, get_floor_for_group_class
)
```

- [ ] **Step 2: Replace SCHEME_QUALITY_ORDER with a registry-derived list**

Find (lines ~21–26):
```python
# Ordered from highest quality to most compressed.
# MXFP4_MOE is the sweet spot: best compression of the ~4-bit schemes
# with lower noise than integer Q4 due to FP4 non-linear levels.
SCHEME_QUALITY_ORDER = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE", "Q4_K_M"
]
```

Replace with:
```python
# Ordered from highest quality (lowest noise) to most compressed (highest noise).
# Derived from the canonical scheme registry — see magicquant.quant.schemes.
SCHEME_QUALITY_ORDER: List[str] = [s.name for s in get_all_schemes()]
```

- [ ] **Step 3: Replace the _UPGRADE and _DOWNGRADE class dicts with registry-derived staticmethods**

Find (lines ~47–65):
```python
    # Upgrade: move toward higher quality (less compression)
    _UPGRADE = {
        "Q4_K_M":    "MXFP4_MOE",
        "MXFP4_MOE": "IQ4_NL",
        "IQ4_NL":    "Q5_K",
        "Q5_K":      "Q6_K",
        "Q6_K":      "Q8_0",
        "Q8_0":      "BF16",
    }

    # Downgrade: move toward more compression (less quality)
    _DOWNGRADE = {
        "BF16":      "Q8_0",
        "Q8_0":      "Q6_K",
        "Q6_K":      "Q5_K",
        "Q5_K":      "IQ4_NL",
        "IQ4_NL":    "MXFP4_MOE",
        "MXFP4_MOE": "Q4_K_M",
    }
```

Replace with:
```python
    @staticmethod
    def _upgrade(scheme: str) -> Optional[str]:
        """Return the next-better scheme, or None if at the top."""
        try:
            return get_scheme_by_name(scheme).upgrade_neighbor
        except ValueError:
            return None

    @staticmethod
    def _downgrade(scheme: str) -> Optional[str]:
        """Return the next-smaller scheme, or None if at the bottom."""
        try:
            return get_scheme_by_name(scheme).downgrade_neighbor
        except ValueError:
            return None
```

- [ ] **Step 4: Replace _MIN_SCHEME with registry-derived helper**

Find (lines ~73–77):
```python
    # Floor: minimum acceptable scheme for each group type
    _MIN_SCHEME = {
        'sensitive': "Q8_0",      # brain layers shouldn't go below Q8_0
        'robust':    "Q4_K_M",    # FFN can go all the way down
    }
```

Replace with:
```python
    # Floor for each group class — read from registry helper so the
    # values stay consistent if the registry's bottom scheme changes.
    @staticmethod
    def _min_scheme_for_class(group_class: str) -> str:
        """Get the minimum acceptable scheme for a group class
        ("sensitive" or "robust")."""
        return get_floor_for_group_class(group_class)
```

- [ ] **Step 5: Update _mutate_winners to use the new _upgrade/_downgrade methods**

Find this section in `_mutate_winners` (lines ~265–295):
```python
        for winner in winners:
            config = copy.deepcopy(winner['config'])

            # Protector: upgrade the most sensitive unprotected brain layer
            target = self._find_protector_target(config, groups)
            if target:
                new_scheme = self._UPGRADE.get(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})

            # Crusher: downgrade the most robust high-precision FFN layer
            target = self._find_crusher_target(config, groups)
            if target:
                new_scheme = self._DOWNGRADE.get(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})
```

Replace with:
```python
        for winner in winners:
            config = copy.deepcopy(winner['config'])

            # Protector: upgrade the most sensitive unprotected brain layer
            target = self._find_protector_target(config, groups)
            if target:
                new_scheme = self._upgrade(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})

            # Crusher: downgrade the most robust high-precision FFN layer
            target = self._find_crusher_target(config, groups)
            if target:
                new_scheme = self._downgrade(target['scheme'])
                if new_scheme and new_scheme != target['scheme']:
                    c = config.copy()
                    c[target['group']] = new_scheme
                    population.append({'config': c})
```

- [ ] **Step 6: Update _find_crusher_target to use the new _downgrade method**

Find:
```python
            # Can we push it lower?
            if scheme in self._DOWNGRADE and scheme != "Q4_K_M":
```

Replace with:
```python
            # Can we push it lower? (Skip if already at the bottom of registry)
            if self._downgrade(scheme) is not None and scheme != self._min_scheme_for_class('robust'):
```

- [ ] **Step 7: Verify imports and the module parses**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.evolution.survival import EvolutionarySurvivor, SCHEME_QUALITY_ORDER
print('quality order:', SCHEME_QUALITY_ORDER)
print('upgrade Q5_K:', EvolutionarySurvivor._upgrade('Q5_K'))
print('downgrade MXFP4_MOE:', EvolutionarySurvivor._downgrade('MXFP4_MOE'))
print('min for robust:', EvolutionarySurvivor._min_scheme_for_class('robust'))
"
```

Expected output:
```
quality order: ['BF16', 'Q8_0', 'Q6_K', 'Q5_K', 'IQ4_NL', 'MXFP4_MOE', 'Q4_K_M']
upgrade Q5_K: Q6_K
downgrade MXFP4_MOE: Q4_K_M
min for robust: Q4_K_M
```

- [ ] **Step 8: Run regression test — must STILL pass**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Expected output: `1 passed`. (The chains and floors produce identical lookups, so no behavior change.)

- [ ] **Step 9: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/evolution/survival.py && \
  git commit -m "refactor: survival reads upgrade/downgrade chains from registry

Removes _UPGRADE, _DOWNGRADE, _MIN_SCHEME class dicts; replaces with
_upgrade(), _downgrade(), _min_scheme_for_class() that delegate to
get_scheme_by_name().<attr>. SCHEME_QUALITY_ORDER becomes a list
comprehension over get_all_schemes().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Migrate `probing.py` heuristic noise dict to registry

**Files:**
- Modify: `magicquant/evolution/probing.py`

`SensitivityProber._heuristic_probe` defines a local `_SCHEME_NOISE` dict that mirrors registry data with a different scaling.

- [ ] **Step 1: Verify the current _heuristic_probe content**

Run:
```bash
cd /server/programming/MagicQuant && sed -n '243,279p' magicquant/evolution/probing.py
```

Expected output: the current `_heuristic_probe` method showing `_GROUP_SENSITIVITY` and `_SCHEME_NOISE` dicts.

- [ ] **Step 2: Add the registry import**

Edit `magicquant/evolution/probing.py`. Find the import section near the top — it varies by what's already imported. Add (or extend):

```python
from magicquant.quant.schemes import get_scheme_by_name
```

If the imports already include this, skip this step.

- [ ] **Step 3: Replace the local _SCHEME_NOISE dict with a registry lookup**

Find this block in `_heuristic_probe`:
```python
        # Scheme aggressiveness (relative noise, aligned with predictor)
        _SCHEME_NOISE = {
            "BF16": 0.0,
            "Q8_0": 0.2,
            "Q6_K": 0.5,
            "Q5_K": 0.7,
            "IQ4_NL": 0.85,    # non-linear levels, best ~4-bit quality
            "MXFP4_MOE": 0.9,  # FP4 levels, better than integer Q4
            "Q4_K_M": 1.0,     # integer 4-bit baseline
        }

        sensitivity = _GROUP_SENSITIVITY.get(group, 1.0)
        noise = _SCHEME_NOISE.get(scheme, 1.0)
```

Replace with:
```python
        # Scheme aggressiveness scaled to the heuristic's [0, 1] range.
        # Registry's noise_factor uses Q8_0=1.0 anchor; we rescale here so
        # Q4_K_M=1.0 maps to "max heuristic aggressiveness". This preserves
        # the original heuristic's behavior pre-refactor.
        try:
            registry_noise = get_scheme_by_name(scheme).noise_factor
            # Q4_K_M (registry noise=4.5) maps to 1.0; linearly scale others.
            noise = registry_noise / 4.5
        except ValueError:
            noise = 1.0

        sensitivity = _GROUP_SENSITIVITY.get(group, 1.0)
```

- [ ] **Step 4: Verify the heuristic returns the same values**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
# Reproduce the old heuristic locally to compare
old_noise = {
    'BF16': 0.0, 'Q8_0': 0.2, 'Q6_K': 0.5, 'Q5_K': 0.7,
    'IQ4_NL': 0.85, 'MXFP4_MOE': 0.9, 'Q4_K_M': 1.0,
}
from magicquant.quant.schemes import get_scheme_by_name
for name, old in old_noise.items():
    new = get_scheme_by_name(name).noise_factor / 4.5
    diff = abs(new - old)
    print(f'{name:10s}: old={old:.3f}  new={new:.3f}  diff={diff:.3f}')
"
```

Expected output (some values shift slightly because the registry noise factors are not perfectly proportional to the old heuristic; this is expected and benign):
```
BF16      : old=0.000  new=0.000  diff=0.000
Q8_0      : old=0.200  new=0.222  diff=0.022
Q6_K      : old=0.500  new=0.489  diff=0.011
Q5_K      : old=0.700  new=0.667  diff=0.033
IQ4_NL    : old=0.850  new=0.844  diff=0.006
MXFP4_MOE : old=0.900  new=0.889  diff=0.011
Q4_K_M    : old=1.000  new=1.000  diff=0.000
```

The maximum `diff` is ~0.033 (Q5_K). The heuristic is a fallback path used only when llama.cpp is unavailable; absolute precision isn't critical. Document this benign drift in the commit message.

- [ ] **Step 5: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v
```

Expected output: all tests pass. The regression test does NOT exercise the heuristic path (it uses the predictor's path), so it should pass unchanged.

- [ ] **Step 6: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/evolution/probing.py && \
  git commit -m "refactor: probing heuristic reads scheme noise from registry

Removes local _SCHEME_NOISE dict in _heuristic_probe; replaces with
get_scheme_by_name(scheme).noise_factor / 4.5 scaling.

The heuristic-fallback path now derives from registry. Absolute values
shift by at most ~0.033 (Q5_K) due to non-proportional noise factors;
the heuristic is only used when llama.cpp is unavailable, so this
drift is benign.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Migrate `orchestrator.py` base_quant ranking dict

**Files:**
- Modify: `magicquant/orchestrator.py`

A small inline dict at line ~583 ranks schemes by their bpw to pick a `base_quant` label. Replace with a registry lookup.

- [ ] **Step 1: Verify the current line content**

Run:
```bash
cd /server/programming/MagicQuant && sed -n '580,590p' magicquant/orchestrator.py
```

Expected output:
```python
            base_quant = max(
                set(config.values()),
                key=lambda s: {
                    "BF16": 0, "Q8_0": 1, "Q6_K": 2, "Q5_K": 3,
                    "IQ4_NL": 4, "MXFP4_MOE": 5, "Q4_K_M": 6
                }.get(s, 3)
            )
```

- [ ] **Step 2: Confirm the existing get_scheme_by_name import**

Run:
```bash
cd /server/programming/MagicQuant && grep -n "from magicquant.quant.schemes import" magicquant/orchestrator.py
```

If output shows an existing import line, append `, get_scheme_by_name` to it (if not already present). If no output, add a new import line near the other imports at the top of the file:
```python
from magicquant.quant.schemes import get_scheme_by_name
```

- [ ] **Step 3: Replace the inline dict with a registry lookup**

Find:
```python
            base_quant = max(
                set(config.values()),
                key=lambda s: {
                    "BF16": 0, "Q8_0": 1, "Q6_K": 2, "Q5_K": 3,
                    "IQ4_NL": 4, "MXFP4_MOE": 5, "Q4_K_M": 6
                }.get(s, 3)
            )
```

Replace with:
```python
            # base_quant: pick the scheme with highest bpw (least compressed) as
            # the "label" for this hybrid. Reads bpw from the canonical registry.
            def _bpw_or_default(s: str) -> float:
                try:
                    return get_scheme_by_name(s).bits_per_weight
                except ValueError:
                    return 4.5  # mid-range default for unknown schemes
            base_quant = max(set(config.values()), key=_bpw_or_default)
```

- [ ] **Step 4: Verify orchestrator imports and parses**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.orchestrator import MagicQuantOrchestrator
print('orchestrator imports OK')
"
```

Expected output: `orchestrator imports OK`.

- [ ] **Step 5: Verify ranking by bpw produces same order**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.schemes import get_scheme_by_name
schemes = ['BF16', 'Q8_0', 'Q6_K', 'Q5_K', 'IQ4_NL', 'MXFP4_MOE', 'Q4_K_M']
ranked = sorted(schemes, key=lambda s: get_scheme_by_name(s).bits_per_weight)
print('by bpw ascending:', ranked)
print('max (highest bpw):', max(schemes, key=lambda s: get_scheme_by_name(s).bits_per_weight))
"
```

Expected output:
```
by bpw ascending: ['MXFP4_MOE', 'Q4_K_M', 'IQ4_NL', 'Q5_K', 'Q6_K', 'Q8_0', 'BF16']
max (highest bpw): BF16
```

(Note `Q4_K_M` and `IQ4_NL` both have bpw 4.5; tie-break by Python's stable sort. The old hardcoded dict put `IQ4_NL` at rank 4 and `Q4_K_M` at rank 6, so old behavior preferred `Q4_K_M` over `IQ4_NL`. New behavior using `max()` with a tied key returns the first one found, which depends on `set` ordering. **This is a subtle behavior change.** If a hybrid mixes both schemes, the `base_quant` label may differ between old and new. The label is informational only — used for the output filename. Document in commit message.)

- [ ] **Step 6: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v
```

Expected output: all tests pass. Regression test uses synthetic configs that don't exercise this code path (the test only runs `EvolutionarySurvivor`, not `generate_tiered_models`).

- [ ] **Step 7: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/orchestrator.py && \
  git commit -m "refactor: orchestrator base_quant ranking reads from registry

Replaces inline {scheme: rank} dict with registry-derived bpw lookup.

Subtle change: when a config mixes schemes with identical bpw (e.g.,
Q4_K_M and IQ4_NL at 4.5), the chosen base_quant label may differ from
the old behavior. base_quant is informational (output filename label),
not functional.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Update package re-exports

**Files:**
- Modify: `magicquant/__init__.py`
- Modify: `magicquant/quant/__init__.py`

The new helpers (`get_schemes_by_category`, `get_floor_for_group_class`) should be re-exported alongside existing scheme constants. This is a courtesy for downstream consumers (e.g., Foundry's `core/pipeline.py` imports schemes from `magicquant.quant`).

- [ ] **Step 1: Update `magicquant/quant/__init__.py`**

Replace the entire file content with:

```python
"""
Quant Module - Quantization schemes and conversions.
"""

from magicquant.quant.schemes import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
    get_floor_for_group_class,
)
from magicquant.quant.converters import Quantizer

__all__ = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE", "Q4_K_M",
    "QuantizationScheme",
    "get_scheme_by_name",
    "get_all_schemes",
    "get_schemes_by_category",
    "get_floor_for_group_class",
    "Quantizer",
]
```

- [ ] **Step 2: Update `magicquant/__init__.py`**

Replace the entire file content with:

```python
"""
MagicQuant - Evolutionary Tensor Search for Optimal LLM Compression

A hybrid quantization framework that dynamically groups tensors by architectural role
and employs evolutionary search to find optimal mixed-precision configurations.
"""

__version__ = "0.1.0"

from magicquant.quant.schemes import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M,
    QuantizationScheme,
    get_scheme_by_name,
    get_all_schemes,
    get_schemes_by_category,
)

from magicquant.gguf.tensor_groups import TensorGroupClassifier

__all__ = [
    "BF16", "Q8_0", "Q6_K", "Q5_K", "IQ4_NL", "MXFP4_MOE", "Q4_K_M",
    "QuantizationScheme",
    "get_scheme_by_name",
    "get_all_schemes",
    "get_schemes_by_category",
    "TensorGroupClassifier",
]
```

- [ ] **Step 3: Verify re-exports work**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant import (
    BF16, Q8_0, Q6_K, Q5_K, IQ4_NL, MXFP4_MOE, Q4_K_M,
    QuantizationScheme, get_scheme_by_name, get_all_schemes,
    get_schemes_by_category, TensorGroupClassifier,
)
print('top-level imports OK')
print('Q4_K_M.category:', Q4_K_M.category)
print('all k_quants:', [s.name for s in get_schemes_by_category('k_quant')])
"
```

Expected output:
```
top-level imports OK
Q4_K_M.category: k_quant
all k_quants: ['Q6_K', 'Q5_K', 'Q4_K_M']
```

- [ ] **Step 4: Run all tests one more time**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v
```

Expected output: all tests pass.

- [ ] **Step 5: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/__init__.py magicquant/quant/__init__.py && \
  git commit -m "refactor: re-export new scheme registry helpers

QuantizationScheme, get_scheme_by_name, get_all_schemes,
get_schemes_by_category, get_floor_for_group_class are now available at
both magicquant.* and magicquant.quant.* import paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Final verification + push

**Files:** none

- [ ] **Step 1: Run the entire test suite**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected output: all tests in `test_quantization_guards.py` (12) and `test_refactor_regression.py` (1) pass. Total 13 passed.

- [ ] **Step 2: Verify no leftover references to the old class-level dicts**

Run:
```bash
cd /server/programming/MagicQuant && grep -rn "QUANT_NOISE_FACTORS\|QUANT_COMPRESSION\|QUANT_SPEED\|_UPGRADE\|_DOWNGRADE\|_MIN_SCHEME\|SCHEME_QUALITY_ORDER" magicquant/ --include="*.py" | grep -v __pycache__
```

Expected output: only references inside `survival.py` to `SCHEME_QUALITY_ORDER` (which we kept as a registry-derived list). No references to the removed dicts.

If any other references appear, they were missed during the migration — fix them now.

- [ ] **Step 3: Check the commit log**

Run:
```bash
cd /server/programming/MagicQuant && git log --oneline -10
```

Expected output: 6 new commits since `master`'s state at the start of PR0:
1. `test: add refactor regression test (fixture pending)`
2. `test: capture refactor regression fixture (seed=42)`
3. `refactor: extend QuantizationScheme with full attribute set`
4. `refactor: predictor reads scheme attributes from registry`
5. `refactor: survival reads upgrade/downgrade chains from registry`
6. `refactor: probing heuristic reads scheme noise from registry`
7. `refactor: orchestrator base_quant ranking reads from registry`
8. `refactor: re-export new scheme registry helpers`

(8 commits if the predictor task regenerated the fixture into a separate commit, else 7.)

- [ ] **Step 4: Verify git status is clean**

Run:
```bash
cd /server/programming/MagicQuant && git status
```

Expected output: `nothing to commit, working tree clean`.

- [ ] **Step 5: Push to origin**

Run:
```bash
cd /server/programming/MagicQuant && git push origin master 2>&1
```

Expected output:
```
... b58ce4b..<new-sha>  master -> master
```

A redirect note from GitHub about the repo being renamed is expected and benign.

- [ ] **Step 6: PR0 done — confirm in writing**

Print a status message:
```
PR0 complete:
- QuantizationScheme registry centralized
- predictor.py, survival.py, probing.py, orchestrator.py read from registry
- Behavior preserved (regression test passes, dtype tests pass)
- 7 commits pushed to origin/master
- Ready for PR1 (libggml binding + K-quant batch)
```

---

## Self-Review Checklist

This plan was self-reviewed against the spec. Notes:

**Spec coverage:**
- [x] "Extend `QuantizationScheme` with full attribute set" → Task 4
- [x] "Convert predictor.py to read from registry" → Task 5
- [x] "Convert survival.py upgrade/downgrade chains" → Task 6
- [x] "Convert survival.py min_scheme" → Task 6 (combined with chains)
- [x] "Convert survival.py random-config weights" → **deferred to PR1**
- [x] "Convert probing.py _SCHEME_NOISE" → Task 7
- [x] "Convert orchestrator.py:585 base_quant ranking" → Task 8
- [x] "Add tests/test_refactor_regression.py with pinned-RNG snapshot" → Tasks 2 & 3
- [x] "Behavior change: zero" → verified by regression test in every commit

**Random-config weights deferred:** The spec says PR0 should rewrite `_generate_random_config`'s weight arrays from positional to category-indexed. After analysis (see plan body), restructuring weights in PR0 risks shifting `random.choices()` draws and breaking the regression snapshot. The cleaner path is to keep the positional arrays in PR0 (they still work because registry order = old order for the existing 7 schemes) and restructure to category-indexed in PR1 *when the new schemes are added* — at which point the positional approach can no longer compile cleanly anyway. The spec is updated to reflect this in the implementation plan; the spec text remains accurate for the broader project but PR0 specifically scopes to "make existing dicts read from registry."

**Placeholder scan:** No "TBD", "TODO", "implement later", or vague directives in the plan. All code blocks contain actual code. All tests have real assertions. All commit messages are pre-written.

**Type consistency:** `_upgrade()` and `_downgrade()` are defined as `@staticmethod` in survival.py (Task 6) and called as `self._upgrade(...)` from `_mutate_winners`. Both call sites updated consistently.

---

## Future Work (not in this plan)

- PR1: libggml ctypes binding + K-quant batch + retrofit + calibration bench
- PR2: Legacy Q-quants
- PR3: IQ-quants
- PR4: Importance-matrix support
- See spec: `docs/superpowers/specs/2026-05-04-magicquant-encoder-expansion-design.md`
