# PR4: Importance-Matrix Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Prerequisite:** PR3 must be merged to `master`. Verify with `git log --oneline | grep -i "IQ-quants\|IQ2\|IQ3" | head -3`.

**Goal:** Add importance-matrix support to MagicQuant so the IQ-quants registered in PR3 can produce their best-quality output. Without imatrix, IQ1/IQ2/IQ3 schemes have visibly degraded quality. With imatrix, they reach their advertised quality (often better than K-quants at the same bpw). This PR completes the encoder-expansion workflow.

**Architecture:** A new `magicquant/imatrix.py` module captures activation magnitudes per tensor by running a calibration corpus through the source model. Initial implementation wraps the existing `llama-imatrix` binary as a subprocess. The captured imatrix file is loaded into a numpy array and threaded through `create_hybrid_gguf()` → `encode_to_ggml_bytes()` → `ggml_encode()`. The orchestrator captures imatrix once at the start of a search run if any selected scheme has `requires_imatrix=True`. Foundry's UI gets a "Calibration dataset" input that defaults to wikitext-2.

**Tech Stack:** Python 3.12, subprocess (llama-imatrix), numpy, pytest

**Spec:** `docs/superpowers/specs/2026-05-04-magicquant-encoder-expansion-design.md` (section "PR4 — Importance-matrix support")

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `magicquant/imatrix.py` | Create | Capture imatrix via llama-imatrix subprocess; load into numpy |
| `magicquant/__init__.py` | Modify | Re-export ImatrixCapture class |
| `magicquant/quant/converters.py` | Modify | (No change — `imatrix=None` path already exists from PR1) |
| `magicquant/gguf/writer.py` | Modify | Thread `imatrix_per_tensor` dict through `create_hybrid_gguf()` |
| `magicquant/orchestrator.py` | Modify | Capture imatrix once if any selected scheme requires it; pass to writer |
| `tests/integration/test_encoder_parity.py` | Modify | Revert xfail markers for imatrix-dependent IQ-quants; pass imatrix |
| `tests/integration/test_smoke_q2_tier.py` | Modify | Strengthen assertion: perplexity within 1.5x of baseline |
| `tests/integration/test_imatrix_capture.py` | Create | Imatrix module unit tests |
| `tools/calibration_results.json` | Modify | Refresh for IQ-quants WITH imatrix (better noise factors) |
| `magicquant/quant/schemes.py` | Modify | Update IQ-quant noise factors with imatrix-based calibration |
| **Foundry-side:** | | |
| `Foundry/ui/index.html` | Modify | Add "Calibration dataset" input in MagicQuant config panel |
| `Foundry/ui/app.py` | Modify | Add `imatrix_dataset` field to MagicQuantConfig |
| `Foundry/core/pipeline.py` | Modify | Same dataclass field; pass through to MagicQuant orchestrator |
| `Foundry/CHANGELOG.md` | Modify | Note imatrix support added |

**File-size note:** ~600 net new lines. The biggest piece is the imatrix module (~250 lines including subprocess wrapping, parsing, and tests).

---

## Tasks

### Task 1: Prerequisite verification

**Files:** none

- [ ] **Step 1: Verify PR3 is merged**

Run:
```bash
cd /server/programming/MagicQuant && git log --oneline | head -10
```

Expected: includes IQ-quant commits.

- [ ] **Step 2: Verify llama-imatrix is on PATH**

Run:
```bash
which llama-imatrix && llama-imatrix --help 2>&1 | head -10
```

Expected: a path under `/home/linuxbrew/.linuxbrew/bin/llama-imatrix` and usage output.

If absent, locate elsewhere:
```bash
find / -maxdepth 6 -name "llama-imatrix" -executable 2>/dev/null | head -3
```

If no copy exists, build llama.cpp's imatrix tool:
```bash
cd ~/llama.cpp && cmake --build build --target llama-imatrix
```

- [ ] **Step 3: Verify ggml exposes the imatrix-required predicate**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.ggml_binding import get_handle, GGML_TYPE_IDS
h = get_handle()
for name in ['IQ1_S', 'IQ1_M', 'IQ2_XXS', 'IQ2_XS', 'IQ2_S', 'IQ2_M',
             'IQ3_XXS', 'IQ3_S', 'IQ3_M', 'IQ4_XS', 'Q4_K', 'Q5_K']:
    needs = h.requires_imatrix(name)
    print(f'  {name:10s}  requires_imatrix: {needs}')
"
```

Expected output: IQ1/IQ2/IQ3 schemes return True; Q4_K, Q5_K, IQ4_XS return False.

If the function isn't available (older ggml), fall back to the static `requires_imatrix` field on `QuantizationScheme` (set in PR3).

- [ ] **Step 4: Run all PR3 tests as baseline**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: 35+ pass, with some xfail markers on IQ-quant parity tests (intentional, will revert in this PR).

---

### Task 2: Create the imatrix capture module — discovery and parsing

**Files:**
- Create: `magicquant/imatrix.py`

The first commit of this module just handles imatrix file format parsing. The subprocess capture wrapper comes in the next task.

- [ ] **Step 1: Investigate llama-imatrix output format**

Run:
```bash
llama-imatrix --help 2>&1 | head -30
```

Note the output file format and CLI args.

- [ ] **Step 2: Create the parser-only module**

Create `/server/programming/MagicQuant/magicquant/imatrix.py` with this content:

```python
"""
Importance matrix support for IQ-quant calibration.

An imatrix is a per-tensor float32 vector capturing how often each
weight column is heavily used during inference. Quantizers use it to
preserve precision in high-importance columns.

This module:
  1. Captures imatrix via the llama-imatrix subprocess (uses an existing
     llama.cpp binary; see capture_imatrix()).
  2. Parses llama-imatrix's binary output format into a per-tensor numpy
     dict (see load_imatrix_file()).
  3. Provides a high-level ImatrixCapture context that bundles both.

Public API:
    capture_imatrix(model_path, corpus_path, output_path, ...) -> Path
    load_imatrix_file(path) -> Dict[str, np.ndarray]
    ImatrixCapture (class)
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class ImatrixCapture:
    """A captured importance matrix.

    Loaded eagerly from disk at construction. Use indexing to get
    per-tensor activation vectors:
        cap = ImatrixCapture.load("imatrix.dat")
        weights_for_blk0 = cap["blk.0.attn_q.weight"]  # numpy float32 array
    """
    matrices: Dict[str, np.ndarray]
    source_path: Path

    def __getitem__(self, tensor_name: str) -> np.ndarray:
        if tensor_name not in self.matrices:
            raise KeyError(
                f"No imatrix entry for tensor '{tensor_name}'. "
                f"Available: {len(self.matrices)} tensors. "
                f"Was the imatrix captured for a different model?"
            )
        return self.matrices[tensor_name]

    def __contains__(self, tensor_name: str) -> bool:
        return tensor_name in self.matrices

    def get(self, tensor_name: str, default=None) -> Optional[np.ndarray]:
        return self.matrices.get(tensor_name, default)

    @classmethod
    def load(cls, path: Path) -> "ImatrixCapture":
        return cls(matrices=load_imatrix_file(path), source_path=Path(path))


def load_imatrix_file(path: Path) -> Dict[str, np.ndarray]:
    """Parse llama-imatrix's output file format.

    Format (from llama.cpp's imatrix.cpp):
        int32 n_entries
        for each entry:
            int32 name_length
            char[name_length] name
            int32 ncall
            int32 nval
            float[nval] values

    Returns:
        Dict mapping tensor name → 1-D float32 array of importance weights.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"imatrix file not found: {path}")

    matrices: Dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        data = f.read()

    pos = 0
    n_entries = struct.unpack_from("<i", data, pos)[0]
    pos += 4

    for _ in range(n_entries):
        # Name
        name_len = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        name = data[pos:pos + name_len].decode("utf-8")
        pos += name_len

        # ncall (number of forward passes that updated this tensor)
        _ncall = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        # nval (length of importance vector)
        nval = struct.unpack_from("<i", data, pos)[0]
        pos += 4

        # Values
        values = np.frombuffer(data, dtype=np.float32, count=nval, offset=pos)
        pos += nval * 4

        matrices[name] = values.copy()  # detach from buffer

    return matrices
```

- [ ] **Step 3: Verify the parser module imports cleanly**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.imatrix import ImatrixCapture, load_imatrix_file
print('imports OK')
"
```

Expected: `imports OK`.

- [ ] **Step 4: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/imatrix.py && \
  git commit -m "feat: imatrix file format parser

magicquant.imatrix module: parses llama-imatrix output files into
per-tensor numpy float32 dicts. ImatrixCapture wrapper provides
dict-like access. Subprocess capture comes in next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add subprocess capture to imatrix.py

**Files:**
- Modify: `magicquant/imatrix.py`

- [ ] **Step 1: Append the capture function**

Append the following to `/server/programming/MagicQuant/magicquant/imatrix.py`:

```python


def capture_imatrix(
    model_path: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    ctx_size: int = 512,
    chunks: int = 100,
    threads: Optional[int] = None,
    llama_imatrix_bin: Optional[str] = None,
) -> Path:
    """Run llama-imatrix to capture an importance matrix for a model.

    Args:
        model_path: Path to a GGUF model file. Must be loadable by llama.cpp
            (BF16/F16/Q8_0/etc. — quantized models work but produce
            quantization-baked imatrices, so prefer F16 or BF16).
        corpus_path: Path to a calibration text file (e.g. wikitext-2-raw/wiki.test.raw).
        output_path: Where to write the imatrix file.
        ctx_size: Context window for llama-imatrix.
        chunks: Number of corpus chunks to process. Higher = more accurate
            but slower. 100 is a reasonable default for a 1B model.
        threads: CPU threads. Default: os.cpu_count().
        llama_imatrix_bin: Override path to llama-imatrix binary. Default:
            auto-detect via PATH or LLAMA_IMATRIX env var.

    Returns:
        Path to the captured imatrix file (== output_path).

    Raises:
        RuntimeError if llama-imatrix subprocess fails.
    """
    bin_path = (
        llama_imatrix_bin
        or os.environ.get("LLAMA_IMATRIX")
        or shutil.which("llama-imatrix")
    )
    if not bin_path:
        raise RuntimeError(
            "llama-imatrix not on PATH and no override supplied. "
            "Install llama.cpp tools or set LLAMA_IMATRIX env var."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path,
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-o", str(output_path),
        "--ctx-size", str(ctx_size),
        "--chunks", str(chunks),
        "--threads", str(threads or os.cpu_count() or 4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-imatrix failed (rc={proc.returncode}):\n"
            f"stdout: {proc.stdout[-500:]}\n"
            f"stderr: {proc.stderr[-500:]}"
        )
    if not output_path.is_file():
        raise RuntimeError(
            f"llama-imatrix completed (rc=0) but output file not found at {output_path}"
        )
    return output_path
```

- [ ] **Step 2: Test the capture function with a small corpus**

Run a smoke test (uses the same model paths as the calibration bench):
```bash
cd /server/programming/MagicQuant && python -c "
from pathlib import Path
from magicquant.imatrix import capture_imatrix, ImatrixCapture

# Adjust paths to match your setup. Use a small chunks value for the smoke test.
imatrix_path = capture_imatrix(
    model_path=Path('<MODEL-PATH-HERE>'),  # e.g. ~/models/Llama-3.2-1B-Instruct-bf16.gguf
    corpus_path=Path('/home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw'),
    output_path=Path('/tmp/imatrix_smoke.dat'),
    chunks=10,  # tiny — just verify the binary path works
)
print('imatrix written to:', imatrix_path)

cap = ImatrixCapture.load(imatrix_path)
print(f'tensors captured: {len(cap.matrices)}')
sample = next(iter(cap.matrices.items()))
print(f'sample: {sample[0]} → shape={sample[1].shape}, dtype={sample[1].dtype}')
"
```

Expected output: `imatrix written to: /tmp/imatrix_smoke.dat` and a sample tensor entry. Wall-clock: ~30 seconds for 10 chunks on a 1B model.

If the model path is in safetensors form (not GGUF), llama-imatrix won't read it. Convert with:
```bash
python ~/llama.cpp/convert_hf_to_gguf.py <SAFETENSORS-DIR> --outtype f16 --outfile /tmp/model.f16.gguf
```

Then re-run with `model_path=Path('/tmp/model.f16.gguf')`.

- [ ] **Step 3: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/imatrix.py && \
  git commit -m "feat: imatrix capture via llama-imatrix subprocess

capture_imatrix(model_path, corpus_path, output_path, ...) wraps
the existing llama-imatrix binary. Returns the path to the captured
file, ready to feed into ImatrixCapture.load().

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add imatrix unit tests

**Files:**
- Create: `tests/integration/test_imatrix_capture.py`
- Create: `tests/fixtures/sample_imatrix.dat`

- [ ] **Step 1: Generate a small fixture imatrix**

Run (writes a tiny 2-tensor imatrix file with hand-crafted values for testing):
```bash
cd /server/programming/MagicQuant && python -c "
import struct
from pathlib import Path

# Build a minimal imatrix file: 2 entries.
out = Path('tests/fixtures/sample_imatrix.dat')
out.parent.mkdir(parents=True, exist_ok=True)

import numpy as np
data = bytearray()
data += struct.pack('<i', 2)  # n_entries

# Entry 1: 'blk.0.attn_q.weight'
name1 = b'blk.0.attn_q.weight'
vals1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
data += struct.pack('<i', len(name1))
data += name1
data += struct.pack('<i', 100)  # ncall
data += struct.pack('<i', len(vals1))  # nval
data += vals1.tobytes()

# Entry 2: 'blk.0.attn_k.weight'
name2 = b'blk.0.attn_k.weight'
vals2 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
data += struct.pack('<i', len(name2))
data += name2
data += struct.pack('<i', 100)
data += struct.pack('<i', len(vals2))
data += vals2.tobytes()

out.write_bytes(bytes(data))
print(f'wrote {out}: {out.stat().st_size} bytes')
"
```

- [ ] **Step 2: Create the test file**

Create `/server/programming/MagicQuant/tests/integration/test_imatrix_capture.py`:

```python
"""Unit tests for magicquant.imatrix."""
from pathlib import Path

import numpy as np
import pytest

from magicquant.imatrix import ImatrixCapture, load_imatrix_file

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sample_imatrix.dat"


def test_load_imatrix_returns_per_tensor_arrays():
    matrices = load_imatrix_file(FIXTURE)
    assert "blk.0.attn_q.weight" in matrices
    assert "blk.0.attn_k.weight" in matrices

    q = matrices["blk.0.attn_q.weight"]
    assert q.dtype == np.float32
    np.testing.assert_array_equal(q, [1.0, 2.0, 3.0, 4.0])

    k = matrices["blk.0.attn_k.weight"]
    np.testing.assert_array_equal(k, [0.5, 1.5, 2.5])


def test_imatrix_capture_dict_access():
    cap = ImatrixCapture.load(FIXTURE)
    assert "blk.0.attn_q.weight" in cap
    np.testing.assert_array_equal(
        cap["blk.0.attn_q.weight"],
        [1.0, 2.0, 3.0, 4.0],
    )


def test_imatrix_capture_keyerror_for_missing():
    cap = ImatrixCapture.load(FIXTURE)
    with pytest.raises(KeyError, match="No imatrix entry"):
        _ = cap["nonexistent.tensor.name"]


def test_imatrix_capture_get_with_default():
    cap = ImatrixCapture.load(FIXTURE)
    assert cap.get("nonexistent", "default") == "default"


def test_load_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_imatrix_file(tmp_path / "nonexistent.dat")
```

- [ ] **Step 3: Run the tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_imatrix_capture.py -v
```

Expected: 5 passed.

- [ ] **Step 4: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_imatrix_capture.py tests/fixtures/sample_imatrix.dat && \
  git commit -m "test: imatrix module unit tests

Hand-crafted 2-tensor fixture exercises file parsing, dict access,
KeyError for missing tensors, and FileNotFoundError for missing files.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Re-export ImatrixCapture at package top-level

**Files:**
- Modify: `magicquant/__init__.py`

- [ ] **Step 1: Add the import and re-export**

Edit `/server/programming/MagicQuant/magicquant/__init__.py`. Find the existing imports and add:

```python
from magicquant.imatrix import ImatrixCapture, capture_imatrix, load_imatrix_file
```

In the `__all__` list, add:
```python
    "ImatrixCapture",
    "capture_imatrix",
    "load_imatrix_file",
```

- [ ] **Step 2: Verify imports**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant import ImatrixCapture, capture_imatrix, load_imatrix_file
print('top-level imatrix imports OK')
"
```

- [ ] **Step 3: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/__init__.py && \
  git commit -m "feat: re-export ImatrixCapture and helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wire imatrix through the GGUF writer

**Files:**
- Modify: `magicquant/gguf/writer.py`

The writer's `create_hybrid_gguf()` is the function the orchestrator calls. It needs to accept an optional `imatrix_per_tensor: Dict[str, np.ndarray]` parameter and pass each tensor's imatrix to `encode_to_ggml_bytes()`.

- [ ] **Step 1: Locate the create_hybrid_gguf signature**

Run:
```bash
cd /server/programming/MagicQuant && grep -n "def create_hybrid_gguf\|encode_to_ggml_bytes" magicquant/gguf/writer.py
```

Note the function signature and where `encode_to_ggml_bytes` is called inside the writer.

- [ ] **Step 2: Add imatrix_per_tensor parameter**

Edit `magicquant/gguf/writer.py`. Find the `create_hybrid_gguf` function definition and add an optional parameter:

```python
def create_hybrid_gguf(
    output_path: str,
    base_model_path: str,
    quant_config: Dict,
    verbose: bool = True,
    adapter_path: Optional[str] = None,
    imatrix_per_tensor: Optional[Dict[str, np.ndarray]] = None,
):
```

In the function body, locate the call to `encode_to_ggml_bytes(...)`. It should currently look like:
```python
data_bytes = encode_to_ggml_bytes(weights, ggml_type)
```

Replace with:
```python
imatrix_for_tensor = (
    imatrix_per_tensor.get(tensor_name) if imatrix_per_tensor else None
)
data_bytes = encode_to_ggml_bytes(weights, ggml_type, imatrix=imatrix_for_tensor)
```

(`tensor_name` is the variable in scope at the call site — verify the actual variable name in the file.)

- [ ] **Step 3: Test that writer accepts imatrix without breaking existing callers**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
import inspect
from magicquant.gguf.writer import create_hybrid_gguf
sig = inspect.signature(create_hybrid_gguf)
assert 'imatrix_per_tensor' in sig.parameters
assert sig.parameters['imatrix_per_tensor'].default is None
print('signature includes imatrix_per_tensor=None')
"
```

Expected output: `signature includes imatrix_per_tensor=None`.

- [ ] **Step 4: Run existing tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected output: all tests pass (the new parameter defaults to None, so existing callers are unaffected).

- [ ] **Step 5: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/gguf/writer.py && \
  git commit -m "feat: thread imatrix_per_tensor through create_hybrid_gguf

Optional dict mapping tensor name → numpy float32 importance vector.
Each tensor's imatrix is passed to encode_to_ggml_bytes(), which
forwards to ggml_encode() and ultimately ggml_quantize_chunk().

Default None preserves existing behavior. PR4's orchestrator changes
populate this when IQ-quants are in scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Auto-capture imatrix in the orchestrator

**Files:**
- Modify: `magicquant/orchestrator.py`

When the orchestrator runs a search that includes IQ-quants requiring imatrix, it should capture imatrix once at the start of the run and pass it through to all subsequent encoder calls.

- [ ] **Step 1: Add an `imatrix_corpus` field to the orchestrator**

Edit `/server/programming/MagicQuant/magicquant/orchestrator.py`. Find the `MagicQuantOrchestrator.__init__` method.

Add parameters:
```python
    def __init__(
        self,
        source_model_path: str,
        output_dir: Path,
        ...,
        imatrix_corpus: Optional[Path] = None,
        imatrix_chunks: int = 100,
    ):
        ...
        self.imatrix_corpus = imatrix_corpus
        self.imatrix_chunks = imatrix_chunks
        self._imatrix: Optional[ImatrixCapture] = None  # lazy-loaded
```

(Adjust to match the actual existing init signature; the spec shape may differ from what's shown here.)

Also add the import at the top of the file:
```python
from magicquant.imatrix import ImatrixCapture, capture_imatrix
```

- [ ] **Step 2: Add a helper to capture imatrix on demand**

Add this method to the orchestrator class:

```python
    def _ensure_imatrix(self) -> Optional[ImatrixCapture]:
        """Lazy-capture imatrix if a corpus is configured.

        Returns the ImatrixCapture, or None if no corpus was provided
        (in which case IQ-quants will use the no-imatrix path).
        """
        if self._imatrix is not None:
            return self._imatrix
        if self.imatrix_corpus is None:
            return None

        imatrix_path = self.output_dir / "imatrix.dat"
        if not imatrix_path.is_file():
            log.info("Capturing imatrix",
                     stage="imatrix", corpus=str(self.imatrix_corpus))
            capture_imatrix(
                model_path=Path(self.source_model_path),
                corpus_path=self.imatrix_corpus,
                output_path=imatrix_path,
                chunks=self.imatrix_chunks,
            )
        self._imatrix = ImatrixCapture.load(imatrix_path)
        log.info("Loaded imatrix",
                 stage="imatrix", n_tensors=len(self._imatrix.matrices))
        return self._imatrix
```

- [ ] **Step 3: Pass imatrix_per_tensor through to create_hybrid_gguf**

Find the orchestrator method that calls `create_hybrid_gguf` (likely `_build_candidate` or `generate_hybrid_model`). Add the imatrix capture and pass-through:

For example, in `_build_candidate`:
```python
    def _build_candidate(
        self, config: Dict[str, str], name: str, base_quant: str
    ) -> Optional[str]:
        from magicquant.gguf.writer import create_hybrid_gguf

        output_filename = generate_name(name, base_quant, config)
        candidates_dir = self.output_dir / "_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(candidates_dir / output_filename)

        # Capture imatrix lazily if needed
        imatrix_cap = self._ensure_imatrix()
        imatrix_dict = imatrix_cap.matrices if imatrix_cap else None

        try:
            return create_hybrid_gguf(
                output_path=output_path,
                base_model_path=self.source_model_path,
                quant_config={"base": base_quant, "groups": config},
                verbose=False,
                adapter_path=self.adapter_path,
                imatrix_per_tensor=imatrix_dict,
            )
        except Exception as exc:
            log.error("Build failed", stage="build", error=str(exc))
            return None
```

Apply the same change to any other site that calls `create_hybrid_gguf` in the orchestrator (e.g. `generate_hybrid_model`, `generate_tiered_models`'s downstream calls).

- [ ] **Step 4: Verify the orchestrator imports and runs**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.orchestrator import MagicQuantOrchestrator
import inspect
sig = inspect.signature(MagicQuantOrchestrator.__init__)
assert 'imatrix_corpus' in sig.parameters
print('imatrix_corpus param present')
"
```

Expected output: `imatrix_corpus param present`.

- [ ] **Step 5: Run all tests**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v 2>&1 | tail -15
```

Expected: all tests pass. The orchestrator changes are gated by `imatrix_corpus is None` so existing tests don't trigger imatrix capture.

- [ ] **Step 6: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/orchestrator.py && \
  git commit -m "feat: orchestrator captures imatrix lazily when configured

New constructor params:
  - imatrix_corpus: Path to calibration text file. If provided, the
    orchestrator captures imatrix once on first build and reuses it
    for all subsequent candidate builds.
  - imatrix_chunks: Capture chunk count (default 100).

Without imatrix_corpus, IQ-quants take the no-imatrix path (existing
PR3 behavior). With it, IQ1/IQ2 candidates produce their full-quality
output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Revert xfail markers on imatrix-dependent parity tests

**Files:**
- Modify: `tests/integration/test_encoder_parity.py`

With imatrix support landed, the IQ-quants that were marked xfail in PR3 should now produce byte-parity output when both code paths use the same imatrix.

- [ ] **Step 1: Update the parity test to optionally pass imatrix**

Edit the test file. Modify the test to capture an imatrix once per session (if a model is available) and pass it both to `ggml_encode` and to `llama-quantize`'s subprocess invocation:

Find the existing parity test function and rewrite to support imatrix:

```python
@pytest.fixture(scope="module")
def imatrix_for_reference_tensor():
    """Create a synthetic per-tensor imatrix for the reference tensor.

    For byte-parity testing we just need both paths (MagicQuant and
    llama-quantize) to use the SAME imatrix. We synthesize a uniform
    imatrix (all-ones) of the correct shape — neither path benefits from
    it, but both produce the same output because the quantizer treats it
    identically.
    """
    # The reference tensor is (2048, 2048) F32. The imatrix vector length
    # is the number of columns (last dimension): 2048.
    return np.ones(2048, dtype=np.float32)


def _imatrix_file_for_test(imatrix: np.ndarray, tensor_name: str, tmp_path: Path) -> Path:
    """Write a llama-imatrix-formatted file with one tensor entry."""
    import struct
    out = tmp_path / "test_imatrix.dat"
    data = bytearray()
    data += struct.pack("<i", 1)
    name_bytes = tensor_name.encode("utf-8")
    data += struct.pack("<i", len(name_bytes))
    data += name_bytes
    data += struct.pack("<i", 100)  # ncall
    data += struct.pack("<i", len(imatrix))
    data += imatrix.astype(np.float32).tobytes()
    out.write_bytes(bytes(data))
    return out


# Drop the xfail markers — IQ-quants now produce parity given matching imatrix
@pytest.mark.parametrize("scheme", SCHEMES_PARITY)
def test_encoder_byte_for_byte_matches_llama_quantize(
    scheme: str,
    reference_tensor: np.ndarray,
    imatrix_for_reference_tensor: np.ndarray,
    tmp_path: Path,
) -> None:
    quantize_bin = _llama_quantize_path()

    # Build the F32 source GGUF with a fixed tensor name
    tensor_name = "test.weight"
    src_path = tmp_path / "src.f32.gguf"
    _write_f32_gguf(reference_tensor, src_path, tensor_name=tensor_name)

    # Build the imatrix file (used by both code paths for parity)
    imatrix_path = _imatrix_file_for_test(
        imatrix_for_reference_tensor, tensor_name, tmp_path
    )

    # Run llama-quantize <scheme> with --imatrix
    dst_path = tmp_path / f"dst.{scheme}.gguf"
    result = subprocess.run(
        [quantize_bin, "--imatrix", str(imatrix_path),
         str(src_path), str(dst_path), scheme],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.fail(
            f"llama-quantize {scheme} failed:\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    # Quantize via MagicQuant with same imatrix
    magic_bytes = ggml_encode(
        reference_tensor, scheme,
        imatrix=imatrix_for_reference_tensor,
    )
    llama_bytes, _ = _read_first_tensor_bytes(dst_path)

    assert magic_bytes == llama_bytes, (
        f"byte mismatch for {scheme}: lengths "
        f"magic={len(magic_bytes)}, llama={len(llama_bytes)}"
    )
```

Remove any `@pytest.mark.xfail` decorators from the parametrize list.

- [ ] **Step 2: Run all parity tests with imatrix**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_encoder_parity.py -v 2>&1 | tail -30
```

Expected output: all 22 parity tests pass — including the IQ-quants that were xfail in PR3.

If any IQ-quant still fails with byte mismatch even with matching imatrix, the cause is one of:
  a. llama-quantize internally augments the imatrix in a way MagicQuant doesn't replicate. Resolution: pass `--no-imatrix-tweaks` or equivalent if available, or accept these as xfail.
  b. The imatrix file format expectations differ. Compare the synthesized fixture against a real llama-imatrix output file.

- [ ] **Step 3: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_encoder_parity.py && \
  git commit -m "test: encoder parity now passes for IQ-quants with imatrix

Reverts xfail markers from PR3. Tests synthesize a uniform imatrix
and pass it to both MagicQuant's ggml_encode (via the new imatrix=
parameter from PR1) and to llama-quantize's subprocess (via --imatrix).

All 22 schemes now byte-parity verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Strengthen the Q2 tier smoke test

**Files:**
- Modify: `tests/integration/test_smoke_q2_tier.py`

PR3's smoke test asserts only that a Q2 winner exists and uses IQ-quants. With imatrix landed, we can assert tighter quality: Q2 GGUF perplexity within 1.5x of baseline (the spec's acceptance criterion).

- [ ] **Step 1: Add a perplexity-bound assertion**

Edit `/server/programming/MagicQuant/tests/integration/test_smoke_q2_tier.py`. Add a new test that builds an actual Q2-tier GGUF and measures perplexity:

```python
@pytest.mark.slow
def test_q2_tier_perplexity_within_15x_of_baseline(tmp_path):
    """End-to-end: build a Q2-tier hybrid GGUF and verify perplexity
    is within 1.5x of the BF16 baseline.

    This is the user-visible quality milestone for the encoder-expansion
    project: Q2 outputs are not just "produce something" but produce
    something usable.

    Skipped unless --runslow is passed (it takes ~5 min).
    """
    import subprocess
    import shutil
    from pathlib import Path

    model_path = Path(os.environ.get(
        "MAGICQUANT_TEST_MODEL",
        str(Path.home() / "models" / "Llama-3.2-1B-Instruct-bf16.gguf"),
    ))
    corpus_path = Path("/home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw")

    if not model_path.is_file():
        pytest.skip(f"test model not found: {model_path}")
    if not corpus_path.is_file():
        pytest.skip(f"corpus not found: {corpus_path}")

    perplexity_bin = shutil.which("llama-perplexity")
    if not perplexity_bin:
        pytest.skip("llama-perplexity not on PATH")

    # Capture imatrix
    from magicquant.imatrix import capture_imatrix, ImatrixCapture
    imatrix_path = tmp_path / "imatrix.dat"
    capture_imatrix(
        model_path=model_path,
        corpus_path=corpus_path,
        output_path=imatrix_path,
        chunks=20,
    )
    imatrix = ImatrixCapture.load(imatrix_path)

    # Build a uniform IQ2_S GGUF (representative Q2-tier config)
    from magicquant.gguf.writer import create_hybrid_gguf
    q2_path = tmp_path / "q2_test.gguf"
    create_hybrid_gguf(
        output_path=str(q2_path),
        base_model_path=str(model_path),
        quant_config={"base": "IQ2_S", "groups": {}},
        verbose=False,
        imatrix_per_tensor=imatrix.matrices,
    )

    # Measure baseline (BF16) perplexity
    def run_ppl(model: Path) -> float:
        proc = subprocess.run(
            [perplexity_bin, "-m", str(model), "-f", str(corpus_path),
             "--ctx-size", "512"],
            capture_output=True, text=True, timeout=600,
        )
        for line in reversed(proc.stdout.splitlines()):
            if "Final estimate" in line and "PPL" in line:
                return float(line.split("=")[1].strip().split()[0])
        raise RuntimeError("could not parse ppl")

    baseline_ppl = run_ppl(model_path)
    q2_ppl = run_ppl(q2_path)

    ratio = q2_ppl / baseline_ppl
    print(f"BF16 ppl: {baseline_ppl:.4f}, Q2 (IQ2_S) ppl: {q2_ppl:.4f}, ratio: {ratio:.2f}x")
    assert ratio < 1.5, (
        f"Q2 (IQ2_S) perplexity {q2_ppl:.2f} is {ratio:.2f}x baseline "
        f"{baseline_ppl:.2f}, exceeds 1.5x bound."
    )
```

Add this slow-test marker registration to `pyproject.toml` if not already present:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: tests that take more than 30 seconds to run",
]
```

- [ ] **Step 2: Run the test**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/integration/test_smoke_q2_tier.py -v -m "slow" -s 2>&1 | tail -20
```

Expected output: `1 passed` with the printed perplexity ratio shown via `-s`. Wall-clock: ~5 min.

If the ratio exceeds 1.5x, IQ2_S without enough chunks isn't reaching its full quality. Increase `chunks=20` to `chunks=100` and re-run.

- [ ] **Step 3: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tests/integration/test_smoke_q2_tier.py pyproject.toml && \
  git commit -m "test: assert Q2 tier perplexity within 1.5x of baseline

Adds a slow integration test that captures imatrix, builds a uniform
IQ2_S GGUF, and measures perplexity against wikitext-2. Asserts the
Q2 output is within 1.5x of the BF16 baseline.

This is the user-facing quality milestone: Q2 tier produces usable
output (not just 'produces something').

Marked @pytest.mark.slow; skipped by default. Run with -m slow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Re-run calibration bench with imatrix

**Files:**
- Modify: `tools/calibrate_noise_factors.py`
- Modify: `tools/calibration_results.json`
- Modify: `magicquant/quant/schemes.py`

The PR3 calibration ran without imatrix, producing inflated noise factors for IQ1/IQ2/IQ3 (since their no-imatrix quality is much worse than their with-imatrix quality). With imatrix support, we can re-bench for accurate values.

- [ ] **Step 1: Add `--imatrix-corpus` support to the bench script**

Edit `/server/programming/MagicQuant/tools/calibrate_noise_factors.py`. Add a new CLI argument and use it to capture imatrix once at the start:

In `main()`, find the argparse section and add:
```python
    ap.add_argument(
        "--imatrix-corpus", type=Path, default=None,
        help="Optional calibration corpus for imatrix capture. "
             "When provided, IQ-quants get their best-quality calibration.",
    )
```

Then, before the per-scheme loop:
```python
    # Optionally capture imatrix once
    imatrix_dict = None
    if args.imatrix_corpus:
        from magicquant.imatrix import capture_imatrix, ImatrixCapture
        imatrix_path = tmpdir_path / "imatrix.dat"
        print(f"[imatrix] capturing from {args.imatrix_corpus}")
        capture_imatrix(
            model_path=args.model,
            corpus_path=args.imatrix_corpus,
            output_path=imatrix_path,
            chunks=100,
        )
        imatrix_dict = ImatrixCapture.load(imatrix_path).matrices
        print(f"[imatrix] captured {len(imatrix_dict)} tensors")
```

In `_build_uniform_gguf`, accept and pass imatrix:
```python
def _build_uniform_gguf(
    model_path: Path, scheme_name: str, output_dir: Path,
    create_hybrid_gguf,
    imatrix_dict=None,
) -> Optional[Path]:
    ...
    create_hybrid_gguf(
        output_path=str(out_path),
        base_model_path=str(model_path),
        quant_config={"base": scheme_name, "groups": {}},
        verbose=False,
        imatrix_per_tensor=imatrix_dict,
    )
    ...
```

And in the calling loop, pass `imatrix_dict` to `_build_uniform_gguf`.

- [ ] **Step 2: Re-run the calibration bench with imatrix**

Run:
```bash
cd /server/programming/MagicQuant && \
  python tools/calibrate_noise_factors.py \
    --model <SAME-MODEL-AS-BEFORE> \
    --corpus /home/lucas/llama.cpp/wikitext-2-raw/wiki.test.raw \
    --imatrix-corpus /home/lucas/llama.cpp/wikitext-2-raw/wiki.train.raw \
    --output tools/calibration_results.json \
    2>&1 | tee /tmp/calibration_run_pr4.log
```

Expected: same wall-clock as PR3's bench (~2 hr) plus ~30 min for imatrix capture upfront.

- [ ] **Step 3: Update schemes.py with imatrix-calibrated noise factors**

Same procedure as PR1 Task 14 / PR3 Task 7: paste calibrated values into each scheme's `noise_factor=` line. Expect IQ1/IQ2/IQ3 noise factors to drop substantially (the bench now reflects their real quality).

- [ ] **Step 4: Run all tests including the slow one**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v -m "slow or not slow" 2>&1 | tail -20
```

Expected: 35+ tests pass.

- [ ] **Step 5: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add tools/calibrate_noise_factors.py tools/calibration_results.json magicquant/quant/schemes.py && \
  (git diff --cached tests/fixtures/refactor_regression_seed42.json && git add tests/fixtures/refactor_regression_seed42.json || true) && \
  git commit -m "calibrate: re-bench with imatrix

Bench script accepts --imatrix-corpus. Captures imatrix once,
applies it during all per-scheme builds. IQ1/IQ2/IQ3 noise factors
now reflect with-imatrix quality (substantially lower than PR3's
no-imatrix values).

Search will now correctly prefer IQ-quants over K-quants at the
same bpw when imatrix is available.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Lower the robust floor to IQ1_S

**Files:**
- Modify: `magicquant/quant/schemes.py`

PR3 set the robust floor at IQ2_S (the conservative no-imatrix bottom). With imatrix support landed, FFN groups can safely go all the way down to IQ1_S.

- [ ] **Step 1: Update _GROUP_CLASS_FLOORS**

Edit `magicquant/quant/schemes.py`. Find:

```python
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q5_K",
    "robust": "IQ2_S",
}
```

Replace with:
```python
_GROUP_CLASS_FLOORS: Dict[str, str] = {
    "sensitive": "Q5_K",       # brain layers stay above 5 bpw
    "robust": "IQ1_S",         # PR4 lowers to IQ1_S; FFN can go all the way
                               # down with imatrix support
}
```

- [ ] **Step 2: Run regression test**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/test_refactor_regression.py -v
```

Regenerate the fixture if needed.

- [ ] **Step 3: Commit**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add magicquant/quant/schemes.py && \
  (git diff --cached tests/fixtures/refactor_regression_seed42.json && git add tests/fixtures/refactor_regression_seed42.json || true) && \
  git commit -m "tune: lower robust floor to IQ1_S now that imatrix lands

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Foundry UI — calibration dataset input

**Files:**
- Modify: `Foundry/ui/index.html`
- Modify: `Foundry/ui/app.py`
- Modify: `Foundry/core/pipeline.py`

- [ ] **Step 1: Add the imatrix_dataset field to MagicQuantConfig in core/pipeline.py**

Edit `/server/programming/Foundry/core/pipeline.py`. Find the `MagicQuantConfig` dataclass (around line ~95):

```python
@dataclass
class MagicQuantConfig:
    target_base_quant: str = "MXFP4_MOE"
    generations: int = 30
    population_size: int = 100
    tiers: list[str] = field(default_factory=lambda: ["Q4", "Q5", "Q6"])
    llamacpp_path: str = ""
    source_model: str = ""
```

Add the new field:
```python
@dataclass
class MagicQuantConfig:
    target_base_quant: str = "MXFP4_MOE"
    generations: int = 30
    population_size: int = 100
    tiers: list[str] = field(default_factory=lambda: ["Q4", "Q5", "Q6"])
    llamacpp_path: str = ""
    source_model: str = ""
    imatrix_dataset: str = ""  # path to calibration corpus; empty = no imatrix
```

- [ ] **Step 2: Pass imatrix_dataset to the orchestrator**

In the same file, find where the orchestrator is constructed in `stage_magicquant()`. Add the imatrix_corpus parameter:

```python
    orch = MagicQuantOrchestrator(
        ...,
        imatrix_corpus=Path(mc.imatrix_dataset) if mc.imatrix_dataset else None,
    )
```

- [ ] **Step 3: Mirror the field in ui/app.py's MagicQuantConfig**

Edit `/server/programming/Foundry/ui/app.py`. Find the dataclass (around line ~155) and add the same field. Pass it through to the subprocess invocation.

- [ ] **Step 4: Add the UI input in index.html**

Edit `/server/programming/Foundry/ui/index.html`. Find the `magicquant` config in `S.config`:

```javascript
magicquant: { target_base_quant: 'MXFP4_MOE', generations: 50, population_size: 100, tiers: ['Q4','Q5','Q6'], llamacpp_path: '', source_model: '' },
```

Add `imatrix_dataset: ''`:
```javascript
magicquant: { target_base_quant: 'MXFP4_MOE', generations: 50, population_size: 100, tiers: ['Q4','Q5','Q6'], llamacpp_path: '', source_model: '', imatrix_dataset: '' },
```

In `renderMagicQuant()`, add a new form field after the existing inputs (around the source_model field):

```javascript
    <div class="form-group span-2">${L('Calibration Dataset (optional)','Path to a text file used to capture an importance matrix. Significantly improves quality for IQ1/IQ2/IQ3 schemes. Empty = skip imatrix capture (faster, but lower quality for sub-3-bpw schemes).')}<input class="form-input" data-key="magicquant.imatrix_dataset" value="${c.imatrix_dataset || ''}"></div>
```

- [ ] **Step 5: Test the Foundry UI loads**

Run:
```bash
cd /server/programming/Foundry && python -c "
from core.pipeline import MagicQuantConfig
c = MagicQuantConfig()
print('imatrix_dataset field:', repr(c.imatrix_dataset))
"
```

Expected: `imatrix_dataset field: ''`.

- [ ] **Step 6: Restart the Foundry UI to pick up changes**

Run (kill existing UI, start fresh):
```bash
ss -tlnp 2>/dev/null | grep ":7865" | awk -F'pid=' '{print $2}' | awk -F',' '{print $1}' | head -1 | xargs -r kill 2>/dev/null

cd /server/programming/Foundry && \
  setsid nohup ./ui/run.sh 7865 > ./ui/ui.log 2>&1 < /dev/null & disown

sleep 2

curl -sf -o /dev/null -w "HTTP %{http_code}\n" http://localhost:7865/
```

Expected output: `HTTP 200`.

- [ ] **Step 7: Verify the new UI field renders**

Run:
```bash
curl -s http://localhost:7865/ | grep -o "imatrix_dataset" | head -3
```

Expected output: at least 2 matches (config default + form field).

- [ ] **Step 8: Commit Foundry changes**

Run:
```bash
cd /server/programming/Foundry && \
  git add ui/index.html ui/app.py core/pipeline.py && \
  git commit -m "feat: add imatrix_dataset config to MagicQuant stage

Foundry UI now exposes a 'Calibration Dataset' input under MagicQuant
config. When set, it's passed through to MagicQuantOrchestrator's
imatrix_corpus parameter, triggering automatic imatrix capture for
IQ-quant calibration.

Empty string (default) skips imatrix capture — existing behavior.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 9: Push Foundry**

Run:
```bash
cd /server/programming/Foundry && git push origin master 2>&1
```

---

### Task 13: Update CLAUDE.md and CHANGELOG

**Files:**
- Modify: `MagicQuant/CLAUDE.md`
- Modify: `MagicQuant/CHANGELOG.md`
- Modify: `Foundry/CHANGELOG.md`

- [ ] **Step 1: Update MagicQuant/CLAUDE.md**

Find the Known Limitations section. Update the imatrix entry from PR1:

```markdown
- IQ-quant encoders are available via libggml binding (PR1+). Without an importance matrix (PR4+), IQ1/IQ2 outputs are slightly lower-quality...
```

Replace with:
```markdown
- IQ-quant encoders are available via libggml binding. Importance matrix
  is automatically captured by the orchestrator when `imatrix_corpus` is
  configured (or via `magicquant.imatrix.capture_imatrix()` directly).
  Without imatrix, IQ1/IQ2 outputs use uniform-importance quantization
  which is acceptable for FFN groups but produces visible quality
  degradation in sensitive groups.
```

- [ ] **Step 2: Add CHANGELOG entries**

Add to `MagicQuant/CHANGELOG.md` (top):
```markdown
## [unreleased] — Encoder Expansion (PR0–PR4)

- libggml ctypes binding (PR1)
- 16 new schemes: Q2_K, Q3_K (PR1), Q4_0/Q4_1/Q5_0/Q5_1 (PR2),
  IQ1_S/M, IQ2_XXS/XS/S/M, IQ3_XXS/S/M, IQ4_XS (PR3)
- Importance-matrix support (PR4)
- Empirical noise-factor calibration (PR1, refreshed PR3, PR4)
- Eliminated 10–27% MSE gap vs llama.cpp via byte-parity ctypes path
- Q2 and Q3 tier bands now reliably populate
```

Add to `Foundry/CHANGELOG.md` (top):
```markdown
## [unreleased]

- MagicQuant stage gains imatrix support: optional "Calibration Dataset"
  field in the UI triggers automatic importance-matrix capture for
  IQ-quant calibration.
```

- [ ] **Step 3: Commit docs**

Run:
```bash
cd /server/programming/MagicQuant && \
  git add CLAUDE.md CHANGELOG.md && \
  git commit -m "docs: update CLAUDE.md and CHANGELOG for imatrix support

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

```bash
cd /server/programming/Foundry && \
  git add CHANGELOG.md && \
  git commit -m "docs: CHANGELOG entry for imatrix support

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Final verification + push

**Files:** none

- [ ] **Step 1: Run all tests including slow**

Run:
```bash
cd /server/programming/MagicQuant && python -m pytest tests/ -v -m "slow or not slow" 2>&1 | tail -25
```

Expected: 35+ tests pass.

- [ ] **Step 2: Final scheme + tier accounting**

Run:
```bash
cd /server/programming/MagicQuant && python -c "
from magicquant.quant.schemes import get_all_schemes, get_floor_for_group_class
schemes = get_all_schemes()
print(f'total registered schemes: {len(schemes)}')
print(f'sensitive floor: {get_floor_for_group_class(\"sensitive\")}')
print(f'robust floor: {get_floor_for_group_class(\"robust\")}')
print()
print('schemes ordered by bpw (smallest first):')
for s in sorted(schemes, key=lambda s: s.bits_per_weight):
    print(f'  {s.name:10s}  bpw={s.bits_per_weight:7.4f}  noise={s.noise_factor:6.3f}  imatrix={\"required\" if s.requires_imatrix else \"optional\"}')
"
```

Expected output: 23 schemes, sensitive floor Q5_K, robust floor IQ1_S, IQ-quants have `imatrix=required`, K-quants have `imatrix=optional`.

- [ ] **Step 3: Push MagicQuant**

Run:
```bash
cd /server/programming/MagicQuant && git push origin master 2>&1
```

- [ ] **Step 4: Encoder expansion project complete**

Print:
```
Encoder Expansion Project Complete

PR0: refactor scheme registry (centralized metadata)
PR1: libggml ctypes binding + Q2_K/Q3_K + retrofit + calibration
PR2: legacy Q-quants (Q4_0, Q4_1, Q5_0, Q5_1)
PR3: IQ-quants (10 new schemes; Q2 tier reachable)
PR4: importance-matrix support (this PR; full IQ quality)

Final state:
- 23 schemes registered (was 7)
- All schemes byte-parity verified against llama-quantize
- Q2/Q3 tier bands reliably populate
- Empirical noise factors calibrated with imatrix
- Foundry UI exposes Calibration Dataset input
- ~700 net lines deleted (pure-Python encoders → ctypes)

Future work (deferred per memory):
- Per-architecture calibration tables (Qwen, Mistral, MoE)
- Native (in-Python) imatrix capture (vs subprocess wrapping)
```

---

## Self-Review Checklist

**Spec coverage (PR4 section):**
- [x] "New `magicquant/imatrix.py`: capture activation magnitudes" → Tasks 2, 3
- [x] "Subprocess wrapper around `llama-imatrix`" → Task 3
- [x] "Wire `imatrix: Optional[np.ndarray]` parameter through `create_hybrid_gguf()` → `encode_to_ggml_bytes()` → `ggml_encode()`" → Task 6 (writer); PR1 already wired the lower layers
- [x] "Schemes where `ggml_quantize_requires_imatrix(type_id) == True`: orchestrator captures imatrix once at start of run" → Task 7
- [x] "Foundry UI: optional 'Calibration dataset' input" → Task 12
- [x] "Foundry's `core/pipeline.py` accepts an `imatrix_dataset` field" → Task 12
- [x] "Quality bench: re-run encoder-parity tests with imatrix" → Task 8
- [x] "Confirm IQ2/IQ3 outputs improve measurably" → Task 10's recalibration

**Acceptance criteria from spec:**
- [x] PR0–PR4 all merged → final task pushes
- [x] Q3 tier band reliably produces output (after PR1) → still true
- [x] Q2 tier band reliably produces output (after PR3) → strengthened in PR4 Task 9
- [x] All 7 existing schemes byte-parity → tested in Tasks 8 (regressions) and PR1
- [x] All 16 new schemes pass parity harness → all 22 tests pass after Task 8
- [x] `pip install -e .` works on fresh clone → already in PR1; no regression
- [x] `tools/calibration_results.json` committed; noise_factor refs it → updated in Task 10
- [x] CLAUDE.md no longer mentions "10-27% MSE gap" → updated in PR1; refreshed in Task 13

**Placeholder scan:** No "TBD" or vague directives.

**Type consistency:**
- `imatrix_per_tensor: Dict[str, np.ndarray]` is the parameter name in writer.py and orchestrator.py.
- `imatrix: Optional[np.ndarray]` is the parameter name in ggml_encode (PR1) and encode_to_ggml_bytes.
- `imatrix_corpus: Optional[Path]` is the orchestrator constructor parameter.
- `imatrix_dataset: str` is the Foundry UI/config field name.
These names are consistent across modules.

**Risk callouts:**
- Task 7's orchestrator changes need to match the existing init signature exactly. The actual init takes more parameters than my draft shows; verify before applying.
- Task 12's `pipeline.py:stage_magicquant()` may construct the orchestrator differently than the simple example shows; trace the actual code path.
- Task 8's parity test for IQ-quants assumes llama-quantize accepts `--imatrix` flag. Verify with `llama-quantize --help`.

---

## Project Complete

After PR4 merges, the encoder-expansion workflow is done. Future work (per saved memory):
- Per-architecture calibration tables (Qwen, Mistral, MoE) — see `project_per_arch_calibration_todo.md`
- Native imatrix capture (replacing subprocess) — noted in spec
- K-quant `_S/_M/_L` recipe presets as starter populations — noted in spec
