> **SUPERSEDED — pre-libggml binding (April 2026).** This audit predates the
> May 2026 ctypes refactor (numpy K-quant encoders replaced by a direct
> `libggml` binding, byte-identical to llama.cpp). Its encoder findings are
> fixed at HEAD. The still-valid structural items (S/X/R search gap,
> GGUF-version reader gap, tied_word_embeddings default) were carried into
> `AUDIT_2026-06-09.md`. Kept for historical reference only.

# MagicQuant Deep Logic Audit

Audited: every `.py` file under `magicquant/`
Date: 2026-04-05

---

## CRITICAL -- Silent Wrong Results

### 1. Collapse penalty is self-amplifying and produces absurd loss values

**File:** `magicquant/evolution/predictor.py:116-121`
**Severity:** CRITICAL

The collapse penalty formula multiplies `total_loss` by `(1 + total_loss * 1.5 + 0.02)`.
This means the penalty is proportional to the *square* of the base loss. For a
moderate `total_loss` of 2.0, the multiplier becomes `1 + (2.0 * 1.5 + 0.02) = 4.02`,
yielding a final loss of 8.04 -- a 4x inflation. For `total_loss` of 4.0, the
multiplier is 7.02, yielding 28.08. This quadratic explosion makes the predictor
strongly discourage *any* compression of brain layers, even reasonable schemes like
Q8_0, because the penalty is so disproportionate that it overwhelms the fitness
score. The evolutionary search will almost never discover configs where brain layers
use anything other than BF16.

```python
collapse_penalty = (
    total_loss * self.collapse_penalty_alpha +   # 1.5 * total_loss
    self.collapse_penalty_beta                    # 0.02
)
total_loss *= (1 + collapse_penalty)  # total_loss * (1 + 1.5*total_loss + 0.02)
```

**Example:** Config `{E: Q8_0, H: Q8_0, Q: MXFP4, K: MXFP4, O: BF16, U: MXFP4, D: MXFP4}`.
Two brain layers (E, H) are compressed, triggering the penalty. Base loss ~1.8.
Penalty multiplier = 1 + 1.8*1.5 + 0.02 = 3.72. Final loss = 6.7. This config
would be ranked far worse than configs with BF16 brain layers, despite Q8_0
being nearly lossless (~0.1% PPL increase).

**Fix:** The collapse penalty should be additive, not multiplicative against
the already-computed loss. Something like:
`total_loss += collapse_penalty_beta * compressed_sensitive_count**alpha`

---

### 2. Two incompatible tier classification systems produce different tier assignments

**File:** `magicquant/orchestrator.py:635-649` vs `magicquant/evolution/survival.py:228-248` vs `magicquant/evolution/predictor.py:296-314`
**Severity:** CRITICAL

There are THREE different tier classification systems with different boundaries:

| Tier | Orchestrator._classify_tier | EvolutionarySurvivor._classify_into_tiers | TierClassifier |
|------|---------------------------|----------------------------------------|----------------|
| Q6   | 0.45 < r <= 0.65          | r > 0.55                               | 0.65 < r <= 0.80 |
| Q5   | 0.33 < r <= 0.45          | 0.40 < r <= 0.55                       | 0.50 < r <= 0.65 |
| Q4   | 0.22 < r <= 0.33          | 0.28 < r <= 0.40                       | 0.35 < r <= 0.50 |
| Q3   | r <= 0.22                 | r <= 0.28                              | 0.20 < r <= 0.35 |

A config at ratio 0.50 is Q6 in the orchestrator, Q5 in survival, and Q5 in TierClassifier.
A config at ratio 0.36 is Q5 in the orchestrator, Q5 in survival, and Q4 in TierClassifier.

The evolutionary search uses `survival._classify_into_tiers` to classify candidates
during search, but the orchestrator uses `_classify_tier` when selecting final
survivors and generating tiered models. A config that won the "Q5 tier" during
evolution might be reclassified as "Q6" when selecting final survivors, leaving
the real Q5 tier empty.

**Example:** A config predicted at 0.50 * baseline GB wins the Q5 tournament in
`survival.py` (since 0.40 < 0.50 <= 0.55 maps to Q5 there). When the orchestrator
calls `_classify_tier(0.50, baseline)`, ratio=0.50 falls in 0.45 < r <= 0.65,
which is Q6. The Q5 tier has no survivor.

**Fix:** Use a single tier classification function everywhere.

---

### 3. Q4_K asymmetric quantization computes min incorrectly

**File:** `magicquant/quant/converters.py:692`
**Severity:** CRITICAL

```python
sub_mins = np.where(sub_min < 0, -sub_min, 0.0).astype(np.float32)
```

This computes `sub_mins` as the absolute value of `sub_min` when negative, or 0
when positive. This means:
- For a sub-block with values in [-3.0, 5.0], `sub_min = -3.0`, `sub_mins = 3.0`, `rng = 5.0 + 3.0 = 8.0`. Correct.
- For a sub-block with values in [1.0, 5.0] (all positive), `sub_min = 1.0`, `sub_mins = 0.0`, `rng = 5.0 + 0.0 = 5.0`. The offset is 0 but the minimum value is 1.0, so the quantization range [0, 5.0] wastes codes on values below 1.0 that don't exist.

This is not technically *wrong* (the dequantized values will still be approximately correct) but it uses quantization resolution suboptimally for purely positive sub-blocks, which does occur in some architectures (e.g., ReLU activations, positional encodings).

However, the **real** issue is in line 693 -- `rng = sub_max + sub_mins` -- which
assumes the dequantization formula is `val = q * scale - offset`. For sub-blocks
where `sub_min >= 0`, this sets the range to `sub_max` (correct), but the quantization
on line 722 adds `eff_m` (which is 0), and the result should be `val = q * (sub_max/15)`.
This does work, but does not match the llama.cpp asymmetric formula which always uses
`rng = max - min` regardless of sign, giving better precision when the min is far
from zero.

**Fix:** Use `sub_mins = -sub_min_vals` unconditionally (matching llama.cpp's
`make_qx_quants` which always uses the actual min, not clamped to 0).

---

### 4. Q5_K asymmetric quantization has the same min-clamping issue

**File:** `magicquant/quant/converters.py:756`
**Severity:** CRITICAL

Same issue as Q4_K:
```python
sub_mins = np.where(sub_min_vals < 0, -sub_min_vals, 0.0).astype(np.float32)
```

Should be `sub_mins = -sub_min_vals` to match llama.cpp behavior.

---

### 5. `_estimate_simple_size` baseline is BF16 size, but source model may already be quantized

**File:** `magicquant/orchestrator.py:664-677` and `magicquant/evolution/predictor.py:202-229`
**Severity:** HIGH

`_estimate_model_size` returns the *file* size, which is the BF16 size only if
the source is BF16. If the source is a Q8_0 GGUF (which the dtype guard allows
through as "can't decode, pass through"), `baseline_size_gb` will be ~half of
the true BF16 size, and all tier boundaries (which are ratios of this baseline)
will be wrong. The comment on line 228 says "baseline_size_gb is the BF16 (16 bpw)
model size" but nothing validates this assumption.

**Example:** User provides a Q8_0 source (8.5 bpw, half the BF16 size).
`_estimate_model_size` returns the Q8_0 file size. `_classify_tier` computes
`ratio = hybrid_size / q8_size`. An MXFP4 hybrid at 4.25 bpw would be
~50% of the Q8_0 size (ratio=0.50, classified as Q6) when it should be ~27% of
the BF16 size (ratio=0.27, classified as Q4).

**Fix:** Compute baseline_size_gb from parameter count * 2 bytes (BF16), not from file size.

---

### 6. `get_parameter_count` skips 1D tensors

**File:** `magicquant/gguf/reader.py:237-246`
**Severity:** HIGH

```python
if len(shape) >= 2:  # Weight matrices have at least 2 dims
```

This skips all 1D tensors (norms, biases) from the parameter count. While these
are typically small, some architectures have large 1D tensors. More importantly,
`get_bits_per_weight` on line 253-259 divides file size by this incomplete
parameter count, producing an inflated bpw estimate.

**Example:** A model with many 1D bias tensors (GPT-NeoX style) would report
a higher bpw than actual, since the denominator is too small.

**Fix:** Count all tensor elements regardless of dimensionality.

---

### 7. Fallback architecture defaults to "llama" for unknown model types

**File:** `magicquant/gguf/source.py:353`
**Severity:** HIGH

```python
arch = arch_map.get(model_type, "llama")
```

When `model_type` is not in the arch_map (e.g., a new architecture like "hymba",
"megalodon", or "recurrentgemma"), the code silently maps it to "llama". All
GGUF metadata keys will then use the `llama.` prefix (e.g., `llama.block_count`),
but llama.cpp will try to load this as a LLaMA model and fail or produce wrong
results because the actual model has different tensor layouts, attention patterns,
or normalization.

**Example:** A HuggingFace model with `"model_type": "hymba"` gets metadata keys
like `llama.block_count`, `llama.embedding_length`. llama.cpp tries to load it as
a LLaMA variant and either crashes or silently produces garbage output.

**Fix:** Raise an explicit error when `model_type` is not in the arch_map, rather
than silently defaulting. The user can then add the mapping or use a GGUF source
that already has correct metadata.

---

### 8. GGUFSource returns "F16" for unknown tensor types

**File:** `magicquant/gguf/source.py:118-120`
**Severity:** HIGH

```python
def get_source_type_name(self, tensor_name: str) -> str:
    info = self._reader.get_tensor_info(tensor_name)
    if info is None:
        return "F16"
    return self._TYPE_NAME.get(info["data_type"], "F16")
```

If a tensor has an unknown ggml type ID (e.g., a new type added to llama.cpp that
MagicQuant doesn't know about), it silently reports it as "F16". The writer's
`can_decode` check sees "F16" (which is decodable), attempts to decode the raw
data as F16, and produces garbage. This also applies when `get_tensor_info`
returns None (tensor not found).

**Example:** A GGUF with type ID 40 (a hypothetical future type). `get_source_type_name`
returns "F16". Writer thinks it can decode. `read_tensor_f32` gets the byte count
wrong (uses F16 size formula instead of actual type size), reads wrong number of
bytes, and `np.frombuffer` either crashes or produces garbage.

**Fix:** Return the actual type name string (even if unknown), or raise an error.
Never silently map unknowns to a decodable type.

---

## HIGH -- Suboptimal/Fragile Defaults

### 9. generate_tiered_models picks base_quant by mode of scheme values, not by intended tier

**File:** `magicquant/orchestrator.py:581-587`
**Severity:** HIGH

```python
base_quant = max(
    set(config.values()),
    key=lambda s: {
        "BF16": 0, "Q8_0": 1, "Q6_K": 2, "Q5_K": 3,
        "IQ4_NL": 4, "MXFP4_MOE": 5, "Q4_K_M": 6
    }.get(s, 3)
)
```

This picks the scheme with the highest "score" (most compressed) present in the
config as the base quant label for the output filename and GGUF metadata. The
`max` call picks the most compressed scheme used by *any* group. But the priority
map gives "Q4_K_M" score 6 (highest) and "MXFP4_MOE" score 5. So if a config
uses both MXFP4_MOE and Q4_K_M, the base quant will be Q4_K_M even though
MXFP4_MOE might be the dominant scheme.

This is also fragile: the priority dict has a default of 3 for unknown schemes,
which means any new scheme name would tie with Q5_K.

**Fix:** Determine base_quant from the scheme that covers the most parameters
(by count), not by an arbitrary priority ranking. Or simply use the tier name
(which is already known at this point).

---

### 10. Evolutionary search has no termination criterion other than generation count

**File:** `magicquant/evolution/survival.py:111-141`
**Severity:** HIGH

The search runs for exactly `max_generations` generations regardless of whether
it has converged. There is no early stopping based on fitness plateau, population
diversity, or improvement rate. For models where the search converges in 10
generations, the remaining 40 generations are wasted computation.

**Fix:** Add convergence detection: if the best composite_score hasn't improved
by more than epsilon for N consecutive generations, terminate early.

---

### 11. Sensitivity weights of 0 cause division by zero in normalized weights

**File:** `magicquant/evolution/probing.py:105-116`
**Severity:** HIGH

```python
total = sum(max(0, s) for s in self.sensitivity_results.values())
if total == 0:
    return {g: 1.0 / len(self.sensitivity_results) ...}
```

If all sensitivity scores are exactly 0 (e.g., baseline PPL is very high and all
probe PPLs are below it due to noise), all weights become equal (1/N). This means
the predictor treats all groups as equally important, which defeats the purpose
of probing. More subtly, if `baseline_ppl` is 0 (which the orchestrator defaults
to 5.0 but could be 0 if measurement fails in a different code path), the
sensitivity calculation `(ppl - baseline) / baseline` would produce Inf or NaN.

**Fix:** Validate that `baseline_ppl > 0` before probing. If all sensitivities
are 0, warn the user that probing produced no useful signal.

---

### 12. `_heuristic_probe` ignores MoE/SSM group "S"

**File:** `magicquant/evolution/probing.py:252-264`
**Severity:** MEDIUM

The `_GROUP_SENSITIVITY` dict does not include "S" (SSM/linear attention) or "N"
(norms) or "V" (vision). The fallback `sensitivity = _GROUP_SENSITIVITY.get(group, 1.0)`
assigns a default of 1.0 (moderate sensitivity) to these groups. For "S" (SSM),
this is probably too conservative -- SSM layers are typically robust to quantization
similar to FFN layers. For "N" (norms), 1.0 is too low -- norms are tiny but
extremely sensitive.

**Fix:** Add heuristic entries for S, N, V groups.

---

### 13. `param_dist` hardcoded distribution doesn't account for MoE models

**File:** `magicquant/evolution/predictor.py:178-181` and `208-211`
**Severity:** HIGH

```python
param_dist = {
    'E': 0.04, 'H': 0.04, 'Q': 0.12, 'K': 0.12,
    'O': 0.06, 'U': 0.31, 'D': 0.31,
}
```

This hardcoded distribution assumes a dense transformer. In MoE models, expert
parameters (X group) can be 60-80% of total parameters, with U and D being much
smaller (shared FFN). The absence of X, R, and S groups from this distribution
means MoE models get wildly wrong size and speed predictions.

**Example:** Qwen3-30B-A3B is an MoE with 30B total params but 3B active. The
expert parameters are ~90% of the model. `param_dist` assigns 0% to X, so
size prediction ignores most of the model.

**Fix:** Either dynamically compute param_dist from actual tensor metadata, or
have separate distributions for dense vs MoE models.

---

### 14. Evolutionary search Q6 tier has no upper bound in survival.py

**File:** `magicquant/evolution/survival.py:237`
**Severity:** MEDIUM

```python
if ratio > 0.55:
    tier = "Q6"
```

Any config above 0.55 ratio is Q6, including ratio=0.99 (barely compressed at all).
This means near-BF16 configs compete in the Q6 tier and could "win" it, producing
an output that is 95% the size of BF16 with negligible compression benefit.

Note: the orchestrator's `_classify_tier` was already fixed to cap Q6 at 0.65,
but survival.py was not updated to match.

**Fix:** Add an upper bound: `0.55 < ratio <= 0.75` for Q6, and add a Q8 tier
for ratio > 0.75. Or better: use the same function as the orchestrator.

---

### 15. `score_hybrid` size_score is inverted -- larger models score higher

**File:** `magicquant/evolution/predictor.py:254-256`
**Severity:** HIGH

```python
if self.baseline_size_gb > 0:
    size_score = min(1, self.baseline_size_gb / max(predicted_size, 0.01))
```

This computes `size_score = baseline / predicted`. A config that is *larger* than
baseline would get `size_score > 1.0` (capped to 1.0). A config that is half
the baseline gets `size_score = 2.0` (capped to 1.0). A config at 25% of baseline
gets `size_score = 4.0` (capped to 1.0). So ALL configs smaller than baseline
get the same size_score of 1.0, and the score provides zero discrimination between
a Q4 and a Q6 config. Size optimization is effectively disabled.

**Example:** Config A at 0.30 * baseline and Config B at 0.60 * baseline both get
size_score = 1.0. The search has no pressure to prefer the smaller one.

**Fix:** Use a scoring function that differentiates between compression levels,
e.g., `size_score = 1.0 - (predicted_size / baseline_size)` or use the compression
ratio directly.

---

### 16. `_write_metadata_value` integer arrays always use INT32, overflowing for large values

**File:** `magicquant/gguf/writer.py:168-170`
**Severity:** HIGH

```python
elif isinstance(first, int):
    f.write(struct.pack("<I", _GGUF_TYPE_INT32))
    f.write(struct.pack("<Q", len(value)))
    for item in value:
        f.write(struct.pack("<i", int(item)))
```

All integer arrays are written as INT32, but GGUF token IDs, vocabulary sizes,
and other metadata can exceed 2^31. For example, `tokenizer.ggml.token_type` is
a list of ints (0 or 3) which is fine, but `tokenizer.ggml.scores` is a list of
floats that could be passed as ints if they happen to be whole numbers due to
Python's dynamic typing. More critically, if any integer in the array exceeds
2^31-1 (e.g., large token IDs in models with 250k+ vocab), `struct.pack("<i", ...)`
will raise a struct.error.

**Fix:** Check the range of integer values in the array and use INT64 if any
exceed INT32 range. Or use UINT32 for non-negative arrays.

---

### 17. `_flatten_to_max_dims` can silently lose squeezed dimensions

**File:** `magicquant/gguf/source.py:29-36`
**Severity:** MEDIUM

```python
if len(shape) > 2:
    squeezed = [shape[0]]
    for d in shape[1:-1]:
        if d != 1:
            squeezed.append(d)
    squeezed.append(shape[-1])
    shape = squeezed
```

This preserves the first and last dims but squeezes all interior singletons.
For a shape like [4096, 1, 1, 32], it produces [4096, 32], changing n_dims from
4 to 2. This is fine for element count but changes the semantic shape. If a
downstream consumer (llama.cpp) interprets the tensor shape to determine operation
(e.g., 2D = matrix, 4D = conv), the squeezed shape could cause incorrect behavior.

**Fix:** Only squeeze when the result exceeds 4 dims, not unconditionally.

---

## MEDIUM -- Edge Cases

### 18. No validation on user-provided config values

**File:** `magicquant/config.py`
**Severity:** MEDIUM

`MagicQuantSettings` has no validators for:
- `search_generations` (could be 0 or negative -- 0 would skip the search entirely, negative would be an empty range)
- `population_size` (0 or 1 would break tournament selection)
- `candidates_per_round` (0 means no measurement, wasting build time)
- `target_base_quant` (arbitrary string, no validation against known schemes)
- `tiers` (empty list would skip all generation)

**Example:** `MAGICQUANT_SEARCH_GENERATIONS=-5` from environment -> range(-5)
produces an empty loop, search returns 0 configs, user gets empty results with
no error message.

**Fix:** Add pydantic validators: `search_generations >= 1`,
`population_size >= 10`, `target_base_quant in known_schemes`, etc.

---

### 19. Reader doesn't handle GGUF version 2 vs 3 differences

**File:** `magicquant/gguf/reader.py:100-151`
**Severity:** MEDIUM

The reader parses the version field but never uses it. GGUF version 2 uses
32-bit tensor count and metadata count fields, while version 3 uses 64-bit.
The reader always reads 64-bit (`<Q`). If a version 2 file is provided, the
reader will misinterpret the counts and either crash or produce garbage.

**Fix:** Branch on version: use `<I` for 32-bit counts in version 2, `<Q` for
64-bit in version 3.

---

### 20. Regex patterns for tensor classification miss several architectures

**File:** `magicquant/gguf/tensor_groups.py:34-51`
**Severity:** MEDIUM

The explicit patterns are specific to llama.cpp's GGUF naming convention. Several
patterns that could miss:

- **GPT-NeoX/Falcon**: Uses `query_key_value` fused tensors, not separate q/k/v.
  The heuristic fallback catches "query" and "key" but not the fused name.
- **Phi-3**: Uses `qkv_proj` (caught by heuristic "q_proj" substring), but the
  entire fused tensor would be classified as Q, not K or O.
- **RWKV**: Uses completely different naming (time_mix, channel_mix) not covered
  by any pattern. All RWKV tensors would be UNKNOWN and use base quant.
- **MTP (multi-token prediction)**: Pattern `r'mtp\.'` in H group might misclassify
  MTP-specific tensors that should be in other groups.

**Fix:** Add explicit patterns for common fused tensor naming variants. Log
warnings when >5% of tensors are UNKNOWN.

---

### 21. LoRA merge does not validate shape compatibility

**File:** `magicquant/gguf/source.py:917`
**Severity:** MEDIUM

```python
base_f32 = base_f32.reshape(delta.shape) + delta
```

If the LoRA delta shape doesn't match the base tensor shape (e.g., due to
vocabulary resizing or head count mismatch between adapter and base), this
`reshape` will raise a cryptic error. There's no pre-check or helpful error
message.

**Fix:** Validate that `delta.shape` is compatible with `base_f32.size` before
reshape, and raise a clear error if not.

---

### 22. Tokenizer score metadata is always zeros

**File:** `magicquant/gguf/source.py:506`
**Severity:** MEDIUM

```python
scores = [0.0] * (max_id + 1)
```

For BPE tokenizers, scores are not used (they're a SentencePiece concept), so
this is technically correct. But if a model uses Unigram tokenization (detected
on line 469 with `meta["tokenizer.ggml.model"] = "llama"`), the scores should
come from the tokenizer model's vocabulary probabilities. Currently, all scores
are 0.0 even for Unigram tokenizers, which means llama.cpp would not properly
rank tokens during tokenization.

**Fix:** For Unigram tokenizers, extract scores from the tokenizer vocabulary.

---

### 23. PPL parsing regex is fragile

**File:** `magicquant/utils/llamacpp.py:286-309`
**Severity:** MEDIUM

The "last resort" regex on line 306 matches any float on a line containing "PPL":
```python
if "PPL" in line:
    m = re.search(r"(\d+\.\d+)", line)
```

This could match a timestamp, a progress percentage, or any other float on the
same line as "PPL". Since the function iterates lines in reverse, it would match
the *last* line containing "PPL" with any float, which in llama.cpp output is
typically the final summary line, but in edge cases (e.g., verbose output with
per-chunk PPL reports) could match an intermediate value.

**Fix:** Be more specific: require "PPL" immediately before or near the float,
or only use the "Final estimate" pattern and skip the fallback.

---

### 24. `_read_encode_worker` silently pads/trims uncompressed formats

**File:** `magicquant/gguf/writer.py:227-239`
**Severity:** MEDIUM

For F32/F16/BF16 targets, if the encoded blob size doesn't match expected size,
the worker silently pads with zeros or trims. This masks bugs in shape handling --
the tensor would load in llama.cpp but have garbage (zeros or truncated) values.
Only a warning is logged, which is easy to miss in verbose output.

**Fix:** Make the size mismatch a hard error even for uncompressed formats.
A mismatch means the shape calculation or encoding is wrong, and the output
would be silently corrupt.

---

### 25. `read_gguf_file` convenience function doesn't call `open()`

**File:** `magicquant/gguf/reader.py:283-293`
**Severity:** MEDIUM

```python
def read_gguf_file(filepath: str) -> GGUFReader:
    return GGUFReader(filepath)
```

This returns an uninitialized reader (no `open()` call). The caller gets back
an object with empty `metadata` and `tensors` lists. All methods will return
empty results without error.

**Fix:** Call `reader.open()` before returning, or document that the caller must.

---

### 26. `_write_metadata_value` doesn't handle numpy integer/float types

**File:** `magicquant/gguf/writer.py:131-184`
**Severity:** MEDIUM

Metadata values from HF config.json are plain Python types, but values computed
internally (e.g., `int(partial_rotary * head_dim)`) or from numpy operations could
be numpy types (`np.int64`, `np.float32`). These are not `isinstance(value, int)`
in pure Python -- numpy integers inherit from `np.integer`, not `int`. The value
would fall through to the `else` clause on line 183 and be written as a string
(`str(numpy_int)` -> "4096"), which llama.cpp would not parse correctly as a
metadata integer.

**Example:** `meta[f"{arch}.attention.key_length"] = head_dim` where `head_dim`
was computed via numpy operations and is `np.int64(128)`. This gets written as
string "128" instead of uint32 128.

**Fix:** Add isinstance checks for `np.integer` and `np.floating` types, or
convert all metadata values to native Python types before writing.

---

### 27. BF16-to-F16 conversion drops all BF16 source data through F32 intermediate

**File:** `magicquant/gguf/writer.py:349-352`
**Severity:** MEDIUM

```python
if target_ggml_name == "BF16":
    target_ggml_name = "F16"
    target_ggml_id = GGML_TYPE["F16"]
```

When the user requests BF16 for a group, the writer silently converts to F16.
The source tensor is read as F32, then encoded to F16. BF16 has 8-bit exponent
(same range as F32) but 7-bit mantissa. F16 has 5-bit exponent (much smaller
range) and 10-bit mantissa. For values outside F16 range (>65504 or very small
subnormals), the conversion produces Inf or zero. While model weights rarely
exceed this range, embedding layers occasionally do, and the user explicitly
requested BF16 precision.

**Fix:** Document this conversion clearly or implement actual BF16 output now
that llama.cpp supports BF16 in many operations. At minimum, warn the user.

---

### 28. `generate_tiered_models` accesses `entry["ppl"]` which may not exist

**File:** `magicquant/orchestrator.py:594-595`
**Severity:** MEDIUM

```python
ppl=round(entry["ppl"], 4) if "ppl" in entry else None,
measured_loss=round(entry["measured_loss"], 4) if "measured_loss" in entry else None,
```

For prediction-only search, the tiered entries come from `_pick_best_per_tier`
which returns raw config dicts from the evolutionary search. These dicts have
`predicted_loss` but NOT `ppl` or `measured_loss`. The `if "ppl" in entry`
guard handles this correctly for the log statement, but the call to
`generate_hybrid_model` on line 598 uses `config=config` which is correct.
However, the tiered dict format expected by `generate_tiered_models` is
inconsistent between measured and prediction-only paths.

**Fix:** Normalize the tiered dict format to always include the same keys
(with None values for unmeasured fields).

---

### 29. `_save_results` duplicates tiered data under two keys

**File:** `magicquant/orchestrator.py:384-418`
**Severity:** LOW

The saved JSON has both `"tiered_survivors"` and `"tiered"` with nearly identical
data (different subsets of fields). `cmd_generate` reads `"tiered"`, but the
measured search path writes both. If only one is populated due to a code path
change, the other becomes stale.

**Fix:** Use a single canonical key for tiered results.

---

### 30. Evolution unique config check is O(n^2) per generation

**File:** `magicquant/evolution/survival.py:122`
**Severity:** LOW

```python
config_key = str(sorted(winner['config'].items()))
if config_key not in [str(sorted(c['config'].items())) for c in best_configs]:
```

For each winner, this rebuilds string keys for ALL previously discovered configs.
With 50 generations and 3 winners per tier per generation, this could involve
thousands of string comparisons per generation. Use a set for O(1) lookup.

**Fix:** Maintain a `seen_keys: set` and check membership in constant time.

---

### 31. `calculate_expected_size` in naming.py is mathematically a no-op

**File:** `magicquant/utils/naming.py:204-226`
**Severity:** LOW

```python
total_params = base_model_size * (16.0 / base_quant_bits)
return total_params * (base_quant_bits / 16.0)  # Simplified
```

This computes `base_model_size * (16/bits) * (bits/16) = base_model_size`. The
function always returns the input size unchanged. It claims to estimate hybrid
size but the override dict is never used.

**Fix:** Either implement it properly (using override bits per group with
param distribution) or remove it to avoid giving callers false confidence.

---

### 32. Vision tensor skip in SafetensorsSource is prefix-based and may miss some encoders

**File:** `magicquant/gguf/source.py:665`
**Severity:** MEDIUM

```python
if stripped.startswith("model.visual.") or hf_name.startswith("mtp."):
    continue
```

This skips vision tensors using "model.visual." prefix, but different models use
different prefixes for vision encoders:
- LLaVA: `model.vision_tower.`
- Idefics: `model.vision_model.`
- InternVL: `vision_model.`
- Qwen-VL: `visual.`

Some of these would be caught by the regex patterns (leading to V-group
classification), but they would still be *included* in the output GGUF, wasting
space and potentially causing issues with llama.cpp loading.

**Fix:** Use the TensorGroupClassifier to check if a tensor is V-group before
deciding to skip it, rather than hardcoding prefixes.

---

### 33. `open_model_source` directory with both .safetensors and .gguf silently picks safetensors

**File:** `magicquant/gguf/source.py:977-983`
**Severity:** LOW

When a directory contains both formats, safetensors is always preferred. This is
a reasonable default but could surprise users who expect the GGUF to be used.
No warning is printed.

**Fix:** Log a warning when both formats are present in the directory.

---

### 34. Missing "S" group in several hardcoded group lists

**File:** multiple locations
**Severity:** MEDIUM

The "S" group (SSM/linear attention) is dynamically added to the probe list in
`orchestrator.py:151-154` but is missing from:
- `survival.py:45` `DEFAULT_GROUPS` (no SSM tensors in evolutionary search)
- `predictor.py:178-181` `param_dist` (SSM layers get 0.05 default, not based on real distribution)
- `probing.py:252` `_GROUP_SENSITIVITY` (SSM layers get 1.0 default sensitivity)
- `__main__.py:103` `groups` list in `cmd_probe`

This means for hybrid architectures like Qwen3.5 (which has both attention and
SSM layers), the evolutionary search doesn't optimize SSM layer quantization at
all -- they always get the base quant.

**Fix:** Include S (and X, R) in DEFAULT_GROUPS when the model has them, and
add proper param_dist and sensitivity entries for these groups.

---

### 35. Integer metadata values that happen to be 0 are written as UINT32

**File:** `magicquant/gguf/writer.py:138-144`
**Severity:** LOW

```python
elif isinstance(value, int):
    if value < 0:
        f.write(struct.pack("<I", _GGUF_TYPE_INT64))
        f.write(struct.pack("<q", value))
    elif value <= 0xFFFFFFFF:
        f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", value))
```

Boolean `False` in Python is `isinstance(False, int) == True`, but the bool check
comes first (line 132) so this is correctly handled. However, numpy bool values
(`np.bool_(True)`) are NOT `isinstance(v, bool)` in Python, so they would be
written as UINT32 with value 1, which llama.cpp would interpret as an integer,
not a boolean. This could matter for metadata like `general.quantized`.

**Fix:** Add `np.bool_` to the bool check.

---

### 36. `_optimize_asymmetric_scale` offset broadcasting may fail for all-positive sub-blocks

**File:** `magicquant/quant/converters.py:459`
**Severity:** MEDIUM

```python
shifted = sub_blocks + offsets[:, :, None]  # (n_blocks, n_sub, sub_size)
```

When `offsets` (sub_mins) is 0 for all-positive sub-blocks (see issue #3), this
is fine. But when the fix for issue #3 is applied (using actual negative min as
offset), sub-blocks with all positive values will have a negative offset, and
`shifted = val + (-min)` would shift all values up, then `q = round(shifted / scale)`
would quantize the shifted range. The dequantization `deq = q * scale - offset`
would then subtract the offset, recovering approximately the original values. This
is correct arithmetic, but the MSE computation on line 466 compares against the
original `sub_blocks` (not shifted), which is also correct.

No actual bug here after re-analysis, but noting it for completeness.

---

### 37. `tied_word_embeddings` defaults to True

**File:** `magicquant/gguf/source.py:695`
**Severity:** LOW

```python
if config.get("tie_word_embeddings", True):
```

When config.json doesn't specify `tie_word_embeddings`, the code assumes embeddings
are tied and creates a duplicate `output.weight` pointing to `token_embd.weight`.
This is correct for many models (LLaMA, Mistral, Qwen) but wrong for models
that have separate output heads (some GPT-NeoX variants, BLOOM). The result would
be a model with shared embeddings that shouldn't be shared, potentially degrading
output quality subtly.

**Fix:** Default to `False` (safer -- llama.cpp can handle missing output.weight
by tying automatically), or check if `lm_head.weight` exists in any safetensors
file before creating the reference.
