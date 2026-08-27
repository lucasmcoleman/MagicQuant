# Upstream Architecture Automation

**Date:** 2026-08-27
**Status:** Approved (pending spec review)

## Problem

New model architectures land in llama.cpp continuously, and MagicQuant currently
requires a human (via an interactive session) to notice, evaluate, and manually
trigger the existing sync tooling. The immediate trigger is
[llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742), which adds
support for Qwen3.8-Flash-Next (`qwen4_exp` / `Qwen4ExpForConditionalGeneration`,
177B total params, 512-expert MoE) — open and unmerged as of this writing. The
existing `upstream-watch.yml` / `check_upstream_drift.py` pipeline cannot see it at
all yet: its trust anchor is the installed `gguf` PyPI package, which only gains a
new architecture constant after the corresponding llama.cpp PR is merged **and**
released to PyPI — a lag on top of the PR's own review cycle.

Goal: close two gaps without weakening the existing safety design.

1. **Detection latency** — know when a *specific, already-identified* upstream PR
   merges, without waiting for the package-based sweep to notice weeks later.
2. **Human-in-the-loop friction** — get pinged when something is actionable,
   instead of periodically asking a fresh Claude session to check.

Non-goal: fully automating the actual conversion logic for novel architectures.
`draft_upstream_sync.py` already documents, by design, that it never derives
value-level transforms (RMSNorm shifts, permutes, expert reshaping, and — for this
model specifically — hyper-connections, an n-gram hash-table embedding, and a new
sparse-attention KV-cache type). That constraint is not relaxed by this work; the
validation gate below exists precisely so automation never merges past it silently.

## Existing pipeline (unchanged, for context)

- `tools/check_upstream_drift.py`: compares `gguf.constants` (installed package)
  against MagicQuant's own `arch_map`/scheme registry. Weekly, via
  `upstream-watch.yml`. Opens/updates a GitHub tracking issue on drift.
- `tools/draft_upstream_sync.py`: given a drifted architecture, mechanically
  derives candidate `arch_map` lines and `_HF_TO_GGUF_PATTERNS` entries from
  `gguf.tensor_mapping`, and writes a report with an explicit manual checklist for
  everything it can't derive. Weekly, via `upstream-sync-pr.yml`, force-updating one
  rolling PR (`bot/upstream-sync`). Merge is currently always a human decision.

Both stay as-is. This spec adds a second detection path and a validation+auto-merge
gate that sits between drafting and merging.

## Components

### 1. Tracked-PR registry (`tools/tracked_prs.json`)

A small, human-maintained list of specific llama.cpp PRs worth watching ahead of
the package-based sweep:

```json
[
  {
    "pr": 27742,
    "repo": "ggml-org/llama.cpp",
    "note": "Qwen3.8-Flash-Next (qwen4_exp)",
    "example_model": "Qwen/Qwen3.8-Flash-Next",
    "added": "2026-08-27"
  }
]
```

- `example_model` (optional): an HF repo whose `config.json` can seed a synthetic
  fixture for the validation gate (see Component 3) — no full weights needed, just
  the small `config.json`.
- Entries are added by hand (or by an agent, on request) when a relevant PR is
  spotted — this is deliberately not auto-discovered. Auto-discovering "relevant"
  PRs from llama.cpp's full PR stream is a precision problem this spec doesn't
  attempt to solve; a human/agent already has to notice the model release to know
  it's worth tracking in the first place.
- Entries are removed once the PR merges and the corresponding architecture is
  fully synced (mapped + validated + merged) or explicitly abandoned.

### 2. `tools/check_tracked_prs.py` + `upstream-pr-watch.yml`

New script: for each entry in `tracked_prs.json`, query the GitHub API
(`GET /repos/{repo}/pulls/{pr}`) for merge state. Reports newly-merged entries
(diffed against a small `tools/tracked_prs_state.json` baseline, same pattern as
`upstream_baseline.json`).

New workflow, daily cadence (tighter than the weekly package sweep, since this is
cheap — a handful of API calls):

- On a newly-merged tracked PR: comment on (or open) the same tracking issue
  `upstream-watch.yml` uses, noting the merge and that package-based drift
  detection is still pending a `gguf` release. This is an **informational**
  signal only — it does not by itself trigger drafting, since the `gguf` package
  may not have the constant yet even post-merge.
- Also immediately attempts `check_upstream_drift.py` (cheap, already fast) in case
  the package happened to catch up already — normal path from there if so.

### 3. Validation gate (new: `tools/validate_arch_sync.py`)

Runs as a new job in `upstream-sync-pr.yml`, after `draft_upstream_sync.py` updates
the bot PR, before any merge decision:

1. **Test suite** — apply the candidate `arch_map`/pattern diff on the bot branch,
   run the full `pytest` suite. Must pass.
2. **Structural round-trip** — build a synthetic tiny fixture: a safetensors dir
   with the target arch's real `config.json` (from `tracked_prs.json`'s
   `example_model`, or the drifted arch's associated model if known) but tensors
   shrunk to a minimal layer count and random-filled data (new:
   `tools/gen_arch_fixture.py`). Run `create_hybrid_gguf` against it end-to-end —
   confirms the writer doesn't crash and every tensor round-trips to a valid GGUF.
   This catches name-mapping and shape mistakes; it cannot catch value-transform
   correctness (by design — see Non-goal above).
3. **Inference smoke check** — *only if* llama.cpp already supports the arch.
   `binary_supports_arch` (`utils/llamacpp.py`) byte-scans a local binary/library
   path; it has no notion of a Docker image reference, so this needs a small new
   adapter, not a direct reuse: extract the binary/`libllama` out of the pulled
   `ghcr.io/ggml-org/llama.cpp:server-rocm` image (e.g. `docker create` + `docker
   cp`, or `docker run --entrypoint cat`) into a local file, then run the existing
   byte-scan against that extracted file.

   If supported, load the fixture GGUF and run one plain-PPL chunk (not KL — see
   below) against it. Detection logic is **new, not a reuse of
   `probing.py`'s existing broken-probe check**: that check (NaN, or PPL over
   `BROKEN_PROBE_RATIO` × baseline) only fires in `probing.py`'s plain-PPL path,
   gated on `not self.kl_base_logits_path`, and it compares against a
   `baseline_ppl` from a previously-measured real reference build — there is no
   such baseline for a freshly-synthesized, random-filled, shape-shrunk fixture
   that corresponds to no real checkpoint. Instead: within the same gate run,
   also build a trivial full-precision (F16-passthrough, no quantization) GGUF
   from the *same* fixture, measure its PPL as a fresh self-baseline, then apply
   the same NaN / PPL-blowup-ratio logic against that self-baseline rather than
   a stored historical one.
4. **Decision**: auto-merge the bot PR only if steps 1–3 all actually ran and
   passed. There is no branch where step 3 being skipped or unavailable still
   permits auto-merge — "structurally valid" (steps 1–2) is necessary but not
   sufficient, since it cannot catch the value-transform errors that caused the
   qwen3_5 incident, which is exactly what step 3 exists to catch. If llama.cpp
   doesn't support the arch yet (step 3 can't run) or any step fails, leave the
   PR open, apply a `needs-human` label, and stop. In this branch — the
   assignment/mention that notifies Lucas (Component 4) is applied here, at the
   point the `needs-human` label is set, and also when a merge-by-bot succeeds.

Step 3 needs this box's ROCm hardware and Docker image — GitHub-hosted runners
have neither. **This requires registering a self-hosted GitHub Actions runner on
this machine**, scoped to the `MagicQuant` repo and this job only. New operational
setup, not currently present (confirmed: every existing workflow runs on
`ubuntu-latest`).

Until that runner exists, step 3 cannot run at all — and auto-merge does **not**
proceed on steps 1–2 alone. Structural round-trip passing is necessary but not
sufficient: it catches name/shape mistakes, not the value-transform correctness
that produced the qwen3_5 incident, so treating "structurally valid" as "safe to
merge unattended" would reintroduce exactly that failure class. Without the
runner, every drift lands in the existing human-merge path — auto-merge is
effectively inert until the runner is registered. This is a prerequisite, not a
nice-to-have; the implementation plan should stand up the runner before or
alongside the gate itself, not after.

For qwen4_exp specifically: step 2 will very likely fail or produce a structurally
valid-but-wrong GGUF, because the hyper-connections/n-gram-embedding/sparse-attention
value transforms aren't derivable by `draft_upstream_sync.py` at all — the candidate
diff it drafts won't include them. Expected outcome: gate fails, PR stays open,
`needs-human` — which is correct: the real conversion code still needs to be written
by hand (or by an agent, in a session, once notified).

### 4. Notification

Two channels, since neither alone is both durable and truly push:

- **GitHub-native** (already free): the bot PR/issue is assigned to Lucas /
  mentions him when merged-by-bot or labeled `needs-human`. GitHub's own mobile
  app or email delivers this without any new build, independent of any Claude
  session's lifetime.
- **Claude-side push**: a cloud-scheduled routine, set up via the `schedule` skill
  (durable — unlike `CronCreate`, which is session-scoped and expires after 7
  days), checks daily via the GitHub API for anything newly merged-by-bot or
  newly labeled `needs-human` on this repo since its last check, and calls
  `PushNotification` only when there's something new. This routine is operational
  setup outside the MagicQuant repo itself (no repo code — a scheduled
  agent/prompt), tracked here for completeness.

## Out of scope

- Auto-discovering which upstream PRs are worth tracking (registry stays
  hand-maintained).
- Writing actual value-transform conversion code for novel architectures — always
  a human/agent task, gated by the validation gate never merging past it silently.
- Downloading full model weights (354GB for qwen4_exp) as part of validation —
  the synthetic fixture (Component 3, step 2) is deliberately shape-only.
- Retiring the existing weekly package-based sweep — it stays as the fallback
  detection path for architectures nobody thought to track ahead of time.

## Suggested implementation phasing

Components differ enough in kind (Python tooling vs. GH Actions vs. local
systems/infra vs. an out-of-repo scheduled routine) that they should be planned as
separate phases rather than one flat task list:

1. Tracked-PR registry + `check_tracked_prs.py` + `upstream-pr-watch.yml`
   (Components 1–2) — self-contained, no dependency on the rest.
2. Self-hosted runner registration (part of Component 3) — infra prerequisite;
   do this before or alongside the gate logic, never after.
3. `gen_arch_fixture.py` + `validate_arch_sync.py` + the gate wiring into
   `upstream-sync-pr.yml` (Component 3) — depends on the runner existing for its
   inference-smoke-check path to ever actually run.
4. GitHub-native assignment/mention wiring + the cloud-scheduled push-notification
   routine (Component 4) — depends on 1–3 existing to have anything to notify
   about.

## Open items for spec review

- Self-hosted runner setup (Component 3) is new infrastructure on this box; flagged
  above but the actual registration/hardening is an implementation-time task, not
  fully specified here (standard GitHub self-hosted runner install + a
  repo-scoped registration token).
- `tools/gen_arch_fixture.py`'s "shrink layer count, random-fill data" fixture
  generator needs per-arch-family awareness of which config keys control layer
  count/expert count/vocab size safely enough to shrink without breaking the
  converter's own structural assumptions — scope this at plan time against 2-3
  concrete example architectures (qwen4_exp among them).
