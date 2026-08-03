#!/usr/bin/env python3
"""Rebuild magicquant/data/calib_corpus.txt from eaddario/imatrix-calibration.

Vendored rather than fetched at runtime so a search never depends on the
network, and so the exact bytes that calibrated a published quant stay
reproducible. Re-run this only to deliberately change the corpus; the output
is committed.

Composition is EXPLICIT rather than taken from the upstream `combined_*`
blend: at micro size that blend is dominated by plain prose (a scan of
combined_all_micro found 162 code and 226 math markers in 737 KB, and zero
tool/JSON), which would leave the tool-calling and structured-output paths
essentially uncalibrated.

    45%  multilingual prose  combined_all_micro   18 languages -- vocab
                                                  coverage matters because a
                                                  248k-token vocab quantized
                                                  against English-only
                                                  importance leaves most
                                                  embedding/head rows weighted
                                                  at ~zero
    20%  code                code_micro
    20%  math / reasoning    math_micro
    15%  agentic requests    tools_micro

On that last one: tools_micro is natural-language multi-step tool-USE requests
("convert $500 USD to MXN, then find local transit options there"), NOT JSON
schemas or function-call syntax -- confirmed by reading the block after a
marker scan for '"parameters"'/'"arguments"' came back at ~0. It calibrates
the instruction-following and planning paths, which is worth having; it does
not calibrate literal structured-output tokens. Anyone wanting the latter
should add their own sample of real tool-call transcripts.

Deterministic: fixed proportions, head slices, no RNG. Same inputs -> same
bytes.

Source: https://huggingface.co/datasets/eaddario/imatrix-calibration (MIT),
itself derived from fineweb/fineweb-2, OpenMathInstruct-2, Open-Critic-GPT,
opc-sft-stage2, Magicoder-Evol-Instruct, McEval-Instruct, BitAgent/tool_calling
and Efficient_ToolCalling.
"""
import sys
from pathlib import Path

TARGET_BYTES = 900_000          # ~1750 chunks at ctx 512; runs cap with `chunks`
MIX = [                          # (parquet stem, share of TARGET_BYTES)
    ("combined_all_micro", 0.45),
    ("code_micro",         0.20),
    ("math_micro",         0.20),
    ("tools_micro",        0.15),
]
OUT = Path(__file__).resolve().parent.parent / "magicquant" / "data" / "calib_corpus.txt"
EVAL_CORPUS = Path("/server/ai/wikitext/wikitext-2-raw/wiki.test.raw")


def _load(stem: str) -> str:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download("eaddario/imatrix-calibration", stem + ".parquet",
                        repo_type="dataset")
    return "\n".join(pq.read_table(p).column("content").to_pylist())


def _disjoint_from_eval(text: str) -> tuple[bool, float]:
    """Shared-8-gram rate against the perplexity eval corpus.

    Calibrating on the text a run is scored against would make every measured
    loss optimistic, so this is checked rather than assumed.
    """
    if not EVAL_CORPUS.exists():
        return True, -1.0
    ev = EVAL_CORPUS.read_text(errors="ignore").split()
    ev_grams = {tuple(ev[i:i + 8]) for i in range(0, len(ev) - 8, 3)}
    cal = text.split()
    cal_grams = [tuple(cal[i:i + 8]) for i in range(0, len(cal) - 8, 50)]
    if not cal_grams:
        return True, 0.0
    hits = sum(1 for g in cal_grams if g in ev_grams)
    return hits / len(cal_grams) < 0.001, hits / len(cal_grams)


def main() -> int:
    parts = []
    for stem, share in MIX:
        want = int(TARGET_BYTES * share)
        text = _load(stem)
        if len(text) < want:
            print(f"WARN {stem}: only {len(text)} chars, wanted {want}")
        chunk = text[:want]
        chunk = chunk[:chunk.rfind("\n") + 1] or chunk    # end on a line break
        parts.append(f"{chunk.rstrip()}\n")
        print(f"  {stem:22s} {share:>4.0%}  {len(chunk):>8,} chars")

    corpus = "\n".join(parts)
    ok, rate = _disjoint_from_eval(corpus)
    print(f"\nshared 8-grams with the eval corpus: {rate:.5%}"
          f"  -> {'DISJOINT' if ok else 'OVERLAP, refusing'}")
    if not ok:
        print("Calibrating on the eval text would make every measured loss "
              "optimistic. Refusing to write.", file=sys.stderr)
        return 1

    OUT.write_text(corpus)
    print(f"\nwrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
