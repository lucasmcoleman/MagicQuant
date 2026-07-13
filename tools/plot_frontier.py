#!/usr/bin/env python
"""Plot the v2 quality-size frontier against measured points.

Usage:
    python tools/plot_frontier.py FRONTIER_JSON OUT_PNG \
        [--v1-results SEARCH_RESULTS_JSON] [--baseline-ppl X] \
        [--extra-measured V2_RESULTS_JSON ...] [--title T]

Inputs:
    FRONTIER_JSON       magicquant --algo v2 output (frontier.json): predicted
                        frontier trace + this run's measured anchors.
    --extra-measured    additional v2_results.json files (e.g. runs at other
                        budgets) whose verified anchors join the measured set.
    --v1-results        a v1 search_results.json; its measurements (or, with
                        --v1-full, a JSON of {label: {gb, ppl}} re-measured at
                        full corpus) are drawn as the comparison series.

The y axis is relative PPL loss vs baseline when a baseline is known for every
series (dimensionless — comparable across measurement conditions); otherwise
raw PPL, and mixing conditions is refused rather than silently plotted.
"""

import argparse
import json
from pathlib import Path

# Palette: dataviz reference instance (validated 2026-07-12: CVD dE 73.6,
# aqua below 3:1 contrast -> relief via direct labels, applied below).
C_V2 = "#2a78d6"       # series 1 (blue): v2 predicted frontier + anchors
C_V1 = "#1baf7a"       # series 2 (aqua): v1 measured points
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e4e3df"
C_SURFACE = "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("frontier_json")
    ap.add_argument("out_png")
    ap.add_argument("--v1-results", default=None)
    ap.add_argument("--v1-full", default=None,
                    help="JSON {label: {gb, ppl}} of v1 winners re-measured "
                         "at full corpus (preferred over --v1-results)")
    ap.add_argument("--extra-measured", nargs="*", default=[])
    ap.add_argument("--baseline-ppl", type=float, default=None)
    ap.add_argument("--title", default="Quality-size frontier: v1 vs v2")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frontier = json.loads(Path(args.frontier_json).read_text())
    pred = [(p["gb"], p["loss"]) for p in frontier["points"]]

    measured_v2 = []
    for m in frontier.get("measured", []):
        if m.get("rel_loss") is not None:
            measured_v2.append((m["gb"], m["rel_loss"], m.get("tag", "")))
    for extra in args.extra_measured:
        r = json.loads(Path(extra).read_text())
        for a in r.get("anchors", []):
            if a.get("measured_rel_loss") is not None:
                gb = (a.get("actual_bytes") or a["predicted_bytes"]) / 1024**3
                measured_v2.append((gb, a["measured_rel_loss"], a["tag"]))

    measured_v1 = []
    if args.v1_full:
        data = json.loads(Path(args.v1_full).read_text())
        base = args.baseline_ppl
        if base is None:
            raise SystemExit("--v1-full requires --baseline-ppl")
        for label, e in data.items():
            measured_v1.append((e["gb"], (e["ppl"] - base) / base, label))
    elif args.v1_results:
        r = json.loads(Path(args.v1_results).read_text())
        for key, m in r.get("measurements", {}).items():
            if m.get("measured_loss") is not None and m.get("size_gb"):
                measured_v1.append((m["size_gb"], m["measured_loss"], ""))

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)

    # Predicted frontier: thin line, no markers (it has hundreds of steps).
    # NOTE: predicted loss is in surrogate units — scale to the measured
    # rel-loss axis via the anchor that has both, so the curve is honest
    # about being a calibrated prediction.
    if pred and measured_v2:
        gb0, rel0, _ = measured_v2[0]
        nearest = min(pred, key=lambda p: abs(p[0] - gb0))
        scale = rel0 / nearest[1] if nearest[1] > 0 else None
        if scale is not None:
            xs = [p[0] for p in pred]
            ys = [p[1] * scale for p in pred]
            ax.plot(xs, ys, color=C_V2, lw=1.4, alpha=0.55, zorder=2,
                    label="v2 predicted frontier (anchor-calibrated)")

    if measured_v2:
        xs, ys = [m[0] for m in measured_v2], [m[1] for m in measured_v2]
        ax.scatter(xs, ys, s=64, color=C_V2, zorder=4, label="v2 measured")
        for gb, rel, tag in measured_v2:
            ax.annotate(f"{rel*100:.1f}%", (gb, rel), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=C_TEXT2)

    if measured_v1:
        xs, ys = [m[0] for m in measured_v1], [m[1] for m in measured_v1]
        ax.scatter(xs, ys, s=64, color=C_V1, marker="s", zorder=4,
                   label="v1 measured (tier winners)")
        for gb, rel, label in measured_v1:
            txt = f"{label} {rel*100:.1f}%".strip()
            ax.annotate(txt, (gb, rel), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=C_TEXT2)

    ax.set_xlabel("model size (GiB)", color=C_TEXT)
    ax.set_ylabel("relative PPL loss vs BF16 baseline", color=C_TEXT)
    ax.set_title(args.title, color=C_TEXT, fontsize=11)
    ax.grid(True, color=C_GRID, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8)
    leg = ax.legend(fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(C_TEXT)

    fig.tight_layout()
    fig.savefig(args.out_png, facecolor=C_SURFACE)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
