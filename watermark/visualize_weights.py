"""
What is the classifier actually looking for?

Three views of the same linear model:
  1. the learned weight vector across frequency -- where it looks, and whether
     it seeks peaks (positive) or their absence (negative)
  2. each file's mean fakeprint on the same axis
  3. the CUMULATIVE logit as frequency increases -- shows where the decision is
     actually made rather than where the features happen to be large

The cumulative view is the informative one: a linear model's verdict is a
running sum, so watching it accumulate says more than any single band.

Usage:
    python visualize_weights.py -o out.png "Label=file.wav" ...
"""

import argparse
import os
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector

INK = "#e9e9ea"
MUTED = "#9a9aa0"
BG = "#141416"
GRID = "#31313a"
# First three slots of the validated categorical order (all-pairs safe).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
WEIGHT_POS = "#e34948"
WEIGHT_NEG = "#4da3ff"


def smooth(x, k=25):
    return np.convolve(x, np.ones(k) / k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='each as "Label=path.wav"')
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    w = np.load(hf_hub_download("lofcz/ai-music-detector", filename="weights.npz"))
    W = w["weights"].ravel()
    bias = float(w["bias"][0])

    det = FakeprintDetector(args.model)
    items = []
    for spec in args.panels:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec), spec
        fp, fr = det.fakeprint(path)
        p, _ = det.predict(path)
        items.append(dict(label=label, fp=fp, fr=fr, p=p))
    fr = items[0]["fr"] / 1000

    fig = plt.figure(figsize=(13.5, 10.2), facecolor=BG)
    gs = fig.add_gridspec(3, 1, hspace=0.42, left=0.075, right=0.985,
                          top=0.895, bottom=0.062)

    # 1 -- the weight vector
    ax = fig.add_subplot(gs[0], facecolor=BG)
    sw = smooth(W)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.fill_between(fr, 0, sw, where=(sw > 0), color=WEIGHT_POS, alpha=0.75,
                    linewidth=0, label="positive → evidence for AI")
    ax.fill_between(fr, 0, sw, where=(sw <= 0), color=WEIGHT_NEG, alpha=0.75,
                    linewidth=0, label="negative → evidence for real")
    ax.set_title("1. What the model looks for  (learned weights, smoothed)",
                 color=INK, fontsize=11.5, loc="left", pad=8)
    ax.set_ylabel("weight", color=MUTED, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    for t in leg.get_texts():
        t.set_color(MUTED)

    # 2 -- each file's fakeprint
    ax2 = fig.add_subplot(gs[1], facecolor=BG)
    for i, it in enumerate(items):
        ax2.plot(fr, smooth(it["fp"], 41), lw=1.4, color=SERIES[i % 3],
                 label=f"{it['label']}  (p={it['p']:.3f})")
    ax2.set_title("2. Spectral residual of each file  (smoothed)",
                  color=INK, fontsize=11.5, loc="left", pad=8)
    ax2.set_ylabel("residual", color=MUTED, fontsize=9)
    leg = ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(MUTED)

    # 3 -- cumulative logit: where the verdict is actually decided
    ax3 = fig.add_subplot(gs[2], facecolor=BG)
    ax3.axhline(0, color=GRID, lw=0.8, ls="--")
    for i, it in enumerate(items):
        run = bias + np.cumsum(W * it["fp"])
        ax3.plot(fr, run, lw=1.7, color=SERIES[i % 3], label=it["label"])
        ax3.plot([fr[-1]], [run[-1]], "o", ms=5, color=SERIES[i % 3])
        ax3.annotate(f"{run[-1]:+.1f}", (fr[-1], run[-1]),
                     textcoords="offset points", xytext=(-6, 6),
                     color=SERIES[i % 3], fontsize=9, ha="right")
    ax3.text(1.03, 1.2, "above 0 → AI", color=MUTED, fontsize=8.5)
    ax3.text(1.03, -3.2, "below 0 → real", color=MUTED, fontsize=8.5)
    ax3.set_title("3. Running logit as frequency increases  "
                  "(this is the decision being made)",
                  color=INK, fontsize=11.5, loc="left", pad=8)
    ax3.set_ylabel("cumulative logit", color=MUTED, fontsize=9)
    ax3.set_xlabel("kHz", color=MUTED, fontsize=9)
    leg = ax3.legend(frameon=False, fontsize=8.5, loc="lower left")
    for t in leg.get_texts():
        t.set_color(MUTED)

    for a in (ax, ax2, ax3):
        a.set_xlim(1, 8)
        a.tick_params(colors=MUTED, labelsize=8.5, length=3)
        a.grid(True, color=GRID, lw=0.5, alpha=0.5)
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_color(GRID)

    fig.text(0.075, 0.972, "Inside the AI-music classifier", color=INK,
             fontsize=15, weight="600", ha="left", va="top")
    fig.text(0.075, 0.936,
             f"Logistic regression, 3585 features (1–8 kHz), bias {bias:+.2f}. "
             "Verdict = bias + Σ weight×residual.",
             color=MUTED, fontsize=9.5, ha="left", va="top")

    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.output}")
    for it in items:
        print(f"  {it['label']:<34} p={it['p']:.4f}  "
              f"final logit={bias + (W * it['fp']).sum():+.2f}")


if __name__ == "__main__":
    main()
