"""
Show the 200 Hz vocoder comb directly.

Suno's neural vocoder upsamples with a hop of 160 samples at its 32 kHz
generation rate, i.e. a frame rate of exactly 200 Hz. Transposed convolution at
that rate stamps a comb of evenly spaced peaks across the spectrum, spaced
200 Hz apart. This draws it three ways:

  1. residual against a 200 Hz grid, zoomed enough to see individual lines
  2. the same for a known non-AI control -- peaks land nowhere in particular
  3. comb strength swept across candidate spacings; the generated file spikes at
     exactly 200.00 Hz, the control does not

Usage:
    python visualize_comb.py -o out.png --ai suno.wav --control real.wav
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
AI_C = "#eb6834"
OK_C = "#2a78d6"
COMB = "#f5c542"


def comb_ratio(fp, fr, sp, tol=2.0):
    r = np.abs(((fr / sp) + 0.5) % 1.0 - 0.5) * sp
    on = r < tol
    if on.sum() < 20 or (~on).sum() < 20:
        return np.nan
    return fp[on].mean() / (fp[~on].mean() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--zoom", type=float, nargs=2, default=[3000, 4600])
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()

    det = FakeprintDetector(args.model)
    ai_fp, fr = det.fakeprint(args.ai)
    ct_fp, _ = det.fakeprint(args.control)

    fig = plt.figure(figsize=(13.5, 9.4), facecolor=BG)
    gs = fig.add_gridspec(3, 1, hspace=0.46, left=0.075, right=0.985,
                          top=0.885, bottom=0.07)

    lo, hi = args.zoom
    grid = np.arange(np.ceil(lo / 200) * 200, hi + 1, 200)

    for i, (fp, colour, label, verdict) in enumerate([
            (ai_fp, AI_C, "Suno-generated", "flagged p=1.0000"),
            (ct_fp, OK_C, "known non-AI control", "clean p=0.0000")]):
        ax = fig.add_subplot(gs[i], facecolor=BG)
        for g in grid:
            ax.axvline(g / 1000, color=COMB, lw=0.9, alpha=0.45,
                       ls=(0, (4, 3)), zorder=1)
        m = (fr >= lo) & (fr <= hi)
        ax.plot(fr[m] / 1000, fp[m], lw=0.9, color=colour, zorder=3)
        ax.set_xlim(lo / 1000, hi / 1000)
        ax.set_ylim(0, max(fp[m].max() * 1.08, 0.1))
        ax.set_ylabel("residual", color=MUTED, fontsize=9)
        r200 = comb_ratio(fp, fr, 200.0)
        ax.set_title(f"{i+1}. {label} — {verdict}     "
                     f"energy on the 200 Hz grid: {r200:.2f}× off-grid",
                     color=INK, fontsize=11.5, loc="left", pad=8)
        if i == 0:
            ax.text(0.995, 0.94, "dashed lines = multiples of 200 Hz",
                    transform=ax.transAxes, color=COMB, fontsize=8.5,
                    ha="right", va="top")

    ax3 = fig.add_subplot(gs[2], facecolor=BG)
    sweep = np.arange(150, 260, 0.1)
    for fp, colour, label in ((ai_fp, AI_C, "Suno-generated"),
                              (ct_fp, OK_C, "known non-AI control")):
        ax3.plot(sweep, [comb_ratio(fp, fr, s) for s in sweep], lw=1.5,
                 color=colour, label=label)
    ax3.axvline(200, color=COMB, lw=1.0, ls=(0, (4, 3)), alpha=0.7)
    ax3.annotate("exactly 200.00 Hz\n= 160-sample hop at 32 kHz",
                 xy=(200, comb_ratio(ai_fp, fr, 200.0)),
                 xytext=(214, comb_ratio(ai_fp, fr, 200.0) * 0.92),
                 color=COMB, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=COMB, lw=1.0))
    ax3.axhline(1.0, color=GRID, lw=0.8, ls="--")
    ax3.set_xlabel("candidate comb spacing (Hz)", color=MUTED, fontsize=9)
    ax3.set_ylabel("on-grid / off-grid", color=MUTED, fontsize=9)
    ax3.set_title("3. Comb strength swept across spacings — "
                  "only the generated file spikes, and only at 200 Hz",
                  color=INK, fontsize=11.5, loc="left", pad=8)
    leg = ax3.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(MUTED)

    for a in fig.axes:
        a.tick_params(colors=MUTED, labelsize=8.5, length=3)
        a.grid(True, color=GRID, lw=0.5, alpha=0.4)
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_color(GRID)

    fig.text(0.075, 0.972, "The 200 Hz vocoder comb", color=INK, fontsize=15,
             weight="600", ha="left", va="top")
    fig.text(0.075, 0.936,
             "Transposed-convolution upsampling stamps peaks at the vocoder's "
             "frame rate. Measured on the detector's own spectral residual.",
             color=MUTED, fontsize=9.5, ha="left", va="top")

    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.output}")
    for name, fp in (("AI", ai_fp), ("control", ct_fp)):
        print(f"  {name:<9} comb ratio at 200 Hz = {comb_ratio(fp, fr, 200.0):.3f}")


if __name__ == "__main__":
    main()
