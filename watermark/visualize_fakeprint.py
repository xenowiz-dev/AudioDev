"""
Plot the vocoder "fakeprint" so a human can judge it, instead of trusting a
black-box probability.

The claim behind the detector (ISMIR 2025, "A Fourier Explanation of AI-Music
Artifacts"): neural vocoders upsample with transposed convolutions, which stamp
EVENLY SPACED peaks across the spectrum. Subtract a local minimum-filter floor
and those peaks stand out as a regular comb. Real recordings have irregular
spectral structure with no such periodicity.

So the thing to look for is not "spiky" -- real music is spiky too -- but
"spiky at a constant spacing". The right-hand panel makes that objective: it is
the autocorrelation of the residual, where a comb produces a tall isolated peak
at the comb spacing.

Usage:
    python visualize_fakeprint.py -o out.png "Label=file.wav" ...
"""

import argparse
import os
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector

INK = "#e9e9ea"
MUTED = "#9a9aa0"
BG = "#141416"
SERIES = "#4da3ff"
ACCENT = "#f5c542"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='each as "Label=path.wav"')
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    ap.add_argument("--title", default="Vocoder fakeprint")
    ap.add_argument("--subtitle", default="")
    args = ap.parse_args()

    det = FakeprintDetector(args.model)

    items = []
    for spec in args.panels:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec), spec
        fp, freqs = det.fakeprint(path)
        p, _ = det.predict(path)

        # Autocorrelation of the mean-removed residual: a regular comb shows a
        # tall peak away from lag 0; irregular structure decays to noise.
        r = fp - fp.mean()
        ac = np.correlate(r, r, mode="full")[len(r) - 1:]
        ac = ac / (ac[0] + 1e-12)
        lags = np.arange(len(ac))
        search = slice(5, min(600, len(ac)))
        best = int(np.argmax(ac[search])) + search.start
        items.append(dict(label=label, fp=fp, freqs=freqs, p=p, ac=ac,
                          lags=lags, best=best, peak=float(ac[best])))

    n = len(items)
    panel_h = 2.15
    header_in = 1.15 if args.subtitle else 0.8
    fig_h = header_in + panel_h * n + 0.55
    fig = plt.figure(figsize=(13.5, fig_h), facecolor=BG)
    gs = GridSpec(n, 2, width_ratios=[3, 1], hspace=0.42, wspace=0.16,
                  left=0.062, right=0.985,
                  top=1 - header_in / fig_h, bottom=0.55 / fig_h)

    for i, it in enumerate(items):
        ax = fig.add_subplot(gs[i, 0], facecolor=BG)
        ax.plot(it["freqs"] / 1000, it["fp"], lw=0.45, color=SERIES)
        ax.set_xlim(1, 8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("residual", color=MUTED, fontsize=8.5)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        for s in ax.spines.values():
            s.set_color("#3a3a40")
        verdict = "AI-GENERATED" if it["p"] >= 0.5 else "real"
        col = ACCENT if it["p"] >= 0.5 else "#5ecf8f"
        ax.text(0.004, 1.14, it["label"], transform=ax.transAxes, color=INK,
                fontsize=10.5, va="top", ha="left", weight="600")
        ax.text(0.998, 1.14, f"model says: {verdict}  (p={it['p']:.4f})",
                transform=ax.transAxes, color=col, fontsize=9,
                va="top", ha="right")
        if i == n - 1:
            ax.set_xlabel("kHz", color=MUTED, fontsize=8.5)

        ax2 = fig.add_subplot(gs[i, 1], facecolor=BG)
        ax2.plot(it["lags"][:600], it["ac"][:600], lw=0.7, color=SERIES)
        ax2.axhline(0, color="#3a3a40", lw=0.6)
        ax2.set_xlim(0, 600)
        ax2.set_ylim(-0.35, 1.02)
        ax2.tick_params(colors=MUTED, labelsize=7.5, length=3)
        for s in ax2.spines.values():
            s.set_color("#3a3a40")
        ax2.plot([it["best"]], [it["peak"]], "o", ms=4, color=ACCENT)
        ax2.text(0.97, 0.93, f"max off-zero: {it['peak']:.3f}",
                 transform=ax2.transAxes, color=MUTED, fontsize=8,
                 ha="right", va="top")
        ax2.set_title("self-similarity of residual", color=MUTED, fontsize=8.5,
                      pad=4)
        if i == n - 1:
            ax2.set_xlabel("lag (freq bins)", color=MUTED, fontsize=8.5)

    fig.text(0.062, 1 - 0.30 / fig_h, args.title, color=INK, fontsize=14.5,
             weight="600", ha="left", va="top")
    if args.subtitle:
        fig.text(0.062, 1 - 0.66 / fig_h, args.subtitle, color=MUTED,
                 fontsize=9.5, ha="left", va="top")

    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.output}")
    for it in items:
        print(f"  {it['label']:<34} p={it['p']:.4f}  "
              f"comb self-similarity={it['peak']:.3f} at lag {it['best']}")


if __name__ == "__main__":
    main()
