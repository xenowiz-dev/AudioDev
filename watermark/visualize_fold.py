"""
Draw the frame-locked waveform pattern directly.

Chop the emphasised residual into blocks exactly one vocoder frame long and
average them. Music is not locked to that grid so it averages away; an
overlap-add synthesis artefact is locked, so it survives and becomes visible as
a repeating shape a few milliseconds long.

Shown against the same file folded at a deliberately wrong period, which is the
null: whatever remains there is just the noise floor.

Usage:
    python visualize_fold.py -o out.png --rate 200 "Label=file.wav" ...
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#e9e9ea"
MUTED = "#9a9aa0"
BG = "#141416"
GRID = "#31313a"
SIG = "#eb6834"
NULL = "#5a5a66"


def profile(x, period, seconds_limit=None):
    x = np.diff(x, n=2)
    n = len(x) // period
    if n < 64:
        return None, 0
    b = x[: n * period].reshape(n, period)
    b = b - b.mean(axis=1, keepdims=True)
    m = b.mean(axis=0)
    floor = np.sqrt(b.var() / n)
    return m, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='each as "Label=path.wav"')
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    items = []
    for spec in args.panels:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec), spec
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        x = x[: int(args.seconds * sr)].mean(axis=1)
        p = sr / args.rate
        if abs(p - round(p)) > 1e-6:
            print(f"skip {label}: {args.rate:g} Hz not integer at {sr} Hz")
            continue
        p = int(round(p))
        m, floor = profile(x, p)
        if m is None:
            continue
        # null: a period nothing would use
        mn, _ = profile(x, p + 3)
        items.append(dict(label=label, m=m, floor=floor, mn=mn, p=p, sr=sr))

    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.5 * n + 1.2),
                             facecolor=BG, squeeze=False)
    for ax, it in zip(axes[:, 0], items):
        ax.set_facecolor(BG)
        t = np.arange(it["p"]) / it["sr"] * 1000
        tn = np.arange(len(it["mn"])) / it["sr"] * 1000
        ax.plot(tn, it["mn"], lw=0.9, color=NULL,
                label=f"wrong period ({it['p']+3} samples) — the null")
        ax.plot(t, it["m"], lw=1.4, color=SIG,
                label=f"frame period ({it['p']} samples)")
        ax.axhline(it["floor"], color=GRID, lw=0.8, ls="--")
        ax.axhline(-it["floor"], color=GRID, lw=0.8, ls="--")
        ax.text(0.996, 0.06, "dashed = 1/√N noise floor", transform=ax.transAxes,
                color=MUTED, fontsize=8, ha="right")
        pk = np.abs(it["m"]).max() / (it["floor"] + 1e-30)
        ax.set_title(f"{it['label']}   —   peak is {pk:.0f}× the noise floor",
                     color=INK, fontsize=11, loc="left", pad=6)
        ax.set_ylabel("residual", color=MUTED, fontsize=9)
        ax.set_xlim(0, t[-1])
        leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for tx in leg.get_texts():
            tx.set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=3)
        ax.grid(True, color=GRID, lw=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[-1, 0].set_xlabel("milliseconds within one frame", color=MUTED,
                           fontsize=9)

    fig.suptitle(f"Frame-locked waveform structure at {args.rate:g} Hz",
                 color=INK, fontsize=14.5, weight="600", x=0.055, ha="left",
                 y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    sys.exit(main())
