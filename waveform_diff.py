"""
Waveform comparison and time-frequency difference maps.

Two views of how a set of derived tracks depart from a reference:

  1. WAVEFORMS  min/max envelope per pixel column, so a 30 s file is drawn
     honestly rather than decimated into a misleading squiggle
  2. DIFFERENCE MAPS  per-bin spectrogram difference in dB against the
     reference, on a diverging scale: warm = the derived track has more energy
     there, cool = the reference does, neutral = they match

Colour follows the diverging rule -- two hues with a neutral midpoint, never a
hue at zero -- because the value being encoded is polarity around a meaningful
zero (no difference), not magnitude.

Usage:
    python waveform_diff.py --ref src.wav -o out.png "Label=a.wav" "Label=b.wav"
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = "#e9e9ea"
MUTED = "#9a9aa0"
BG = "#141416"
GRID = "#31313a"
REF_C = "#9a9aa0"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# Diverging: cool -> neutral -> warm. Neutral is a dark grey that reads as
# "no difference" against the panel, not as a third colour.
DIVERGING = LinearSegmentedColormap.from_list(
    "diff", ["#1b4f8f", "#2a78d6", "#3a3a42", "#2b2b31", "#3a3a42",
             "#eb6834", "#8f3418"][::1], N=512)


def envelope(x, width):
    """Min/max per pixel column -- the honest way to draw a long waveform."""
    n = len(x)
    step = max(1, n // width)
    m = (n // step) * step
    b = x[:m].reshape(-1, step)
    return b.min(axis=1), b.max(axis=1)


def spec_db(x, n_fft=2048, hop=512):
    win = np.hanning(n_fft)
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * win, axis=1)).T
    return 20 * np.log10(S + 1e-8)


def band_db(x, sr, n_bands=56, fmax=12000, n_fft=2048, hop=512, smooth=5):
    """Coarse log-spaced band energies over time.

    A per-bin difference between two *different performances* is dominated by
    phase and micro-timing mismatch -- it looks like noise regardless of how
    similar the music is. Grouping into bands and smoothing in time reveals the
    timbral difference that is actually of interest.
    """
    S = spec_db(x, n_fft, hop)
    freqs = np.linspace(0, sr / 2, S.shape[0])
    edges = np.geomspace(60, fmax, n_bands + 1)
    lin = 10 ** (S / 20)
    out = np.empty((n_bands, S.shape[1]))
    for i in range(n_bands):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = lin[m].mean(axis=0) if m.any() else 1e-8
    if smooth > 1:
        k = np.ones(smooth) / smooth
        out = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"),
                                  1, out)
    return 20 * np.log10(out + 1e-8), edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='each as "Label=path.wav"')
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-label", default="ORIGINAL (reference)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=12000)
    ap.add_argument("--span", type=float, default=18.0,
                    help="dB range of the diverging scale")
    ap.add_argument("--title", default="Waveforms and difference maps")
    args = ap.parse_args()

    ref, sr = sf.read(args.ref, always_2d=True, dtype="float64")
    ref = ref[: int(args.seconds * sr)].mean(axis=1)

    items = []
    for spec in args.panels:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(spec), spec
        y, s = sf.read(path, always_2d=True, dtype="float64")
        y = y[: int(args.seconds * s)].mean(axis=1)
        if s != sr:
            import librosa
            y = librosa.resample(y, orig_sr=s, target_sr=sr)
        n = min(len(y), len(ref))
        # Match RMS, not a least-squares projection. These are different
        # performances, so they are not sample-aligned; ref.y is then near zero
        # and a projection-based gain scales the track almost to silence.
        g = np.sqrt((ref[:n] ** 2).mean() / ((y[:n] ** 2).mean() + 1e-20))
        items.append(dict(label=label, y=y[:n] * g))
    n = min([len(ref)] + [len(it["y"]) for it in items])
    ref = ref[:n]
    for it in items:
        it["y"] = it["y"][:n]

    S_ref = spec_db(ref)
    freqs = np.linspace(0, sr / 2, S_ref.shape[0])
    fsel = freqs <= args.fmax
    B_ref, edges = band_db(ref, sr, fmax=args.fmax)
    dur = n / sr

    rows = len(items) + 1
    fig = plt.figure(figsize=(14, 2.05 * rows + 2.4), facecolor=BG)
    gs = fig.add_gridspec(rows, 2, width_ratios=[1, 1], hspace=0.34,
                          wspace=0.11, left=0.055, right=0.955,
                          top=1 - 1.15 / (2.05 * rows + 2.4),
                          bottom=0.55 / (2.05 * rows + 2.4))

    px = 1800
    lo, hi = envelope(ref, px)
    t = np.linspace(0, dur, len(lo))
    amp = max(np.abs(ref).max(), 1e-6) * 1.1

    ax = fig.add_subplot(gs[0, 0], facecolor=BG)
    ax.fill_between(t, lo, hi, color=REF_C, lw=0)
    ax.set_ylim(-amp, amp)
    ax.set_xlim(0, dur)
    ax.set_title(args.ref_label, color=INK, fontsize=10.5, loc="left", pad=5)
    ax.set_ylabel("amp", color=MUTED, fontsize=8.5)

    axr = fig.add_subplot(gs[0, 1], facecolor=BG)
    axr.imshow(S_ref[fsel], origin="lower", aspect="auto", cmap="magma",
               extent=[0, dur, 0, args.fmax / 1000],
               vmin=S_ref[fsel].max() - 80, vmax=S_ref[fsel].max())
    axr.set_title("its spectrogram (for reference)", color=MUTED,
                  fontsize=10, loc="left", pad=5)
    axr.set_ylabel("kHz", color=MUTED, fontsize=8.5)

    im = None
    for i, it in enumerate(items):
        r = i + 1
        a1 = fig.add_subplot(gs[r, 0], facecolor=BG)
        lo, hi = envelope(it["y"], px)
        a1.fill_between(np.linspace(0, dur, len(lo)), lo, hi,
                        color=SER[i % len(SER)], lw=0)
        a1.set_ylim(-amp, amp)
        a1.set_xlim(0, dur)
        a1.set_title(it["label"], color=INK, fontsize=10.5, loc="left", pad=5)
        a1.set_ylabel("amp", color=MUTED, fontsize=8.5)

        a2 = fig.add_subplot(gs[r, 1], facecolor=BG)
        B, _ = band_db(it["y"], sr, fmax=args.fmax)
        m = min(B.shape[1], B_ref.shape[1])
        D = B[:, :m] - B_ref[:, :m]
        # blank bands where the original has essentially nothing to compare to
        quiet = B_ref[:, :m] < (B_ref.max() - 70)
        D = np.where(quiet, 0.0, D)
        im = a2.imshow(D, origin="lower", aspect="auto", cmap=DIVERGING,
                       vmin=-args.span, vmax=args.span,
                       extent=[0, dur, edges[0] / 1000, args.fmax / 1000])
        a2.set_yscale("log")
        a2.set_ylim(edges[0] / 1000, args.fmax / 1000)
        a2.set_title("difference vs original", color=MUTED, fontsize=10,
                     loc="left", pad=5)
        a2.set_ylabel("kHz", color=MUTED, fontsize=8.5)

        if r == rows - 1:
            a1.set_xlabel("seconds", color=MUTED, fontsize=8.5)
            a2.set_xlabel("seconds", color=MUTED, fontsize=8.5)

    for a in fig.axes:
        a.tick_params(colors=MUTED, labelsize=8, length=3)
        for s in a.spines.values():
            s.set_color(GRID)

    cax = fig.add_axes([0.957, 0.10, 0.010, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dB vs original   (warm = cover louder, cool = original louder)",
                 color=MUTED, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
    cb.outline.set_edgecolor(GRID)

    H = 2.05 * rows + 2.4
    fig.text(0.055, 1 - 0.34 / H, args.title, color=INK, fontsize=15,
             weight="600", ha="left", va="top")
    fig.text(0.055, 1 - 0.74 / H,
             "Gain-matched before differencing, so the maps show spectral "
             "change rather than level. Neutral = identical to the original.",
             color=MUTED, fontsize=9.5, ha="left", va="top")

    fig.savefig(args.output, dpi=140, facecolor=BG, bbox_inches="tight")
    print("wrote", args.output)


if __name__ == "__main__":
    sys.exit(main())
