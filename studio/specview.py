"""Single-panel spectrogram with EXACT, published pixel geometry.

`spectrogram.py` renders multi-panel report figures and saves with
`bbox_inches="tight"`, which crops the canvas to the drawn content -- fine for a
document, useless for click-picking, because the final pixel size and the axes
position are then both unknown.

This renderer exists so a click at pixel (px, py) can be converted to (time,
frequency) by arithmetic rather than guesswork:

  * figsize x dpi is chosen to land on integer pixels
  * the axes are placed with an explicit `add_axes([L, B, W, H])`
  * NONE of `tight_layout()`, `constrained_layout`, or `bbox_inches="tight"`
    is used -- any one of them silently moves the axes and breaks the mapping
  * the plot rectangle is emitted as JSON, derived from the SAME constants that
    positioned the axes, so the two cannot drift apart

Prints one JSON object to stdout; everything else goes to stderr.

Usage:
    python specview.py --input a.wav --out a.png [--region t0:t1:flo:fhi]
                       [--marker T:F] [--max-seconds 300]
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 1300 x 560 canvas; every derived edge below is a whole pixel.
FIG_W_IN, FIG_H_IN, DPI = 13.0, 5.6, 100
L, B, W, H = 0.06, 0.10, 0.90, 0.85

INK, MUTED, BG, GRID = "#e9e9ea", "#9a9aa0", "#141416", "#3a3a40"
ACCENT, REGION, MARK = "#f5c542", "#4fd1c5", "#ff5c8a"
DB_FLOOR = -95.0


def stft_db(x, n_fft, hop):
    win = np.hanning(n_fft)
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * win, axis=1)).T
    return 20 * np.log10(S + 1e-10)


def detect_cliff(mono, sr, n=8192):
    win = np.hanning(n)
    acc, count = np.zeros(n // 2 + 1), 0
    step = max(n, (len(mono) - n) // 400) if len(mono) > n else n
    for s in range(0, max(1, len(mono) - n), step):
        acc += np.abs(np.fft.rfft(mono[s:s + n] * win))
        count += 1
    if count:
        acc /= count
    fr = np.fft.rfftfreq(n, 1 / sr)
    db = 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)
    sm = np.convolve(db, np.ones(9) / 9, mode="same")
    d = np.diff(sm)
    sel = fr[:-1] > 6000
    if not sel.any():
        return None, 0.0
    i = int(np.argmin(np.where(sel, d, 0)))
    cliff = float(fr[i])
    lo = db[(fr >= cliff - 1500) & (fr < cliff)]
    hi = db[(fr > cliff) & (fr <= cliff + 1500)]
    return cliff, (float(lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0)


def parse4(spec, dur, nyq):
    p = spec.split(":")
    if len(p) != 4:
        raise ValueError(f"expected t0:t1:flo:fhi, got {spec!r}")
    v = lambda s, d: (d if s.strip() in ("", "*") else float(s))
    return v(p[0], 0.0), v(p[1], dur), v(p[2], 0.0), v(p[3], nyq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region", default=None, help="t0:t1:flo:fhi")
    ap.add_argument("--marker", default=None,
                    help="TIME:FREQ crosshair -- draws where the caller "
                         "*thinks* the click landed, so a coordinate-space "
                         "mismatch is visible instead of silent")
    ap.add_argument("--max-seconds", type=float, default=300.0)
    args = ap.parse_args()

    x, sr = sf.read(args.input, always_2d=True, dtype="float32")
    m = x.mean(axis=1)
    full_dur = len(m) / sr
    if full_dur > args.max_seconds:
        m = m[: int(args.max_seconds * sr)]
    dur = len(m) / sr
    nyq = sr / 2

    n_fft = 2048
    hop = max(256, int(len(m) / 2400) // 2 * 2 or 256)
    S = stft_db(m, n_fft, hop)
    vmax = float(S.max())

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([L, B, W, H], facecolor=BG)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
              vmin=vmax + DB_FLOOR, vmax=vmax,
              extent=[0.0, dur, 0.0, nyq / 1000])
    ax.set_xlim(0.0, dur)
    ax.set_ylim(0.0, nyq / 1000)
    ax.set_xlabel("seconds", color=MUTED, fontsize=9)
    ax.set_ylabel("kHz", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for s in ax.spines.values():
        s.set_color(GRID)

    cliff, drop = detect_cliff(m, sr)
    if cliff and drop > 8:
        ax.axhline(cliff / 1000, color=ACCENT, lw=1.1, ls=(0, (5, 3)),
                   alpha=0.9)
        ax.text(dur * 0.995, cliff / 1000 + 0.25,
                f"cliff {cliff/1000:.1f} kHz", color=ACCENT, fontsize=8.5,
                ha="right", va="bottom")

    if args.region:
        t0, t1, flo, fhi = parse4(args.region, dur, nyq)
        t1 = min(t1, dur)
        ax.add_patch(Rectangle((t0, flo / 1000), max(t1 - t0, 0),
                               (fhi - flo) / 1000, fill=True,
                               facecolor=REGION, alpha=0.13, edgecolor=REGION,
                               lw=1.5, ls=(0, (4, 2)), zorder=5))
        ax.text(min(t1, dur) - dur * 0.004, fhi / 1000 - 0.25,
                f"{t0:.2f}–{t1:.2f}s · {flo/1000:.1f}–{fhi/1000:.1f} kHz",
                color=REGION, fontsize=8, ha="right", va="top", zorder=6)

    if args.marker:
        mt, mf = (float(v) for v in args.marker.split(":"))
        ax.plot([mt], [mf / 1000], marker="+", ms=13, mew=1.6, color=MARK,
                zorder=8)

    # No tight_layout / constrained_layout / bbox_inches -- see the docstring.
    fig.savefig(args.out, dpi=DPI, facecolor=BG)
    plt.close(fig)

    px_w, px_h = int(round(FIG_W_IN * DPI)), int(round(FIG_H_IN * DPI))
    geom = {
        "png": os.path.abspath(args.out),
        "img_w": px_w, "img_h": px_h,
        # Plot rectangle in image pixels, from the same L/B/W/H used above.
        # Image y grows downward, so the axes' top edge is the smaller y.
        # Rounded: these are pixel edges, and the fractions land on whole
        # pixels by construction -- rounding only strips float noise.
        "x0": round(L * px_w), "x1": round((L + W) * px_w),
        "y0": round((1.0 - (B + H)) * px_h), "y1": round((1.0 - B) * px_h),
        "t0": 0.0, "t1": dur, "f0": 0.0, "f1": nyq,
        "sr": sr, "full_duration": full_dur, "truncated": full_dur > dur,
        "cliff": cliff, "drop": drop,
    }
    print(json.dumps(geom))
    return 0


if __name__ == "__main__":
    sys.exit(main())
