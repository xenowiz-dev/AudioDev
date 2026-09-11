"""
Before/after spectrogram comparison for audio super-resolution results.

Renders N files as stacked spectrogram panels sharing one STFT config and one
dB colour scale, so panels are actually comparable. Detects and annotates each
file's high-frequency cliff -- that horizontal edge is the thing you're looking
for.

Colour: magma. A spectrogram encodes magnitude, so the ramp must be monotonic in
lightness; magma is perceptually uniform and CVD-safe. (A rainbow/jet ramp is
non-monotonic and invents structure that isn't in the data.)

Usage:
    python spectrogram.py -o out.png "Before=a.wav" "After=b.wav"
    python spectrogram.py -o out.png --start 60 --dur 20 "Ref=r.wav" "SR=s.wav"
"""

import argparse
import os

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

N_FFT = 2048
HOP = 512
DB_FLOOR = -95.0   # dB below panel peak to show
INK = "#e9e9ea"
MUTED = "#9a9aa0"
BG = "#141416"
ACCENT = "#f5c542"
# Distinct from the cliff line (amber): a selection is a different kind of
# annotation and the two often sit within a kHz of each other.
REGION = "#4fd1c5"


def stft_db(x, n_fft=N_FFT, hop=HOP):
    win = np.hanning(n_fft)
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 1:
        raise ValueError("clip shorter than one FFT window")
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * win
    S = np.abs(np.fft.rfft(frames, axis=1)).T
    return 20 * np.log10(S + 1e-10)


def detect_cliff(x, sr):
    """Return (cliff_hz, drop_db) using the averaged spectrum."""
    n = 8192
    if len(x) < n * 2:
        return None, 0.0
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    cnt = 0
    for s in range(0, len(x) - n, max(n, (len(x) - n) // 300)):
        acc += np.abs(np.fft.rfft(x[s:s + n] * win))
        cnt += 1
    acc /= max(cnt, 1)
    fr = np.fft.rfftfreq(n, 1 / sr)
    db = 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)
    sm = np.convolve(db, np.ones(9) / 9, mode="same")
    d = np.diff(sm)
    searchable = fr[:-1] > 6000
    if not searchable.any():
        return None, 0.0
    i = int(np.argmin(np.where(searchable, d, 0)))
    cliff = fr[i]
    lo = db[(fr >= cliff - 1500) & (fr < cliff)]
    hi = db[(fr > cliff) & (fr <= cliff + 1500)]
    drop = (lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0
    return cliff, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panels", nargs="+", help='each as "Label=path.wav"')
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="offset seconds")
    ap.add_argument("--dur", type=float, default=20.0, help="seconds to render")
    ap.add_argument("--title", default="Spectrogram comparison")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--fmax", type=float, default=24000)
    ap.add_argument("--fmin", type=float, default=0,
                    help="zoom the frequency axis (Hz) - use to inspect a "
                         "narrow band such as 18000-24000")
    ap.add_argument("--floor", type=float, default=None,
                    help="override dB range below peak (default 95)")
    ap.add_argument("--region", action="append", default=None,
                    help="draw a t0:t1:flo:fhi box on every panel (repeatable) "
                         "- the selection region_fill.py would splice. '' or "
                         "'*' means the open end.")
    args = ap.parse_args()

    # Parsed once: None means "open end", resolved per panel since panels can
    # differ in duration and sample rate.
    regions = []
    for spec in (args.region or []):
        parts = spec.split(":")
        if len(parts) != 4:
            raise SystemExit(f"--region must be t0:t1:flo:fhi, got {spec!r}")
        v = lambda s, d: (d if s.strip() in ("", "*") else float(s))
        regions.append((v(parts[0], 0.0), v(parts[1], None),
                        v(parts[2], 0.0), v(parts[3], None)))

    items = []
    for p in args.panels:
        label, _, path = p.partition("=")
        if not path:
            label, path = os.path.basename(p), p
        x, sr = sf.read(path, always_2d=True, dtype="float32")
        m = x.mean(axis=1)
        a = int(args.start * sr)
        b = a + int(args.dur * sr)
        seg = m[a:b]
        if len(seg) < N_FFT * 2:
            seg = m[:int(args.dur * sr)]
        cliff, drop = detect_cliff(m, sr)
        items.append(dict(label=label, seg=seg, sr=sr, cliff=cliff, drop=drop,
                          ch=x.shape[1]))

    # One shared dB window across every panel, anchored to the loudest panel.
    specs = [stft_db(it["seg"]) for it in items]
    floor = args.floor if args.floor is not None else DB_FLOOR
    if args.fmin > 0:
        # When zoomed into a quiet high band, anchoring the scale to the
        # full-spectrum peak would render everything black. Anchor to the
        # loudest content actually inside the visible band instead.
        peaks = []
        for it, S in zip(items, specs):
            nyq = it["sr"] / 2
            lo = int(args.fmin / nyq * (S.shape[0] - 1))
            hi = int(min(args.fmax, nyq) / nyq * (S.shape[0] - 1))
            if hi > lo:
                peaks.append(S[lo:hi].max())
        vmax = max(peaks) if peaks else max(s.max() for s in specs)
    else:
        vmax = max(s.max() for s in specs)
    vmin = vmax + floor

    # Lay the header out in inches, not figure fractions, so the subtitle can
    # never land on top of the first panel as the panel count changes.
    n = len(items)
    panel_h = 2.55
    header_in = 1.15 if args.subtitle else 0.80
    footer_in = 0.55
    fig_h = header_in + panel_h * n + footer_in
    fig = plt.figure(figsize=(13, fig_h), facecolor=BG)
    gs = GridSpec(n, 2, width_ratios=[60, 1], hspace=0.34, wspace=0.02,
                  left=0.075, right=0.94,
                  top=1 - header_in / fig_h,
                  bottom=footer_in / fig_h)

    im = None
    for i, (it, S) in enumerate(zip(items, specs)):
        ax = fig.add_subplot(gs[i, 0], facecolor=BG)
        dur = len(it["seg"]) / it["sr"]
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax,
                       extent=[0, dur, 0, it["sr"] / 2 / 1000])
        ax.set_ylim(args.fmin / 1000, args.fmax / 1000)
        ax.set_ylabel("kHz", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        for s in ax.spines.values():
            s.set_color("#3a3a40")

        if it["cliff"] and it["drop"] > 8:
            y = it["cliff"] / 1000
            ax.axhline(y, color=ACCENT, lw=1.1, ls=(0, (5, 3)), alpha=0.9)
            # Flip the label under the line when the line sits near the top,
            # otherwise it runs off the panel.
            span = (args.fmax - args.fmin) / 1000
            near_top = y > args.fmax / 1000 - span * 0.092
            ax.text(dur * 0.995, y + (-0.45 if near_top else 0.45),
                    f"cliff {it['cliff']/1000:.1f} kHz",
                    color=ACCENT, fontsize=8.5, ha="right",
                    va="top" if near_top else "bottom")

        # Selection boxes, drawn on every panel so the same region can be
        # compared before and after filling it.
        for t0, t1, flo, fhi in regions:
            t1v = dur if t1 is None else min(t1, dur)
            fhiv = it["sr"] / 2 if fhi is None else fhi
            ax.add_patch(Rectangle(
                (t0, flo / 1000), max(t1v - t0, 0), (fhiv - flo) / 1000,
                fill=False, edgecolor=REGION, lw=1.4, ls=(0, (4, 2)),
                zorder=5))
            if i == 0:
                # Right-aligned at the box's trailing edge: the panel label
                # plate occupies the top-left of every panel.
                ax.text(t1v - 0.06, fhiv / 1000 - 0.10,
                        f"{flo/1000:.1f}–{fhiv/1000:.1f} kHz",
                        color=REGION, fontsize=7.5, va="top", ha="right",
                        zorder=6)

        # Opaque plate behind the panel label so a cliff line can't strike
        # through the text.
        plate = dict(facecolor=BG, edgecolor="none", alpha=0.82,
                     boxstyle="square,pad=0.28")
        tag = f"{it['sr']/1000:.1f} kHz · {'stereo' if it['ch'] > 1 else 'mono'}"
        ax.text(0.006, 0.965, it["label"], transform=ax.transAxes,
                color=INK, fontsize=11, va="top", ha="left", weight="600",
                bbox=plate)
        ax.text(0.006, 0.80, tag, transform=ax.transAxes,
                color=MUTED, fontsize=8.5, va="top", ha="left", bbox=plate)

        if i == n - 1:
            ax.set_xlabel("seconds", color=MUTED, fontsize=9)
        else:
            ax.set_xticklabels([])

    cax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dB (relative)", color=MUTED, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
    cb.outline.set_edgecolor("#3a3a40")

    fig.text(0.075, 1 - 0.30 / fig_h, args.title, color=INK, fontsize=14.5,
             weight="600", ha="left", va="top")
    if args.subtitle:
        fig.text(0.075, 1 - 0.66 / fig_h, args.subtitle,
                 color=MUTED, fontsize=9.5, ha="left", va="top")

    fig.savefig(args.output, dpi=150, facecolor=BG, bbox_inches="tight")
    print(f"wrote {args.output}")
    for it in items:
        c = f"{it['cliff']:.0f} Hz (drop {it['drop']:.1f} dB)" if it["cliff"] else "n/a"
        print(f"  {it['label']:<28} {it['sr']} Hz  cliff {c}")


if __name__ == "__main__":
    main()
