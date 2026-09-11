"""
Time-domain corroboration: is there frame-synchronous structure in the waveform?

A 200 Hz spectral comb implies a 5 ms period in the time domain. If that comes
from a vocoder's overlap-add frame boundaries, then chopping the waveform into
exactly-period-length blocks and averaging them should reveal a repeating
pattern -- everything musical averages away (it is not locked to the frame
grid), while a synthesis artefact survives because it is.

This is an independent check on the spectral measurement: it uses the raw
waveform, no STFT, no learned weights.

Statistic
    fold the emphasised residual at period P, average N blocks, and compare the
    surviving energy against what uncorrelated noise would leave (which falls as
    1/N). A "gain" of 1 means nothing is frame-locked; >>1 means something is.

Wrong periods are folded too, as a null: a real artefact spikes only at the
true period.

Usage:
    python frame_fold.py file.wav --rate 200
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


def fold_gain(x, period, emphasise=True):
    """Energy surviving a period-synchronous average, relative to the noise
    floor that an unlocked signal would leave."""
    if emphasise:
        # Second difference: flattens musical (smooth, low-order) content and
        # exposes sample-scale discontinuities such as frame joins.
        x = np.diff(x, n=2)
    n_blocks = len(x) // period
    if n_blocks < 64:
        return np.nan
    blocks = x[: n_blocks * period].reshape(n_blocks, period)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)
    mean_profile = blocks.mean(axis=0)
    # Uncorrelated content averages down as 1/sqrt(N); anything above that
    # floor is locked to this period.
    expected = blocks.var() / n_blocks
    got = (mean_profile ** 2).mean()
    return float(got / (expected + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--rate", type=float, default=200.0,
                    help="frame rate to test, Hz (default 200)")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    print(f"testing frame rate {args.rate:g} Hz "
          f"(and nearby wrong rates as a null)\n")
    print(f"{'file':<30} {'period':>8} {'GAIN':>9} {'null median':>12} {'ratio':>8}")
    print("-" * 72)

    for f in args.files:
        try:
            x, sr = sf.read(f, always_2d=True, dtype="float64")
        except Exception as e:
            print(f"{os.path.basename(f)[:30]:<30}  {type(e).__name__}")
            continue
        x = x[: int(args.seconds * sr)].mean(axis=1)

        p = sr / args.rate
        if abs(p - round(p)) > 1e-6:
            print(f"{os.path.basename(f)[:30]:<30}  "
                  f"{args.rate:g} Hz is not an integer period at {sr} Hz")
            continue
        p = int(round(p))
        g = fold_gain(x, p)

        # Null: neighbouring periods that no vocoder would use.
        nulls = [fold_gain(x, p + d) for d in (-7, -5, -3, -2, 2, 3, 5, 7)]
        nulls = [v for v in nulls if np.isfinite(v)]
        nmed = float(np.median(nulls)) if nulls else float("nan")
        ratio = g / (nmed + 1e-12)
        mark = "   <-- FRAME-LOCKED" if ratio > 3 and g > 3 else ""
        print(f"{os.path.basename(f)[:30]:<30} {p:>8d} {g:>9.2f} "
              f"{nmed:>12.2f} {ratio:>8.2f}{mark}")

    print()
    print("GAIN = energy surviving a period-synchronous average, vs the 1/N")
    print("       floor uncorrelated content would leave. 1.0 = nothing locked.")


if __name__ == "__main__":
    sys.exit(main())
