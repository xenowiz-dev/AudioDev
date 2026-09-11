"""
Confirm a frame rate by folding, after integerising the period by resampling.

Closes the one gap in `frame_detect.py`. Folding is the most sensitive
instrument available, but it needs an integer period, and any speed change
takes the period off the integers -- a 0.5 % resample turns 240 samples into
238.806, and folding then fails completely (measured: 32.8x -> 2.1x).

Interpolated folding does not fix it: the artefact is sample-scale and linear
interpolation smears it away. But a *real* polyphase resampler does not smear it
nearly as much, so the fix is to resample the audio so the period becomes an
exact integer, then fold normally.

The rate estimate comes from the spectral comb search, which is the one method
that tracks a speed change accurately (measured: recovers 201.00 / 204.00 /
194.00 Hz from +0.5 / +2 / -3 % resamples).

Pipeline:  comb search -> rate estimate -> resample to integerise -> fold

Usage:
    python frame_confirm.py file.wav                # auto rate from comb search
    python frame_confirm.py file.wav --rate 201.0   # explicit
"""

import argparse
import os
import sys
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_detect import emphasise, fold_gain, decode


def integerise(x, sr, rate):
    """Resample so that sr/rate becomes an exact integer number of samples."""
    p = sr / rate
    p_target = int(round(p))
    if p_target < 8:
        return None, None
    f = Fraction(p_target / p).limit_denominator(20000)
    up, down = f.numerator, f.denominator
    if up == down:
        return x, p_target
    # Guard against a pathological rational blow-up.
    if max(up, down) > 200000:
        return None, None
    return resample_poly(x, up, down), p_target


def confirm(path, rate, seconds=30.0):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x[: int(seconds * sr)].mean(axis=1)
    y, p = integerise(x, sr, rate)
    if y is None:
        return None
    e = emphasise(y)
    g = fold_gain(e, p)
    nulls = [fold_gain(e, p + d) for d in (-7, -5, -3, -2, 2, 3, 5, 7)]
    nulls = [v for v in nulls if np.isfinite(v)]
    nmed = float(np.median(nulls)) if nulls else np.nan
    if not np.isfinite(g):
        return None
    return dict(rate=rate, period=p, gain=g, null=nmed,
                ratio=g / (nmed + 1e-12), blocks=len(e) // p)


def comb_rate(path):
    """Rate estimate from the spectral comb search, which follows resampling."""
    from comb_detect import analyse
    r = analyse(path, 1000.0, None, lo=15.0, hi=400.0)
    return r["spacing"] if r else None


def hypotheses(path, lo=15.0, hi=400.0):
    """Rate hypotheses from BOTH sources, because each fails where the other
    works: the enumerated sr/hop list survives lossy coding but cannot express
    a speed-shifted rate, while the comb estimate tracks a resample but is lost
    by heavy codecs. Their union covers both.
    """
    from frame_detect import candidate_rates
    rates = set(candidate_rates(lo, hi))
    try:
        c = comb_rate(path)
    except Exception:
        c = None
    if c:
        # include harmonics, since the comb search may land on a submultiple
        for k in (1, 2, 3, 4):
            if lo <= c * k <= hi:
                rates.add(round(c * k, 4))
    return sorted(rates), c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--rate", type=float, default=None,
                    help="frame rate Hz; omit to estimate via comb search")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--ratio", type=float, default=4.0)
    args = ap.parse_args()

    print(f"{'file':<28} {'rate Hz':>9} {'period':>7} {'gain':>9} "
          f"{'ratio':>8}  verdict")
    print("-" * 84)
    for f in args.files:
        try:
            if args.rate is not None:
                rates, comb_c = [args.rate], None
            else:
                rates, comb_c = hypotheses(f)
            r, src = None, "given"
            for R in rates:
                try:
                    cand = confirm(f, R, args.seconds)
                except Exception:
                    continue
                if cand and (r is None or cand["ratio"] > r["ratio"]):
                    r = cand
                    src = ("comb" if comb_c and
                           abs(R / comb_c - round(R / comb_c)) < 0.02
                           else "list") if args.rate is None else "given"
        except Exception as e:
            print(f"{os.path.basename(f)[:28]:<28}  {type(e).__name__}: {e}")
            continue
        if r is None:
            print(f"{os.path.basename(f)[:28]:<28}  no usable rate hypothesis")
            continue
        hit = r["ratio"] >= args.ratio and r["gain"] >= 4
        v = "FRAME-LOCKED" if hit else "none"
        extra = ""
        if hit:
            d = decode(r["rate"])
            if d:
                extra = "  hop " + ", ".join(d)
        print(f"{os.path.basename(f)[:28]:<28} {r['rate']:>9.2f} "
              f"{r['period']:>7d} {r['gain']:>9.2f} {r['ratio']:>8.2f}  "
              f"{v}  [{src}]{extra}")
    print()
    print("Audio is resampled so the frame period lands on an integer, then")
    print("folded at full sample resolution -- the sensitive measurement.")


if __name__ == "__main__":
    sys.exit(main())
