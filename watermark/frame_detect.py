"""
Vocoder frame-rate detector: cheap search, exact confirmation.

Blind epoch folding does not work here, and the reason is worth stating. The
artefact is sample-scale, so the fold needs roughly one phase bin per sample to
see it; but with one bin per sample the trial period must be accurate to about
P/N (a few thousandths of a sample over a 20 s clip). Searching a wide range at
that precision is computationally hopeless -- and coarsening the bins to make
the search affordable averages the artefact away. Measured directly: Suno folded
at its true period scores z = 63 with one bin per sample and only z = 5.4 with
48 bins.

So: don't search blindly. Generate candidates cheaply, confirm them exactly.

  1. CANDIDATES  a) every plausible (hop, sample-rate) pair -- vocoder frame
                    rates are not arbitrary, they are sr/hop for round hops
                 b) whatever a spectral comb search suggests
  2. CONFIRM     fold at full resolution (one bin per sample) at each candidate,
                 with a local period refinement
  3. NULL        fold at deliberately wrong periods nearby; a real frame lock
                 towers over them, musical periodicity does not

Usage:
    python frame_detect.py file.wav [more.wav ...]
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

COMMON_RATES = (16000, 22050, 24000, 32000, 44100, 48000)
COMMON_HOPS = (64, 80, 100, 120, 128, 160, 192, 200, 240, 256, 300, 320, 320,
               384, 400, 441, 480, 512, 600, 640, 768, 800, 960, 1024, 1200,
               1280, 1536, 1920, 2048)


def candidate_rates(lo=15.0, hi=400.0):
    """Frame rates are sr/hop for round hop sizes, not arbitrary numbers."""
    out = set()
    for sr in COMMON_RATES:
        for h in COMMON_HOPS:
            r = sr / h
            if lo <= r <= hi:
                out.add(round(r, 4))
    return sorted(out)


def emphasise(x):
    y = np.diff(x, n=2)
    return y / (y.std() + 1e-20)


def fold_gain(y, period_samples):
    """Energy surviving a period-synchronous average, over the 1/N noise floor.

    Integer period only -- this is the exact, full-resolution measurement.
    """
    p = int(round(period_samples))
    n = len(y) // p
    if p < 8 or n < 32:
        return np.nan
    b = y[: n * p].reshape(n, p)
    b = b - b.mean(axis=1, keepdims=True)
    prof = b.mean(axis=0)
    expected = b.var() / n
    return float((prof ** 2).mean() / (expected + 1e-30))


def fold_gain_frac(y, period, bins=None):
    """Same measurement at a FRACTIONAL period, via interpolated folding.

    NEGATIVE RESULT -- this does not work, and the reason is worth keeping.
    The intent was to follow a speed change, which moves the frame period off
    the integers (a 0.5 % resample turns 240 samples into 238.806). But linear
    interpolation spreads every sample across two phase bins, which low-passes
    the folded profile; and the artefact is *sample-scale*, so it smears below
    detectability. Measured: the same file scores 32.8x under integer folding
    and 3.1x through this path. `--drift` is therefore off by default and
    should stay off.

    The correct fix is the other way round: sweep RESAMPLING ratios, resample
    the audio properly (a real polyphase resampler preserves the artefact far
    better than fold-time interpolation), and integer-fold each candidate. That
    is one sweep over ratios rather than a 2-D search, so it is affordable --
    it just is not built yet.
    """
    if period < 8:
        return np.nan
    b = int(round(period)) if bins is None else bins
    n = len(y)
    if n / period < 32:
        return np.nan
    pos = (np.arange(n) % period) / period * b
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i0 = np.mod(i0, b)
    i1 = np.mod(i0 + 1, b)
    w0, w1 = 1.0 - frac, frac
    sums = np.bincount(i0, weights=y * w0, minlength=b) + \
        np.bincount(i1, weights=y * w1, minlength=b)
    cnts = np.bincount(i0, weights=w0, minlength=b) + \
        np.bincount(i1, weights=w1, minlength=b)
    good = cnts > 8
    if good.sum() < b * 0.5:
        return np.nan
    prof = sums[good] / cnts[good]
    # Under the null each bin mean has variance var(y)/count.
    expected = y.var() * np.mean(1.0 / cnts[good])
    return float(np.mean(prof ** 2) / (expected + 1e-30))


def evaluate(y, sr, rate, drift=0.0):
    """Fold at this rate, and at nearby wrong periods as a local null.

    drift > 0 additionally sweeps the period by +/-drift (fractionally), which
    recovers material whose speed has been changed -- a resample moves the frame
    rate to rate*r and takes the period off the integers.
    """
    p0 = sr / rate
    best_p, best_g = None, -np.inf
    if drift <= 0:
        if abs(p0 - round(p0)) > 0.02:
            return None                 # non-integer period, no sweep requested
        best_p = float(round(p0))
        best_g = fold_gain(y, best_p)
    else:
        # Two-stage sweep. The tolerable period step scales as P^2/N, so a
        # short prefix allows a far coarser grid; locate the peak there, then
        # sharpen it against the whole record. Sweeping the full record at the
        # fine step directly costs ~500 folds per candidate rate and is far too
        # slow to be useful.
        y_c = y[: max(len(y) // 5, 3 * 48000)]
        step_c = max(1e-3, p0 * p0 / len(y_c) * 0.35)
        for p in np.arange(p0 * (1 - drift), p0 * (1 + drift), step_c):
            g = fold_gain_frac(y_c, p)
            if np.isfinite(g) and g > best_g:
                best_p, best_g = float(p), float(g)
        if best_p is None:
            return None
        step_f = max(1e-4, p0 * p0 / len(y) * 0.35)
        best_g = -np.inf
        for p in np.arange(best_p - step_c, best_p + step_c, step_f):
            g = fold_gain_frac(y, p)
            if np.isfinite(g) and g > best_g:
                best_p, best_g = float(p), float(g)
    if best_p is None or not np.isfinite(best_g):
        return None

    nulls = [fold_gain_frac(y, best_p + d) for d in (-7, -5, -3, -2, 2, 3, 5, 7)]
    nulls = [v for v in nulls if np.isfinite(v)]
    nmed = float(np.median(nulls)) if nulls else np.nan
    return dict(rate=sr / best_p, period=best_p, gain=best_g, null=nmed,
                ratio=best_g / (nmed + 1e-12), blocks=int(len(y) / best_p))


def detect(path, seconds=30.0, lo=15.0, hi=400.0, extra_rates=(), drift=0.0):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x[: int(seconds * sr)].mean(axis=1)
    if len(x) < sr:
        raise ValueError("clip too short")
    y = emphasise(x)

    rates = sorted(set(list(candidate_rates(lo, hi)) + list(extra_rates)))
    results = [r for r in (evaluate(y, sr, R, drift) for R in rates) if r]
    if not results:
        return None, []
    results.sort(key=lambda r: r["ratio"], reverse=True)

    # Report the strongest detection, and the whole harmonically-related family
    # alongside it. Pinning down the FUNDAMENTAL is genuinely ambiguous: the
    # artefact behaves like an impulse train at the frame boundaries, and such a
    # train folds just as well at any submultiple period (every join still lands
    # at phase zero, with more blocks to average). Attempts to prefer either the
    # highest or the lowest rate both mis-identify known cases, so the honest
    # output is the family, not a single guess.
    best = results[0]
    family = [r for r in results
              if r["ratio"] > best["ratio"] * 0.5
              and abs(max(r["rate"], best["rate"]) / min(r["rate"], best["rate"])
                      - round(max(r["rate"], best["rate"])
                              / min(r["rate"], best["rate"]))) < 0.02]
    family.sort(key=lambda r: r["rate"])
    return best, family


def decode(rate):
    out = []
    for sr in COMMON_RATES:
        h = sr / rate
        if abs(h - round(h)) < 0.02 and 32 <= round(h) <= 4096:
            out.append(f"{round(h)}@{sr//1000}k")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--lo", type=float, default=15.0)
    ap.add_argument("--hi", type=float, default=400.0)
    ap.add_argument("--ratio", type=float, default=4.0,
                    help="fold gain over local null to call it (default 4)")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="EXPERIMENTAL, DOES NOT WORK: fractional period sweep "
                         "intended to tolerate speed changes. Interpolated "
                         "folding smears the sample-scale artefact and loses "
                         "even undegraded files (32.8x -> 3.1x). Leave at 0.")
    args = ap.parse_args()

    print(f"{'file':<30} {'rate Hz':>9} {'gain':>9} {'null':>7} {'ratio':>8} "
          f"{'blocks':>7}  verdict")
    print("-" * 96)
    for f in args.files:
        try:
            best, family = detect(f, args.seconds, args.lo, args.hi,
                                  drift=args.drift)
        except Exception as e:
            print(f"{os.path.basename(f)[:30]:<30}  {type(e).__name__}: {e}")
            continue
        if best is None:
            print(f"{os.path.basename(f)[:30]:<30}  no measurable candidate")
            continue
        hit = best["ratio"] >= args.ratio and best["gain"] >= 4
        v = "FRAME-LOCKED" if hit else "none"
        print(f"{os.path.basename(f)[:30]:<30} {best['rate']:>9.2f} "
              f"{best['gain']:>9.2f} {best['null']:>7.2f} {best['ratio']:>8.2f} "
              f"{best['blocks']:>7d}  {v}")
        if hit and len(family) > 1:
            fam = " ".join(f"{r['rate']:g}({r['ratio']:.0f}x)" for r in family)
            print(f"{'':<30} harmonic family: {fam}")
            lowest = family[0]["rate"]
            d = decode(lowest)
            if d:
                print(f"{'':<30} lowest member {lowest:g} Hz -> hop "
                      + ", ".join(d))
    print()
    print("gain  = energy surviving a period-synchronous average vs the 1/N floor")
    print("ratio = that gain over the same measure at neighbouring wrong periods")
    print("The family is reported because an impulse-train artefact folds at")
    print("every submultiple; the fundamental cannot be read off folding alone.")


if __name__ == "__main__":
    sys.exit(main())
