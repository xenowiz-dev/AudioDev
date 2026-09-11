"""
Epoch-folding scan for a vocoder frame rate — the best detector found here.

`frame_fold.py` needed you to already know the frame rate and to pick one that
divided the sample rate exactly. This searches for it, at fractional periods, and
attaches a real significance to what it finds.

Why folding beats a spectral comb search: folding averages N blocks coherently,
so the artefact gains ~sqrt(N) over the music (N ≈ 750 for 30 s at 25 Hz). A
magnitude-domain comb ratio gets no such gain, which is why ACE-Step's artefact
reads 1.3x in the spectrum but 20x under folding.

Method (standard epoch folding, as used for periodic signals of unknown period)
  1. emphasise sample-scale discontinuities with a second difference
  2. for each trial period P, assign every sample a phase bin
  3. chi-square the folded profile against a flat one; under the null of no
     periodicity that is chi2 with (bins-1) degrees of freedom, so it converts
     to a z-score with no calibration needed
  4. report the best period, and check its harmonics to reject subharmonics

Usage:
    python frame_scan.py file.wav [more.wav ...]
    python frame_scan.py --lo 20 --hi 400 --seconds 60 file.wav
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

COMMON_RATES = (16000, 22050, 24000, 32000, 44100, 48000)


def emphasise(x):
    """Second difference: flattens smooth musical content, exposes joins."""
    y = np.diff(x, n=2)
    s = y.std()
    return y / (s + 1e-20)


def fold_chi2(y, period, bins=48):
    """Chi-square of the phase-folded profile against flat, as a z-score."""
    n = len(y)
    if period < 8 or n / period < 40:
        return np.nan
    idx = np.arange(n)
    phase = np.mod(idx, period) / period
    b = (phase * bins).astype(np.int64)
    np.clip(b, 0, bins - 1, out=b)
    counts = np.bincount(b, minlength=bins).astype(np.float64)
    sums = np.bincount(b, weights=y, minlength=bins)
    good = counts > 0
    means = np.zeros(bins)
    means[good] = sums[good] / counts[good]
    # y is unit-variance, so each bin mean has variance 1/count under the null.
    chi2 = float(np.sum(counts[good] * means[good] ** 2))
    dof = int(good.sum())
    return (chi2 - dof) / np.sqrt(2.0 * dof)


def scan(path, lo_hz, hi_hz, seconds, bins=48):
    """Coarse scan on a short segment, then refine on the full one.

    Folding needs period resolution finer than P^2/(N*bins) or the phase drifts
    across the record and the peak is missed. That scales with record length, so
    scanning the full clip at usable resolution is enormously expensive. A short
    segment has a correspondingly broader peak, which can be located cheaply and
    then sharpened against all the data.
    """
    x_full, sr = sf.read(path, always_2d=True, dtype="float64")
    x_full = x_full[: int(seconds * sr)].mean(axis=1)
    if len(x_full) < sr:
        raise ValueError("clip too short")
    p_hi, p_lo = sr / lo_hz, sr / hi_hz          # period in samples

    def grid(p_from, p_to, n_samples, nbins, safety=0.7):
        """Period steps sized so phase drifts less than one bin across the
        record. The tolerable step scales as P^2/(N*bins), so the grid must be
        dense at short periods and sparse at long ones -- a uniform step is
        either ruinously slow or silently misses the peak."""
        out, p = [], p_from
        while p < p_to:
            out.append(p)
            p += max(0.02, p * p / (n_samples * nbins) * safety)
        return np.array(out)

    # Hierarchical: cheap wide scan, then progressively more data and bins
    # around the surviving candidates.
    stages = [(1.5, 8, 12), (6.0, 16, 4), (seconds, bins, 1)]
    cands = None
    null_z = 0.0
    for k, (secs, nb, keep) in enumerate(stages):
        y = emphasise(x_full[: int(secs * sr)])
        if k == 0:
            ps = grid(p_lo, p_hi, len(y), nb)
        else:
            ps = np.concatenate([grid(max(p_lo, c - w), c + w, len(y), nb)
                                 for c, w in cands])
        z = np.array([fold_chi2(y, p, nb) for p in ps])
        ok = np.isfinite(z)
        ps, z = ps[ok], z[ok]
        if not len(z):
            return None
        if k == 0:
            null_z = float(np.median(z))
            # The grid is far denser at short periods, so a global argmax is
            # biased toward them purely by trial count. Take the best few from
            # each octave instead, which gives every octave equal opportunity.
            oct_idx = np.floor(np.log2(ps / p_lo)).astype(int)
            order = []
            for o in np.unique(oct_idx):
                sel = np.where(oct_idx == o)[0]
                order.extend(sel[np.argsort(z[sel])[::-1][:3]])
            order = np.array(order, dtype=int)
        else:
            order = np.argsort(z)[::-1][:keep]
        width = (ps[1] - ps[0]) * 3 if len(ps) > 1 else 1.0
        cands = [(float(ps[i]), max(width, ps[i] * 0.004)) for i in order]
        top = order[int(np.argmax(z[order]))]
        best_p, best_z = float(ps[top]), float(z[top])

    # Subharmonic guard: if P/2 folds nearly as well, the true period is P/2.
    y_full = emphasise(x_full)
    for div in (2, 3):
        cand = best_p / div
        if cand >= p_lo:
            zc = fold_chi2(y_full, cand, bins)
            if np.isfinite(zc) and zc > best_z * 0.7:
                best_p, best_z = cand, float(zc)
                break

    return dict(sr=sr, period=best_p, rate=sr / best_p, z=best_z,
                null=null_z, n_blocks=int(len(y_full) / best_p))


def decode(rate):
    out = []
    for r in COMMON_RATES:
        h = r / rate
        if abs(h - round(h)) < 0.02 and 32 <= round(h) <= 4096:
            out.append(f"{round(h)}@{r//1000}k")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--lo", type=float, default=20.0, help="lowest frame rate Hz")
    ap.add_argument("--hi", type=float, default=400.0, help="highest frame rate Hz")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--z", type=float, default=25.0,
                    help="z threshold to call it generated (default 25)")
    args = ap.parse_args()

    print(f"{'file':<30} {'rate Hz':>9} {'z':>10} {'null':>7} {'blocks':>7}  verdict")
    print("-" * 88)
    for f in args.files:
        try:
            r = scan(f, args.lo, args.hi, args.seconds, args.bins)
        except Exception as e:
            print(f"{os.path.basename(f)[:30]:<30}  {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        hit = r["z"] >= args.z
        v = "FRAME-LOCKED" if hit else "none"
        extra = ""
        if hit:
            d = decode(r["rate"])
            if d:
                extra = "  hop " + ", ".join(d)
        print(f"{os.path.basename(f)[:30]:<30} {r['rate']:>9.2f} {r['z']:>10.1f} "
              f"{r['null']:>7.1f} {r['n_blocks']:>7d}  {v}{extra}")
    print()
    print("z = chi-square of the folded profile vs flat, in sigmas. Under 'no")
    print("    periodicity' this is ~0 regardless of the material.")


if __name__ == "__main__":
    sys.exit(main())
