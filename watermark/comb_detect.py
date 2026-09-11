"""
Generator-agnostic vocoder-comb detector.

The shipped classifier is 3585 learned weights pinned to Suno's 200 Hz frame
rate, so it misses any generator with a different hop. But the artefact is
physical, not statistical: a neural vocoder that upsamples with transposed
convolutions at frame rate F stamps spectral peaks every F Hz. Search for the
spacing directly and you need no training and no per-generator retuning.

Method
  1. average power spectrum at native rate, large FFT for fine resolution
  2. subtract a local minimum-filter floor -> residual isolates peaks
  3. for each candidate spacing F, compare mean residual ON multiples of F
     against mean residual off them
  4. score the winning spacing against the spread of the whole sweep, so the
     multiple-comparisons problem of a fine sweep is accounted for

Reports peak spacing, its on/off-grid ratio, and a robust z-score. Also decodes
the spacing into plausible (sample rate, hop) pairs, which usually identifies
the architecture.

Usage:
    python comb_detect.py file.wav [more.wav ...]
    python comb_detect.py --fmin 1000 --fmax 16000 file.wav
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d

COMMON_RATES = (16000, 22050, 24000, 32000, 44100, 48000)


def residual_spectrum(path, n_fft=65536, fmin=1000.0, fmax=None,
                      max_seconds=180.0):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = x[: int(max_seconds * sr)].mean(axis=1)
    if len(x) < n_fft * 2:
        n_fft = max(2048, 1 << int(np.log2(max(len(x) // 2, 1024))))
    hop = n_fft // 2
    win = np.hanning(n_fft)
    acc = np.zeros(n_fft // 2 + 1)
    cnt = 0
    for s in range(0, len(x) - n_fft, hop):
        acc += np.abs(np.fft.rfft(x[s:s + n_fft] * win)) ** 2
        cnt += 1
    if cnt == 0:
        raise ValueError("clip too short")
    acc /= cnt
    db = 10 * np.log10(np.clip(acc, 1e-20, None))
    fr = np.fft.rfftfreq(n_fft, 1 / sr)

    # Local floor over ~40 Hz, well below any plausible comb spacing.
    width = max(5, int(40.0 / (fr[1] - fr[0])))
    floor = minimum_filter1d(db, size=width, mode="nearest")
    res = np.clip(db - floor, 0, None)

    hi = fmax if fmax else sr / 2 * 0.92
    m = (fr >= fmin) & (fr <= hi)
    return fr[m], res[m], sr


def comb_ratio(fr, res, spacing, tol_hz, odd_only=False):
    """Mean residual on multiples of `spacing`, relative to off-grid.

    odd_only restricts to ODD multiples, which is how a subharmonic is caught:
    a real comb at 2F puts energy on every even multiple of F, so F scores well
    overall while its odd multiples show nothing. Requiring odd multiples to be
    elevated forces the true fundamental.
    """
    q = fr / spacing
    d = np.abs(((q) + 0.5) % 1.0 - 0.5) * spacing
    on = d < tol_hz
    if odd_only:
        on = on & (np.round(q).astype(int) % 2 == 1)
    off = d > tol_hz * 2
    if on.sum() < 12 or off.sum() < 25:
        return np.nan
    return res[on].mean() / (res[off].mean() + 1e-12)


def analyse(path, fmin, fmax, lo=40.0, hi=400.0, step=0.05):
    """Score a candidate spacing by HARMONIC CONSISTENCY, not by its own ratio.

    Two problems with taking the argmax of a raw ratio sweep:
      * a comb at F also puts energy on every multiple of 2F, 3F, 4F, and the
        ratio there can be higher (fewer, stronger on-grid bins), so the sweep
        reports a harmonic instead of the fundamental;
      * the max of a noisy statistic over thousands of candidates is inflated,
        which false-positives on ordinary audio.

    Both are fixed by requiring F, 2F and 3F to be elevated *together* -- real
    comb structure repeats, noise does not -- and by capping the search at
    400 Hz, above which no plausible vocoder frame rate lives.
    """
    fr, res, sr = residual_spectrum(path, fmin=fmin, fmax=fmax)
    binw = fr[1] - fr[0]
    spacings = np.arange(lo, hi, step)

    # Tolerance scales with spacing so the on-grid fraction stays constant.
    # A fixed 2 Hz window is 1 % of a 200 Hz comb but 8 % of a 25 Hz one, which
    # would quietly destroy contrast at the low end.
    def tol_for(s):
        return max(binw * 1.5, s * 0.01)

    scores, base = [], []
    for s in spacings:
        tol = tol_for(s)
        rs = [comb_ratio(fr, res, s * k, tol) for k in (1, 2, 3)]
        rs = [r for r in rs if np.isfinite(r) and r > 0]
        if len(rs) < 2:
            scores.append(np.nan); base.append(np.nan); continue
        scores.append(float(np.exp(np.mean(np.log(rs)))))   # geometric mean
        base.append(rs[0])
    scores = np.array(scores); base = np.array(base)
    ok = np.isfinite(scores)
    if not ok.any():
        return None
    spacings, scores, base = spacings[ok], scores[ok], base[ok]

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) + 1e-12
    z_all = (scores - med) / (1.4826 * mad)

    top = float(scores.max())
    near = np.where(scores >= top * 0.85)[0]
    # Prefer the smallest spacing near the top, but only if its ODD multiples
    # are also elevated. Without that gate every genuine comb at F is reported
    # as F/2, F/4 ... since those inherit the score from their even multiples.
    solid = []
    for j in near:
        r_odd = comb_ratio(fr, res, spacings[j], tol_for(spacings[j]),
                           odd_only=True)
        if np.isfinite(r_odd) and r_odd >= 1.15:
            solid.append(j)
    i = int(solid[0] if solid else near[int(np.argmax(scores[near]))])

    harm = [comb_ratio(fr, res, spacings[i] * k, tol_for(spacings[i] * k))
            for k in (2, 3)]
    harm = [h for h in harm if np.isfinite(h)]
    return dict(sr=sr, spacing=float(spacings[i]), ratio=float(base[i]),
                score=float(scores[i]), z=float(z_all[i]), median=med,
                harmonics=harm)


def decode(spacing):
    """Which (sample rate, hop) pairs would give this frame rate?"""
    out = []
    for r in COMMON_RATES:
        h = r / spacing
        if abs(h - round(h)) < 0.02 and 32 <= round(h) <= 2048:
            out.append(f"{round(h)} @ {r//1000}k")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--fmin", type=float, default=1000.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--z", type=float, default=6.0,
                    help="z-score threshold to call a comb (default 6)")
    ap.add_argument("--lo", type=float, default=40.0,
                    help="lowest spacing to search, Hz (default 40; lower it "
                         "for VAE-style codecs whose frame rate is ~25 Hz)")
    ap.add_argument("--hi", type=float, default=400.0,
                    help="highest spacing to search, Hz (default 400)")
    args = ap.parse_args()

    print(f"{'file':<30} {'spacing':>9} {'ratio':>7} {'z':>7} {'2f/3f':>12}  verdict")
    print("-" * 88)
    for f in args.files:
        try:
            r = analyse(f, args.fmin, args.fmax, lo=args.lo, hi=args.hi)
        except Exception as e:
            print(f"{os.path.basename(f)[:30]:<30} {'error':>9}  "
                  f"{type(e).__name__}")
            continue
        if r is None:
            continue
        h = "/".join(f"{x:.2f}" for x in r["harmonics"]) or "-"
        strong = (r["z"] >= args.z and r["score"] > 1.3
                  and all(x > 1.15 for x in r["harmonics"]))
        verdict = "VOCODER COMB" if strong else "none"
        extra = ""
        if strong:
            d = decode(r["spacing"])
            if d:
                extra = "  hop " + ", ".join(d)
        print(f"{os.path.basename(f)[:30]:<30} {r['spacing']:>8.2f}H "
              f"{r['score']:>7.2f} {r['z']:>7.1f} {h:>12}  {verdict}{extra}")
    print()
    print("score = geometric mean of the ratio at F, 2F and 3F (a real comb")
    print("        repeats; noise does not).  z = above the sweep median in MADs.")


if __name__ == "__main__":
    sys.exit(main())
