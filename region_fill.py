"""Splice a restored file into an original over chosen time-frequency regions.

`band_splice.py` already does the frequency-domain version of this, cleanly, but
it is whole-file and frequency-only: one crossover, applied for the entire
duration. This generalises it to rectangles in the time-frequency plane, so you
can take the model's high band only where the source actually needs it -- a
lossy section, a dull chorus -- and leave everything else untouched.

How the untouched part stays untouched:

    hybrid = orig + ISTFT( mask * (STFT(restored) - STFT(orig)) )

The spliced-in quantity is a *correction*, not a replacement. ISTFT is linear,
so wherever the mask is zero across every overlapping frame the correction is
exactly 0.0 and `orig + 0.0` is bit-identical -- no STFT round-trip error is
introduced outside the regions at all. The tool prints that as a measurement
rather than asking you to trust it.

What this is NOT: inpainting. Regions are filled *from the upscaler's rendition
of that region*, so this means "apply the restoration only here". A true dropout
-- actual silence in the source -- has nothing in the restored file to splice
from, and none of the installed tools invent content from nothing.

Usage:
    # default region: detected cliff -> Nyquist, whole file (= band_splice)
    python region_fill.py --orig in.wav --restored sr.wav -o hybrid.wav

    # explicit rectangles, repeatable: t0:t1:flo:fhi  ("" or * = open end)
    python region_fill.py --orig in.wav --restored sr.wav -o hybrid.wav \
        --region 0:30:16000:* --region 45:60:12000:18000
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, resample_poly

N_FFT = 2048
HOP = N_FFT // 4          # 75 % overlap: Hann satisfies COLA here


def detect_cliff(mono, sr, n=8192):
    """Steepest high-frequency rolloff -- same method as check_bandwidth.py."""
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    count = 0
    step = max(n, (len(mono) - n) // 400) if len(mono) > n else n
    for start in range(0, max(1, len(mono) - n), step):
        acc += np.abs(np.fft.rfft(mono[start:start + n] * win))
        count += 1
    if count:
        acc /= count
    fr = np.fft.rfftfreq(n, 1 / sr)
    db = 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)
    sm = np.convolve(db, np.ones(9) / 9, mode="same")
    d = np.diff(sm)
    searchable = fr[:-1] > 6000
    if not searchable.any():
        return sr / 2 * 0.9, 0.0
    i = int(np.argmin(np.where(searchable, d, 0)))
    cliff = float(fr[i])
    lo = db[(fr >= cliff - 1500) & (fr < cliff)]
    hi = db[(fr > cliff) & (fr <= cliff + 1500)]
    drop = float(lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0
    return cliff, drop


def parse_region(spec, dur, nyquist):
    """'t0:t1:flo:fhi' -> tuple of floats. '' or '*' means the open end."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"region must be t0:t1:flo:fhi, got {spec!r}")

    def val(s, default):
        s = s.strip()
        return default if s in ("", "*") else float(s)

    t0, t1 = val(parts[0], 0.0), val(parts[1], dur)
    flo, fhi = val(parts[2], 0.0), val(parts[3], nyquist)
    if t1 <= t0 or fhi <= flo:
        raise ValueError(f"empty region {spec!r}")
    return t0, t1, flo, fhi


def _ramp(axis_vals, lo, hi, taper):
    """Raised-cosine window over `axis_vals`, 0 outside [lo, hi].

    A hard rectangular mask rings: a step in frequency smears in time and a step
    in time clicks. Tapering both edges is what keeps the seam inaudible.
    """
    w = np.zeros_like(axis_vals, dtype=float)
    inside = (axis_vals >= lo) & (axis_vals <= hi)
    w[inside] = 1.0
    if taper > 0:
        rise = (axis_vals >= lo - taper) & (axis_vals < lo)
        if rise.any():
            t = (axis_vals[rise] - (lo - taper)) / taper
            w[rise] = 0.5 - 0.5 * np.cos(np.pi * t)
        fall = (axis_vals > hi) & (axis_vals <= hi + taper)
        if fall.any():
            t = (axis_vals[fall] - hi) / taper
            w[fall] = 0.5 + 0.5 * np.cos(np.pi * t)
    return w


def build_mask(freqs, times, regions, f_taper, t_taper):
    mask = np.zeros((len(freqs), len(times)))
    for t0, t1, flo, fhi in regions:
        fw = _ramp(freqs, flo, fhi, f_taper)
        tw = _ramp(times, t0, t1, t_taper)
        mask = np.maximum(mask, np.outer(fw, tw))
    return mask


def load_pair(orig_path, restored_path):
    o, sr_o = sf.read(orig_path, always_2d=True, dtype="float64")
    r, sr_r = sf.read(restored_path, always_2d=True, dtype="float64")

    if sr_r != sr_o:
        # The correction has to live in the ORIGINAL's domain, otherwise the
        # untouched samples cannot stay bit-exact. Upscalers routinely change
        # rate (AudioSR/FlashSR emit 48 kHz; Apollo forces 44.1), so this is
        # the common case, not an edge case.
        from fractions import Fraction
        f = Fraction(sr_o, sr_r)
        r = resample_poly(r, f.numerator, f.denominator, axis=0)
        print(f"  resampled restored {sr_r} -> {sr_o} Hz")

    if r.shape[1] != o.shape[1]:
        r = (r.mean(axis=1, keepdims=True) if o.shape[1] == 1
             else np.repeat(r[:, :1], o.shape[1], axis=1))

    dn = abs(len(r) - len(o)) / sr_o
    if dn > 0.05:
        print(f"  WARNING: lengths differ by {dn*1000:.0f} ms -- a time-shifted "
              f"restored file comb-filters at the seams")
    n = min(len(o), len(r))
    return o[:n], r[:n], sr_o


def region_fill(orig, restored, sr, regions, f_taper=500.0, t_taper=0.05):
    """Return (hybrid, correction). Correction is exactly 0 outside the mask."""
    corr = np.zeros_like(orig)
    freqs = times = None
    for c in range(orig.shape[1]):
        f, t, So = stft(orig[:, c], fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                        window="hann", boundary="zeros", padded=True)
        _, _, Sr = stft(restored[:, c], fs=sr, nperseg=N_FFT,
                        noverlap=N_FFT - HOP, window="hann",
                        boundary="zeros", padded=True)
        if freqs is None:
            freqs, times = f, t
            mask = build_mask(f, t, regions, f_taper, t_taper)
        _, y = istft(mask * (Sr - So), fs=sr, nperseg=N_FFT,
                     noverlap=N_FFT - HOP, window="hann", boundary=True)
        m = min(len(y), len(corr))
        corr[:m, c] = y[:m]
    return orig + corr, corr


def verify(orig, hybrid, sr, regions, t_taper):
    """Prove the claim instead of asserting it."""
    n = len(orig)
    touched = np.zeros(n, dtype=bool)
    for t0, t1, _, _ in regions:
        a = max(0, int((t0 - t_taper) * sr) - N_FFT)
        b = min(n, int((t1 + t_taper) * sr) + N_FFT)
        touched[a:b] = True
    d = np.abs(hybrid - orig)
    outside = (~touched).sum()
    print(f"  samples outside any region : {outside} "
          f"({100*outside/max(n,1):.1f}%)")
    if outside:
        print(f"  max |hybrid-orig| there    : {d[~touched].max():.3e} "
              f"(0.0 means bit-identical)")
    if touched.any():
        print(f"  max |hybrid-orig| in region: {d[touched].max():.3e}")
    rms = np.sqrt((d ** 2).mean())
    print(f"  correction RMS overall     : "
          f"{20*np.log10(rms + 1e-20):.1f} dBFS")

    # The point of splicing rather than replacing: content BELOW the crossover
    # must not move, even inside a touched region. Whole-file replacement by an
    # upscaler measurably degrades that band (12-19% relative error, per
    # band_splice.py's notes) -- this is the number that shows it did not.
    flo_min = min(r[2] for r in regions)
    if flo_min > 0 and touched.any():
        seg_o = orig[touched].mean(axis=1)
        seg_h = hybrid[touched].mean(axis=1)
        m = min(len(seg_o), 1 << 20)
        if m > 1024:
            w = np.hanning(m)
            fr = np.fft.rfftfreq(m, 1 / sr)
            So = np.abs(np.fft.rfft(seg_o[:m] * w))
            Sh = np.abs(np.fft.rfft(seg_h[:m] * w))
            below = fr < flo_min
            num = np.linalg.norm(Sh[below] - So[below])
            den = np.linalg.norm(So[below]) + 1e-20
            print(f"  drift below {flo_min:.0f} Hz in region: "
                  f"{100*num/den:.4f}% relative "
                  f"({20*np.log10(num/den + 1e-20):.1f} dB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--restored", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--region", action="append", default=None,
                    help="t0:t1:flo:fhi, repeatable; '' or '*' = open end")
    ap.add_argument("--f-taper", type=float, default=500.0,
                    help="Hz of raised-cosine taper at band edges")
    ap.add_argument("--t-taper", type=float, default=0.05,
                    help="seconds of taper at region start/end")
    args = ap.parse_args()

    print(f"orig     : {os.path.basename(args.orig)}")
    print(f"restored : {os.path.basename(args.restored)}")
    orig, restored, sr = load_pair(args.orig, args.restored)
    dur, nyq = len(orig) / sr, sr / 2

    if args.region:
        regions = [parse_region(s, dur, nyq) for s in args.region]
    else:
        cliff, drop = detect_cliff(orig.mean(axis=1), sr)
        regions = [(0.0, dur, cliff, nyq)]
        print(f"  no --region given; using detected cliff "
              f"{cliff:.0f} Hz (drop {drop:.1f} dB) -> Nyquist, whole file")

    for t0, t1, flo, fhi in regions:
        print(f"  region: {t0:.2f}-{t1:.2f}s  {flo:.0f}-{fhi:.0f} Hz")

    hybrid, _ = region_fill(orig, restored, sr, regions,
                            args.f_taper, args.t_taper)
    peak = np.abs(hybrid).max()
    if peak > 1.0:
        print(f"  WARNING: peak {peak:.3f} > 1.0, scaling to avoid clipping")
        hybrid = hybrid / peak

    # Match the original's subtype, or the bit-exact claim dies at write time.
    sub = sf.info(args.orig).subtype
    sf.write(args.output, hybrid, sr, subtype=sub)
    verify(orig, hybrid, sr, regions, args.t_taper)
    print(f"  wrote {args.output}  ({sr} Hz, {sub})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
