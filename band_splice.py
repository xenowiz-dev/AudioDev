"""
Surgical high-band splice: keep the original untouched, add ONLY the
reconstructed band above a chosen frequency.

Motivation: super-resolution models rewrite the whole signal, and on a source
that is already full-bandwidth up to its codec cliff they measurably *degrade*
the audible band while adding content above it. Splicing in the frequency domain
avoids that -- every bin below the crossover comes bit-for-bit from the original,
and only the bins above it come from the model.

Also emits the isolated high band on its own, and a slowed copy of it. Content
above 20 kHz is inaudible to humans, so slowing it 4x (by declaring a quarter
sample rate) drops 20-24 kHz to 5-6 kHz, where you can actually hear what the
model invented -- musical overtones vs. hiss vs. nothing.

Usage:
    python band_splice.py --orig ref.wav --restored sr.wav --cut 20010 -o outdir
"""

import argparse
import os

import numpy as np
import soundfile as sf


def rfft_cols(x):
    return np.stack([np.fft.rfft(x[:, c]) for c in range(x.shape[1])], axis=1)


def irfft_cols(X, n):
    return np.stack([np.fft.irfft(X[:, c], n=n) for c in range(X.shape[1])],
                    axis=1)


def band_energy_db(x, sr, lo, hi):
    n = min(len(x), 1 << 20)
    m = x[:n].mean(axis=1) if x.ndim > 1 else x[:n]
    S = np.abs(np.fft.rfft(m * np.hanning(len(m))))
    fr = np.fft.rfftfreq(len(m), 1 / sr)
    sel = (fr >= lo) & (fr < hi)
    if not sel.any() or S.max() == 0:
        return float("-inf")
    return 20 * np.log10(S[sel].mean() / S.max() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--restored", required=True)
    ap.add_argument("--cut", type=float, default=20010.0)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--slow", type=int, default=4,
                    help="slow-down factor for the audible rendering")
    args = ap.parse_args()

    a, sr_a = sf.read(args.orig, always_2d=True, dtype="float64")
    b, sr_b = sf.read(args.restored, always_2d=True, dtype="float64")
    if sr_a != sr_b:
        raise SystemExit(f"sample rates differ ({sr_a} vs {sr_b}); "
                         "resample first or use a 48 kHz-native model")
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.shape[1] != b.shape[1]:
        raise SystemExit("channel count differs")

    os.makedirs(args.outdir, exist_ok=True)

    A = rfft_cols(a)
    B = rfft_cols(b)
    fr = np.fft.rfftfreq(n, 1 / sr_a)
    hi = (fr >= args.cut)[:, None]

    # Hard bin split. A crossfade would blend two uncorrelated spectra and
    # partially cancel; a hard edge at 20 kHz rings only where nobody can hear.
    hybrid = irfft_cols(np.where(hi, B, A), n)
    band = irfft_cols(np.where(hi, B, 0), n)

    outs = {}
    p = os.path.join(args.outdir, f"hybrid_{args.tag}.wav")
    sf.write(p, hybrid, sr_a, subtype="PCM_24"); outs["hybrid"] = p

    p = os.path.join(args.outdir, f"band_only_{args.tag}.wav")
    sf.write(p, band, sr_a, subtype="PCM_24"); outs["band only"] = p

    # Declaring a lower sample rate IS the slow-down -- no resampling, no
    # interpolation artefacts, exactly like running tape slow.
    p = os.path.join(args.outdir, f"band_slowed{args.slow}x_{args.tag}.wav")
    sf.write(p, band, max(1000, int(sr_a / args.slow)), subtype="PCM_24")
    outs[f"band slowed {args.slow}x"] = p

    # Verify the audible band survived, by re-analysing the file we actually
    # wrote rather than the arrays we built it from -- that catches indexing
    # and round-trip bugs instead of restating the construction.
    written, _ = sf.read(outs["hybrid"], always_2d=True, dtype="float64")
    W = rfft_cols(written[:n])
    low_orig = irfft_cols(np.where(~hi, A, 0), n)
    low_written = irfft_cols(np.where(~hi, W, 0), n)
    drift = np.abs(low_orig - low_written).max()

    # How much the model WOULD have altered below the crossover, had we let it.
    num = np.linalg.norm(np.where(~hi, A - B, 0))
    den = np.linalg.norm(np.where(~hi, A, 0))
    avoided = 20 * np.log10(num / den + 1e-15)

    print(f"crossover: {args.cut:.0f} Hz   ({sr_a} Hz, {a.shape[1]} ch, "
          f"{n/sr_a:.2f}s)")
    print(f"  audible band drift vs original: {drift:.2e} "
          f"(24-bit LSB is 6e-08)")
    print(f"  damage avoided below crossover: {avoided:+.1f} dB "
          f"relative error the model would have introduced")
    for k, v in outs.items():
        print(f"  {k:<20} {os.path.basename(v)}")

    print(f"  band {args.cut/1000:.0f}-24 kHz level: "
          f"orig {band_energy_db(a, sr_a, args.cut, 24000):.1f} dB   "
          f"model {band_energy_db(b, sr_a, args.cut, 24000):.1f} dB")
    rms = np.sqrt((band ** 2).mean())
    print(f"  isolated band RMS: {20*np.log10(rms+1e-12):.1f} dBFS")


if __name__ == "__main__":
    main()
