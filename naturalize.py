"""naturalize.py -- the "make it sound like it came off a desk" pass.

Neural generators produce audio that is technically clean and characteristically
lifeless: a dead-silent noise floor, symmetric waveforms with no harmonic
colour, subsonic wander from the vocoder, and whatever loudness the model felt
like. None of that is a bug in the model; it is the absence of the analogue
chain every real recording passed through.

This restores four of those things, in the order a mastering chain would:

  1. HIGHPASS   -- 24 Hz, 2nd order. Removes DC offset and subsonic wander,
                   which eat headroom you cannot hear.
  2. SATURATE   -- soft asymmetric tanh, oversampled 4x. Generates harmonics
                   the way tape and tubes do: the asymmetry is what produces
                   EVEN harmonics (2nd, 4th), which is the "warm" part; a
                   symmetric curve gives only odd harmonics and sounds harsh.
                   Oversampling matters -- saturating at 44.1 kHz folds the new
                   harmonics of a 15 kHz cymbal back down as inharmonic
                   aliasing, which is exactly the brittle digital sound this is
                   meant to remove.
  3. NOISE      -- pink noise at a chosen dBFS. A real noise floor. Absolute
                   silence between notes is one of the strongest unconscious
                   tells that something was synthesised; -65 dBFS is below
                   audibility on its own but fills the gaps.
  4. LOUDNESS   -- ITU-R BS.1770 (pyloudnorm) to a target LUFS, then a peak
                   ceiling so nothing clips.

Everything is optional and everything is gentle by default. The point is a
change you notice only in an A/B, not an effect.

    python naturalize.py -i in.wav -o out.wav --strength medium --json
"""

import argparse
import json
import sys

import numpy as np
import soundfile as sf
from scipy import signal

# drive_db, wet, bias, noise_dbfs
STRENGTHS = {
    "subtle": (2.0, 0.35, 0.02, -72.0),
    "medium": (4.0, 0.55, 0.035, -66.0),
    "strong": (7.0, 0.75, 0.05, -60.0),
}

OVERSAMPLE = 4

# Paul Kellet's pink-noise filter: white noise through this is 1/f to within
# about 0.3 dB from 10 Hz up. Cheaper and better behaved than an FFT method.
PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


def _as_2d(x):
    """(n,) or (n, ch) -> (n, ch). Keeps the caller's channel count."""
    return x[:, None] if x.ndim == 1 else x


def highpass(x, sr, hz=24.0):
    """2nd-order Butterworth, zero-phase. DC and subsonics only."""
    sos = signal.butter(2, hz / (sr / 2.0), btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def saturate(x, sr, drive_db=4.0, wet=0.55, bias=0.035):
    """Asymmetric soft clip, oversampled so the harmonics stay harmonic."""
    if wet <= 0:
        return x
    try:
        import soxr
        up = soxr.resample(x, sr, sr * OVERSAMPLE, quality="VHQ")
    except Exception:
        up, sr_up = x, sr          # no soxr: still works, just aliases more
    else:
        sr_up = sr * OVERSAMPLE

    g = 10.0 ** (drive_db / 20.0)
    # Subtracting tanh(g*bias) removes the DC the offset would otherwise add;
    # dividing by tanh(g) keeps unity gain at full scale so `wet` is the only
    # thing changing level.
    shaped = (np.tanh(g * (up + bias)) - np.tanh(g * bias)) / np.tanh(g)
    up = (1.0 - wet) * up + wet * shaped

    if sr_up != sr:
        up = soxr.resample(up, sr_up, sr, quality="VHQ")
    return up[:len(x)] if len(up) >= len(x) else np.pad(
        up, ((0, len(x) - len(up)), (0, 0)))


def pink_noise(n, ch, rng):
    w = rng.standard_normal((n + 2048, ch))
    p = signal.lfilter(PINK_B, PINK_A, w, axis=0)[2048:]   # drop the transient
    rms = np.sqrt(np.mean(p ** 2, axis=0, keepdims=True))
    return p / np.maximum(rms, 1e-12)                       # unit RMS per ch


def add_noise(x, dbfs, seed=7):
    """A pink floor at `dbfs` RMS. dbfs >= 0 or None disables it."""
    if dbfs is None or dbfs >= 0:
        return x
    rng = np.random.default_rng(seed)
    return x + pink_noise(len(x), x.shape[1], rng) * (10.0 ** (dbfs / 20.0))


def loudness(x, sr):
    import pyloudnorm as pyln
    return float(pyln.Meter(sr).integrated_loudness(x))


def normalize(x, sr, target_lufs=-14.0, ceiling_db=-1.0):
    """Loudness-match, then guarantee the peak ceiling. Returns (y, before)."""
    before = loudness(x, sr)
    if not np.isfinite(before):
        return x, before
    y = x * (10.0 ** ((target_lufs - before) / 20.0))
    peak = float(np.max(np.abs(y))) or 1.0
    limit = 10.0 ** (ceiling_db / 20.0)
    if peak > limit:
        # Plain gain, not a limiter: a limiter would change the dynamics this
        # whole pass is trying to preserve. Losing 0.5 LU beats pumping.
        y *= limit / peak
    return y, before


def measure(x, sr):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return {
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 2),
        "rms_dbfs": round(20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))),
                                            1e-12)), 2),
        "dc_offset": round(float(np.mean(x)), 6),
        "lufs": round(loudness(x, sr), 2),
    }


def naturalize(src, out, strength="medium", drive_db=None, wet=None,
               noise_dbfs=None, target_lufs=-14.0, do_highpass=True,
               seed=7, log=print):
    d_def, w_def, bias, n_def = STRENGTHS[strength]
    drive_db = d_def if drive_db is None else drive_db
    wet = w_def if wet is None else wet
    noise_dbfs = n_def if noise_dbfs is None else noise_dbfs

    x, sr = sf.read(src, always_2d=True, dtype="float64")
    before = measure(x, sr)
    log(f"in : {before['lufs']} LUFS, peak {before['peak_dbfs']} dBFS, "
        f"DC {before['dc_offset']:+.5f}")

    if do_highpass:
        x = highpass(x, sr)
        log("highpass 24 Hz (DC and subsonic wander)")
    if wet > 0:
        x = saturate(x, sr, drive_db, wet, bias)
        log(f"saturation {drive_db:.1f} dB drive, {int(wet * 100)}% wet, "
            f"{OVERSAMPLE}x oversampled")
        # An asymmetric curve rectifies, so it MAKES DC out of real programme
        # material -- the tanh(g*bias) term only cancels the offset at silence.
        # Measured: without this, DC came out 30x worse than it went in.
        if do_highpass:
            x = highpass(x, sr)
    if noise_dbfs is not None and noise_dbfs < 0:
        x = add_noise(x, noise_dbfs, seed)
        log(f"pink noise floor at {noise_dbfs:.0f} dBFS")
    if target_lufs is not None:
        x, meas = normalize(x, sr, target_lufs)
        log(f"loudness {meas:.1f} -> {target_lufs:.1f} LUFS")

    after = measure(x, sr)
    log(f"out: {after['lufs']} LUFS, peak {after['peak_dbfs']} dBFS, "
        f"DC {after['dc_offset']:+.5f}")

    sub = "PCM_24" if str(src).lower().endswith((".wav", ".flac")) else "PCM_16"
    sf.write(out, x, sr, subtype=sub)
    return {"out": out, "sr": sr, "channels": x.shape[1],
            "strength": strength, "drive_db": drive_db, "wet": wet,
            "noise_dbfs": noise_dbfs, "target_lufs": target_lufs,
            "before": before, "after": after}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--strength", choices=list(STRENGTHS), default="medium")
    ap.add_argument("--drive-db", type=float, default=None)
    ap.add_argument("--wet", type=float, default=None)
    ap.add_argument("--noise-dbfs", type=float, default=None,
                    help="0 or above disables the noise floor")
    ap.add_argument("--lufs", type=float, default=-14.0,
                    help="target loudness; pass 0 to leave the level alone")
    ap.add_argument("--no-highpass", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # stderr for progress, stdout for the one JSON line -- the studio reads
    # them separately (same contract as transcribe.py).
    res = naturalize(a.input, a.output, strength=a.strength,
                     drive_db=a.drive_db, wet=a.wet, noise_dbfs=a.noise_dbfs,
                     target_lufs=(None if a.lufs == 0 else a.lufs),
                     do_highpass=not a.no_highpass, seed=a.seed,
                     log=lambda m: print(m, file=sys.stderr, flush=True))
    if a.json:
        print(json.dumps(res))


if __name__ == "__main__":
    main()
