"""
Report a file's real bandwidth and whether upresing is worth it.

Finds the steepest high-frequency rolloff — a lossy codec leaves a hard cliff,
an unprocessed recording just fades out. Run this before spending GPU time.

Usage:
    python check_bandwidth.py "track.wav" ["another.flac" ...]
"""

import subprocess
import sys
import tempfile
import os

import numpy as np
import soundfile as sf


def load(path):
    """Read via soundfile, falling back to ffmpeg for mp3/aac/etc."""
    try:
        x, sr = sf.read(path, always_2d=True)
        return x, sr
    except Exception:
        tmp = os.path.join(tempfile.gettempdir(),
                           f"bwchk_{os.getpid()}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-c:a", "pcm_s16le", tmp],
            check=True,
        )
        try:
            x, sr = sf.read(tmp, always_2d=True)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return x, sr


def average_spectrum(mono, sr, n=8192, windows=400):
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    count = 0
    step = max(n, (len(mono) - n) // windows) if len(mono) > n else n
    for start in range(0, max(1, len(mono) - n), step):
        acc += np.abs(np.fft.rfft(mono[start:start + n] * win))
        count += 1
    if count:
        acc /= count
    return np.fft.rfftfreq(n, 1 / sr), 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)


def verdict(cliff_hz, drop_db):
    if drop_db < 8:
        return ("no hard cliff -- likely lossless, or the recording simply has "
                "no HF content. Nothing to restore.")
    if cliff_hz < 13000:
        return "low-bitrate lossy. Big audible win from Apollo."
    if cliff_hz < 16500:
        return "~128 kbps lossy. Audible win from Apollo."
    if cliff_hz < 19500:
        return "~192-256 kbps. Marginal -- near the edge of hearing."
    return ("~320 kbps / 256k AAC. The cliff is already above what you can "
            "hear; processing costs more than it gains.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    for path in sys.argv[1:]:
        x, sr = load(path)
        mono = x.mean(axis=1)
        fr, db = average_spectrum(mono, sr)

        # Smooth, then find the steepest drop above 6 kHz.
        sm = np.convolve(db, np.ones(9) / 9, mode="same")
        d = np.diff(sm)
        searchable = fr[:-1] > 6000
        if not searchable.any():
            print(f"{path}: sample rate too low to analyse")
            continue
        i = int(np.argmin(np.where(searchable, d, 0)))
        cliff = fr[i]

        # Level 1 kHz below vs 1 kHz above the cliff = how hard the cliff is.
        lo = db[(fr >= cliff - 1500) & (fr < cliff)]
        hi = db[(fr > cliff) & (fr <= cliff + 1500)]
        drop = (lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0

        print(f"\n{os.path.basename(path)}")
        print(f"  {sr} Hz, {x.shape[1]} ch, {len(x)/sr:.1f}s")
        print(f"  cliff at ~{cliff:.0f} Hz  (drop {drop:.1f} dB across it)")
        print(f"  -> {verdict(cliff, drop)}")

        if sr == 48000:
            print("  note: Apollo would resample this to 44.1 kHz. "
                  "Use AudioSR to keep 48 kHz.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
