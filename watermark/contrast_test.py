"""
Causal test: is the classifier essentially a spectral-peakiness meter?

Across 14 files, one scalar -- the mean spectral residual (how far the spectrum
rises above its own local minimum floor) -- separated every flagged file from
every clean one. That is a correlation. This tests it causally by manipulating
the residual directly and nothing else.

Per STFT frame: measure the local floor with the same minimum filter the
detector uses, then rescale the distance above it by k, and resynthesise with
the original phase.

    k > 1  exaggerate spectral peaks  (should push toward "real")
    k < 1  flatten toward the floor   (should push toward "AI")

If the verdict tracks k in both directions, the mechanism is established. Note
this is a gross distortion, not a practical process -- SNR against the source is
reported so the damage is visible alongside any verdict change.

Usage:
    python contrast_test.py --ai suno.wav --control real.wav
"""

import argparse
import os
import sys
import tempfile

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector

N_FFT = 4096
HOP = 1024


def rescale_contrast(x, sr, k):
    """Scale each frame's spectrum distance above its local floor by k."""
    import librosa
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        S = librosa.stft(x[:, c], n_fft=N_FFT, hop_length=HOP)
        mag, phase = np.abs(S), np.angle(S)
        db = 20 * np.log10(mag + 1e-10)
        floor = minimum_filter1d(db, size=10, axis=0, mode="nearest")
        new_db = floor + (db - floor) * k
        new_mag = 10 ** (new_db / 20)
        y = librosa.istft(new_mag * np.exp(1j * phase), hop_length=HOP,
                          length=len(x))
        out[:, c] = y
    peak = np.abs(out).max()
    if peak > 0.999:
        out = out / peak * 0.999
    return out


def snr(ref, test):
    n = min(len(ref), len(test))
    a, b = ref[:n].mean(axis=1), test[:n].mean(axis=1)
    g = (a @ b) / (b @ b + 1e-20)
    d = a - g * b
    return 10 * np.log10((a ** 2).mean() / ((d ** 2).mean() + 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()

    det = FakeprintDetector(args.model)
    tmp = tempfile.mkdtemp(prefix="contrast_")

    srcs = {}
    for name, path in (("SUNO (flagged)", args.ai),
                       ("control (clean)", args.control)):
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        srcs[name] = (x[:30 * sr], sr)

    def score(y, sr):
        p = os.path.join(tmp, "t.wav")
        sf.write(p, y, sr, subtype="PCM_24")
        try:
            prob, _ = det.predict(p)
            fp, _ = det.fakeprint(p)
        finally:
            os.remove(p)
        return prob, fp.mean()

    print(f"{'file':<18} {'k':>5} {'p(AI)':>9} {'mean resid':>11} {'SNR dB':>8}")
    print("-" * 58)
    for name, (x, sr) in srcs.items():
        base_p, _ = score(x, sr)
        for k in (0.3, 0.6, 1.0, 1.6, 2.5, 4.0):
            y = x if k == 1.0 else rescale_contrast(x, sr, k)
            prob, meanfp = score(y, sr)
            s = snr(x, y)
            tag = "  <-- FLIPPED" if (prob >= 0.5) != (base_p >= 0.5) else ""
            sn = "    inf" if k == 1.0 else f"{s:>8.1f}"
            print(f"{name:<18} {k:>5.1f} {prob:>9.4f} {meanfp:>11.3f} "
                  f"{sn}{tag}")
        print()
    os.rmdir(tmp)


if __name__ == "__main__":
    main()
