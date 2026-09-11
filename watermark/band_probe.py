"""
Where in the spectrum does the AI signature actually live?

Two probes, both done as exact FFT-domain masks:

  ISOLATE  keep one band, zero everything else
  ABLATE   zero one band, keep everything else

Critical confound: the classifier's bias is +4.98, so an all-zero feature
vector already scores p=0.993 "AI". Emptying most of the spectrum therefore
pushes toward AI no matter what the source was. Absolute p on an isolated band
is meaningless on its own -- every band is run against a known non-AI control
of the same treatment, and only the DIFFERENCE carries information.

Sanity checks built in: the classifier only sees 1-8 kHz of a 16 kHz downmix,
so isolating 8-24 kHz must collapse to the empty-vector score for both files,
and ablating it must change nothing.

Usage:
    python band_probe.py --ai suno.wav --control real.wav -o outdir
"""

import argparse
import os
import tempfile

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector

BANDS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000),
         (5000, 6000), (6000, 7000), (7000, 8000), (8000, 24000)]


def mask_audio(x, sr, lo, hi, keep=True):
    """Exact frequency-domain band isolate (keep=True) or ablate (keep=False)."""
    n = len(x)
    out = np.empty_like(x)
    fr = np.fft.rfftfreq(n, 1 / sr)
    sel = (fr >= lo) & (fr < hi)
    for c in range(x.shape[1]):
        X = np.fft.rfft(x[:, c])
        X = np.where(sel, X, 0) if keep else np.where(sel, 0, X)
        out[:, c] = np.fft.irfft(X, n=n)
    return out


def score(det, audio, sr, tmpdir, tag):
    p = os.path.join(tmpdir, f"{tag}.wav")
    peak = np.abs(audio).max()
    a = audio / peak * 0.9 if peak > 1e-9 else audio
    sf.write(p, a, sr, subtype="PCM_24")
    try:
        prob, _ = det.predict(p)
    except Exception:
        prob = float("nan")
    os.remove(p)
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True, help="known AI-generated file")
    ap.add_argument("--control", required=True, help="known non-AI file")
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()

    det = FakeprintDetector(args.model)
    tmp = tempfile.mkdtemp(prefix="bandprobe_")

    files = {}
    for name, path in (("AI", args.ai), ("control", args.control)):
        x, sr = sf.read(path, always_2d=True, dtype="float64")
        n = min(len(x), 30 * sr)
        files[name] = (x[:n], sr)

    base = {k: score(det, v[0], v[1], tmp, f"base_{k}")
            for k, v in files.items()}
    empty = det.sess.run(None, {det.input_name:
                                np.zeros((1, det.n_features), np.float32)})[0][0, 0]

    print(f"reference points")
    print(f"  full file        AI={base['AI']:.4f}   control={base['control']:.4f}")
    print(f"  all-zero features -> {float(empty):.4f}  "
          f"(what an empty band collapses to)")
    print()
    print(f"{'band':<14} {'ISOLATE: AI':>12} {'control':>9} {'gap':>8}   "
          f"{'ABLATE: AI':>11} {'control':>9}")
    print("-" * 74)

    for lo, hi in BANDS:
        row = {}
        for mode, keep in (("iso", True), ("abl", False)):
            for k, (x, sr) in files.items():
                y = mask_audio(x, sr, lo, hi, keep=keep)
                row[(mode, k)] = score(det, y, sr, tmp, f"{mode}_{k}_{lo}")
        gap = row[("iso", "AI")] - row[("iso", "control")]
        label = f"{lo//1000}-{hi//1000}k"
        print(f"{label:<14} {row[('iso','AI')]:>12.4f} "
              f"{row[('iso','control')]:>9.4f} {gap:>+8.4f}   "
              f"{row[('abl','AI')]:>11.4f} {row[('abl','control')]:>9.4f}")

    os.rmdir(tmp)
    print()
    print("ISOLATE gap  = how much this band ALONE distinguishes AI from real.")
    print("ABLATE AI    = score with this band REMOVED; a drop means the band")
    print("               was carrying the evidence.")


if __name__ == "__main__":
    main()
