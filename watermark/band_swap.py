"""
Band-swap probe: which frequency range actually carries the AI signature?

Isolating a band and zeroing the rest is confounded -- an empty feature vector
scores 0.993 "AI" from the bias alone, so narrow bands collapse toward "AI"
regardless of content. This avoids that entirely by keeping the spectrum full
at all times and only EXCHANGING one band between two files of known, opposite
provenance:

    graft   control audio, with band B replaced by the AI track's band B
            -> if p rises, band B carries the AI evidence
    heal    AI audio, with band B replaced by the control's band B
            -> if p falls, band B carries the AI evidence

Both files keep full-spectrum content, so the classifier always sees a
populated 3585-dim vector and the bias never dominates.

Usage:
    python band_swap.py --ai suno.wav --control real.wav
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
         (5000, 6000), (6000, 7000), (7000, 8000), (8000, 12000),
         (12000, 24000)]


def graft(dst, src, sr, lo, hi):
    """Return dst with the band [lo,hi) replaced by src's."""
    n = len(dst)
    fr = np.fft.rfftfreq(n, 1 / sr)
    sel = (fr >= lo) & (fr < hi)
    out = np.empty_like(dst)
    for c in range(dst.shape[1]):
        D = np.fft.rfft(dst[:, c])
        S = np.fft.rfft(src[:, c])
        out[:, c] = np.fft.irfft(np.where(sel, S, D), n=n)
    return out


def score(det, audio, sr, tmpdir, tag):
    p = os.path.join(tmpdir, f"{tag}.wav")
    peak = np.abs(audio).max()
    a = audio / peak * 0.9 if peak > 1e-9 else audio
    sf.write(p, a, sr, subtype="PCM_24")
    try:
        prob, _ = det.predict(p)
    finally:
        if os.path.exists(p):
            os.remove(p)
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()

    det = FakeprintDetector(args.model)
    tmp = tempfile.mkdtemp(prefix="bandswap_")

    A, sra = sf.read(args.ai, always_2d=True, dtype="float64")
    C, src = sf.read(args.control, always_2d=True, dtype="float64")
    if sra != src:
        raise SystemExit(f"sample rates differ: {sra} vs {src}")
    n = int(min(len(A), len(C), args.seconds * sra))
    A, C = A[:n], C[:n]
    if A.shape[1] != C.shape[1]:
        raise SystemExit("channel counts differ")

    pA = score(det, A, sra, tmp, "A")
    pC = score(det, C, sra, tmp, "C")
    print(f"baselines:  AI track p={pA:.4f}   control p={pC:.4f}")
    print()
    print(f"{'band swapped':<14} {'GRAFT ctrl<-AI':>16} {'HEAL AI<-ctrl':>16}")
    print("-" * 50)

    for lo, hi in BANDS:
        g = score(det, graft(C, A, sra, lo, hi), sra, tmp, f"g{lo}")
        h = score(det, graft(A, C, sra, lo, hi), sra, tmp, f"h{lo}")
        label = f"{lo//1000}-{hi//1000}k"
        gmark = "  <-- flips" if g > 0.5 else ""
        hmark = "  <-- clears" if h < 0.5 else ""
        print(f"{label:<14} {g:>16.4f} {h:>16.4f}{gmark}{hmark}")

    os.rmdir(tmp)
    print()
    print("GRAFT high => that band alone is enough to make real audio look AI.")
    print("HEAL  low  => replacing that band alone is enough to clear the AI track.")


if __name__ == "__main__":
    main()
