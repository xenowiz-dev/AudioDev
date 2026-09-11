"""
Does passing known-real audio through a neural vocoder stamp the AI signature?

This is the decisive test of the mechanism. The claim behind the detector is
that transposed-convolution upsampling in neural vocoders leaves a
characteristic spectral fingerprint. If that is true, then round-tripping
audio of KNOWN non-AI origin through a neural codec -- whose decoder is exactly
such a vocoder -- must flip the verdict. If the verdict does not move, the
detector is keying on something else entirely.

Encodec is ideal for this: its decoder is a transposed-conv stack, it has no
relationship to Suno or Udio, and the bitrate knob varies how hard the vocoder
has to work.

Usage:
    python vocoder_test.py --input control.wav -o outdir
"""

import argparse
import os
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector


def encodec_roundtrip(path, out, bandwidth):
    import torchaudio
    from encodec import EncodecModel
    from encodec.utils import convert_audio

    model = EncodecModel.encodec_model_48khz()
    model.set_target_bandwidth(bandwidth)
    model.eval()

    wav, sr = torchaudio.load(path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        frames = model.encode(wav)
        dec = model.decode(frames)
    y = dec.squeeze(0).cpu().numpy().T
    peak = np.abs(y).max()
    if peak > 0.999:
        y = y / peak * 0.999
    sf.write(out, y, model.sample_rate, subtype="PCM_24")
    return y.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="a file of KNOWN non-AI origin")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    det = FakeprintDetector(args.model)
    base, _ = det.predict(args.input)
    print(f"input (known non-AI): p={base:.4f}")
    print()
    print(f"{'Encodec bandwidth':<22} {'p(AI)':>9} {'mean(fp)':>10}  verdict")
    print("-" * 58)

    for bw in (24.0, 12.0, 6.0, 3.0, 1.5):
        out = os.path.join(args.outdir, f"encodec_{bw:g}kbps.wav")
        try:
            encodec_roundtrip(args.input, out, bw)
            p, _ = det.predict(out)
            fp, _ = det.fakeprint(out)
            verdict = "AI-GENERATED" if p >= 0.5 else "real"
            mark = "  <-- FLIPPED" if (p >= 0.5) != (base >= 0.5) else ""
            print(f"{bw:>6g} kbps{'':<12} {p:>9.4f} {fp.mean():>10.3f}  "
                  f"{verdict}{mark}")
        except Exception as e:
            print(f"{bw:>6g} kbps{'':<12} {'error':>9}  "
                  f"{type(e).__name__}: {e}")

    print()
    print("A flip proves the detector responds to neural-vocoder resynthesis")
    print("in general, not to Suno/Udio specifically.")


if __name__ == "__main__":
    main()
