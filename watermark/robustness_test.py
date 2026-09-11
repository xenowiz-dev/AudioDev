"""
Does an AudioSeal watermark survive an audio-restoration pipeline?

This matters for pipeline hygiene, not for evasion: if super-resolution or
transcoding silently strips a provenance mark, then processing someone's
audio destroys the very signal that says where it came from, and downstream
verification quietly fails open.

Applies benign, everyday transforms to a watermarked file and re-runs the
detector on each.

Usage:
    python robustness_test.py --wm controls/control_audioseal_watermarked_16k.wav
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
import torch


def detect(path):
    import librosa
    from audioseal import AudioSeal
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = x.mean(axis=1)
    if sr != 16000:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(mono)[None, None, :]
    det = AudioSeal.load_detector("audioseal_detector_16bits")
    with torch.no_grad():
        result, msg = det.detect_watermark(wav, sample_rate=16000)
    bits = None
    if float(result) > 0.5 and msg is not None:
        bits = "".join(str(int(round(b))) for b in msg.squeeze().tolist())
    return float(result), bits


def ff(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + args, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True, help="watermarked input")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    out = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.wm)),
                                      "robustness")
    os.makedirs(out, exist_ok=True)

    cases = {}
    cases["original watermarked"] = args.wm

    p = os.path.join(out, "mp3_128k.wav")
    ff(["-i", args.wm, "-c:a", "libmp3lame", "-b:a", "128k",
        os.path.join(out, "_t.mp3")])
    ff(["-i", os.path.join(out, "_t.mp3"), "-ar", "16000", "-c:a",
        "pcm_s16le", p])
    cases["MP3 128 kbps round-trip"] = p

    p = os.path.join(out, "mp3_64k.wav")
    ff(["-i", args.wm, "-c:a", "libmp3lame", "-b:a", "64k",
        os.path.join(out, "_t2.mp3")])
    ff(["-i", os.path.join(out, "_t2.mp3"), "-ar", "16000", "-c:a",
        "pcm_s16le", p])
    cases["MP3 64 kbps round-trip"] = p

    p = os.path.join(out, "resample_48k.wav")
    ff(["-i", args.wm, "-ar", "48000", "-c:a", "pcm_s24le", p])
    cases["resampled to 48 kHz"] = p

    p = os.path.join(out, "lowpass_8k.wav")
    ff(["-i", args.wm, "-af", "lowpass=f=6000", "-ar", "16000",
        "-c:a", "pcm_s16le", p])
    cases["lowpass 6 kHz"] = p

    p = os.path.join(out, "volume_down.wav")
    ff(["-i", args.wm, "-af", "volume=0.5", "-ar", "16000",
        "-c:a", "pcm_s16le", p])
    cases["gain -6 dB"] = p

    p = os.path.join(out, "noise.wav")
    x, sr = sf.read(args.wm, always_2d=True, dtype="float32")
    rng = np.random.RandomState(0)
    noisy = x + rng.normal(0, 10 ** (-40 / 20) * np.abs(x).std(), x.shape)
    sf.write(p, noisy, sr, subtype="PCM_16")
    cases["+40 dB SNR noise"] = p

    print(f"{'transform':<28} {'detector p':>11}  message")
    print("-" * 64)
    for name, path in cases.items():
        try:
            pr, bits = detect(path)
            print(f"{name:<28} {pr:>11.4f}  {bits or '-'}")
        except Exception as e:
            print(f"{name:<28} {'error':>11}  {type(e).__name__}")

    for f in os.listdir(out):
        if f.startswith("_t"):
            os.remove(os.path.join(out, f))
    print(f"\nfiles kept in {out}")


if __name__ == "__main__":
    sys.exit(main())
