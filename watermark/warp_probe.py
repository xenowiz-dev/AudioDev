"""
Do time warps and heavy degradation disturb the vocoder frame signature?

Sharp, opposite predictions, which is what makes this worth running:

  * RESAMPLE (speed change). Playing r times faster scales every frequency by
    r, so a 200 Hz comb should MOVE to 200*r Hz and stay just as strong. The
    artefact survives; only its address changes. A detector that sweeps
    spacings should follow it, one hard-coded to 200 Hz should lose it.

  * TIME-STRETCH (tempo change at constant pitch). A phase vocoder resynthesises
    from overlapping analysis frames on its OWN grid, unrelated to the
    generator's. That should smear the original frame lock without moving it
    anywhere in particular -- destruction, not translation.

Both are reported with the quality cost, and against the trained Suno classifier
as well, so it is visible whether a hard-coded detector and a searching detector
diverge.

Usage:
    python warp_probe.py --input suno.wav -o outdir
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_audio_forensics import FakeprintDetector
from comb_detect import analyse as comb_analyse

TREATMENTS = [
    # label, ffmpeg -af, predicted comb spacing (None = unknown)
    ("resample +0.1%",  "asetrate=48000*1.001,aresample=48000", 200 * 1.001),
    ("resample +0.5%",  "asetrate=48000*1.005,aresample=48000", 200 * 1.005),
    ("resample +2%",    "asetrate=48000*1.02,aresample=48000",  200 * 1.02),
    ("resample -3%",    "asetrate=48000*0.97,aresample=48000",  200 * 0.97),
    ("stretch +0.5% (atempo)", "atempo=1.005", None),
    ("stretch +3% (atempo)",   "atempo=1.03", None),
    ("stretch -5% (atempo)",   "atempo=0.95", None),
    ("vibrato warp",    "vibrato=f=0.4:d=0.35", None),
    ("mp3 64k",         None, 200.0),
    ("mp3 32k",         None, 200.0),
    ("opus 48k",        None, 200.0),
    ("heavy chain",     "acompressor=threshold=0.05:ratio=12,"
                        "aexciter=amount=3,highpass=f=60,lowpass=f=15000",
                        200.0),
]


def run_ff(inp, out, af=None, codec=None, bitrate=None, tmpdir=None):
    if codec:
        mid = os.path.join(tmpdir, "_m." + ("opus" if codec == "libopus" else "mp3"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp,
                        "-c:a", codec, "-b:a", bitrate, mid], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mid,
                        "-ar", "48000", "-c:a", "pcm_s24le", out], check=True)
        os.remove(mid)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp,
                        "-af", af, "-ar", "48000", "-c:a", "pcm_s24le", out],
                       check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    det = FakeprintDetector(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))

    base_p, _ = det.predict(args.input)
    base = comb_analyse(args.input, 1000.0, None)
    print(f"baseline: trained classifier p={base_p:.4f} | "
          f"comb {base['spacing']:.2f} Hz score {base['score']:.2f}")
    print()
    print(f"{'treatment':<26} {'predicted':>10} {'found Hz':>10} {'score':>7} "
          f"{'z':>6} {'classifier p':>13}")
    print("-" * 82)

    for label, af, pred in TREATMENTS:
        out = os.path.join(args.outdir, label.replace(" ", "_")
                           .replace("%", "pc").replace("(", "").replace(")", "")
                           + ".wav")
        try:
            if label.startswith("mp3"):
                run_ff(args.input, out, codec="libmp3lame",
                       bitrate=label.split()[1], tmpdir=args.outdir)
            elif label.startswith("opus"):
                run_ff(args.input, out, codec="libopus", bitrate="48k",
                       tmpdir=args.outdir)
            else:
                run_ff(args.input, out, af=af)
            r = comb_analyse(out, 1000.0, None)
            p, _ = det.predict(out)
            ps = f"{pred:.2f}" if pred else "smeared"
            hit = ""
            if pred and r and abs(r["spacing"] - pred) < 1.5:
                hit = "  <-- moved as predicted"
            elif pred and r and abs(r["spacing"] - 200.0) < 1.5:
                hit = "  <-- held at 200"
            print(f"{label:<26} {ps:>10} {r['spacing']:>10.2f} "
                  f"{r['score']:>7.2f} {r['z']:>6.1f} {p:>13.4f}{hit}")
        except Exception as e:
            print(f"{label:<26} {'':>10} {'error':>10}  "
                  f"{type(e).__name__}: {str(e)[:40]}")

    print()
    print("A searching detector follows a resampled comb; a 200 Hz-hardcoded")
    print("one does not. Stretching smears rather than moves it.")


if __name__ == "__main__":
    sys.exit(main())
