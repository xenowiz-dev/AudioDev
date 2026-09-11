"""
Build a control set so the detectors' verdicts can be interpreted.

A detector that says "AI" is only meaningful if you know what it says about
material whose provenance you already know. This creates:

  * an AudioSeal-watermarked file  -> layer 2 MUST fire (validates layer 2)
  * the same file unwatermarked    -> layer 2 MUST NOT fire (false-positive check)

Usage:
    python make_controls.py --in ref.wav --outdir controls
"""

import argparse
import os

# AudioSeal wraps its SEANet blocks in torch.compile, which needs Triton and so
# fails on Windows. Must be set before audioseal is imported.
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()

    import librosa
    from audioseal import AudioSeal

    os.makedirs(args.outdir, exist_ok=True)

    x, sr = sf.read(args.inp, always_2d=True, dtype="float32")
    mono = x.mean(axis=1)[: int(args.seconds * sr)]
    # AudioSeal's 16-bit models operate at 16 kHz.
    mono16 = librosa.resample(mono, orig_sr=sr, target_sr=16000) if sr != 16000 else mono

    clean = os.path.join(args.outdir, "control_clean_16k.wav")
    sf.write(clean, mono16, 16000, subtype="PCM_16")

    gen = AudioSeal.load_generator("audioseal_wm_16bits")
    gen.eval()
    wav = torch.from_numpy(mono16)[None, None, :]

    msg_bits = torch.randint(0, 2, (1, 16))
    with torch.no_grad():
        wm = gen.get_watermark(wav, sample_rate=16000, message=msg_bits)
        marked = (wav + wm).squeeze().numpy()

    out = os.path.join(args.outdir, "control_audioseal_watermarked_16k.wav")
    sf.write(out, marked, 16000, subtype="PCM_16")

    delta = marked - mono16
    snr = 10 * np.log10((mono16 ** 2).mean() / ((delta ** 2).mean() + 1e-20))
    print(f"embedded message : {''.join(str(int(b)) for b in msg_bits[0].tolist())}")
    print(f"watermark SNR    : {snr:.1f} dB  "
          f"(higher = more inaudible; the mark is a tiny additive signal)")
    print(f"watermark peak   : {np.abs(delta).max():.5f} full-scale")
    print(f"wrote {clean}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
