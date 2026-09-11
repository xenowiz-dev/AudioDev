"""
Does denoising remove the vocoder fakeprint?

The signature is a broadband low-level spectral texture spread over ~1-5 kHz,
which is exactly the region and the amplitude scale that denoisers operate on.
So there is a real question: does ordinary noise reduction destroy it?

This matters in BOTH directions and is a detector-reliability question, not a
recipe:
  * false negatives - a denoised generated file that stops being detectable
  * false positives - does denoising push real audio toward "AI"?

Every result is reported alongside the damage the denoiser did (SNR against the
untouched input, and change in high-frequency energy), because a "pass" bought
by mangling the audio is not a pass.

Usage:
    python denoise_probe.py --ai suno.wav --control real.wav -o outdir
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

# name -> ffmpeg -af string (None = handled in python)
TREATMENTS = [
    ("afftdn nr=6 (light FFT)",      "afftdn=nr=6:nf=-40"),
    ("afftdn nr=20",                 "afftdn=nr=20:nf=-30"),
    ("afftdn nr=40 (heavy)",         "afftdn=nr=40:nf=-20"),
    # nf is clamped to [-80,-20]; -20 is the most aggressive legal noise floor.
    ("afftdn nr=97 (extreme)",       "afftdn=nr=97:nf=-20"),
    ("afwtdn (wavelet)",             "afwtdn=sigma=0.05"),
    ("afwtdn sigma=0.2 (heavy)",     "afwtdn=sigma=0.2"),
    # anlmdn segfaults on this ffmpeg build with stereo input; run it mono.
    ("anlmdn (non-local means)",     "aformat=channel_layouts=mono,anlmdn=s=0.001"),
    ("anlmdn s=0.01 (heavy)",        "aformat=channel_layouts=mono,anlmdn=s=0.01"),
    ("noisereduce 0.5",              None),
    ("noisereduce 1.0",              None),
    ("DeepFilterNet (neural)",       None),
]


def ff(inp, out, af):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", inp,
                    "-af", af, "-ar", "48000", "-c:a", "pcm_s24le", out],
                   check=True)


def noisereduce_file(inp, out, prop):
    import noisereduce as nr
    x, sr = sf.read(inp, always_2d=True, dtype="float32")
    y = np.stack([nr.reduce_noise(y=x[:, c], sr=sr, prop_decrease=prop,
                                  stationary=False)
                  for c in range(x.shape[1])], axis=1)
    sf.write(out, y, sr, subtype="PCM_24")


_DF = {}


def deepfilternet_file(inp, out):
    """Neural denoiser: it resynthesises rather than subtracts, which is the
    strongest available test of 'the signature hides in the noise floor'.
    Operates at 48 kHz mono; run per channel."""
    import torch
    from df.enhance import enhance, init_df
    if "m" not in _DF:
        _DF["m"], _DF["state"], _ = init_df()
    x, sr = sf.read(inp, always_2d=True, dtype="float32")
    if sr != 48000:
        raise RuntimeError(f"DeepFilterNet needs 48 kHz, got {sr}")
    chans = []
    for c in range(x.shape[1]):
        t = torch.from_numpy(x[:, c]).unsqueeze(0)
        chans.append(enhance(_DF["m"], _DF["state"], t).squeeze().numpy())
    k = min(len(c_) for c_ in chans)
    sf.write(out, np.stack([c_[:k] for c_ in chans], axis=1), 48000,
             subtype="PCM_24")


def quality(ref_path, test_path):
    a, sr = sf.read(ref_path, always_2d=True, dtype="float64")
    b, _ = sf.read(test_path, always_2d=True, dtype="float64")
    n = min(len(a), len(b))
    a, b = a[:n].mean(axis=1), b[:n].mean(axis=1)

    # FFT- and network-based denoisers introduce latency. Comparing
    # sample-aligned makes a gentle filter look like total destruction, so
    # find and remove the delay first (searched over +/-0.1 s).
    m = min(len(a), 1 << 19)
    lim = int(0.1 * sr)
    xc = np.correlate(a[:m] - a[:m].mean(), b[:m] - b[:m].mean(), mode="full")
    centre = len(b[:m]) - 1
    lo, hi = max(0, centre - lim), min(len(xc), centre + lim)
    lag = int(np.argmax(np.abs(xc[lo:hi]))) + lo - centre
    if lag > 0:
        a2, b2 = a[lag:], b[:len(b) - lag]
    elif lag < 0:
        a2, b2 = a[:len(a) + lag], b[-lag:]
    else:
        a2, b2 = a, b
    k = min(len(a2), len(b2))
    a2, b2 = a2[:k], b2[:k]

    # align gain so SNR reflects distortion, not level change
    g = (a2 @ b2) / (b2 @ b2 + 1e-20)
    d = a2 - g * b2
    snr = 10 * np.log10((a2 ** 2).mean() / ((d ** 2).mean() + 1e-20))
    a, b = a2, b2

    def hf(z):
        S = np.abs(np.fft.rfft(z * np.hanning(len(z))))
        fr = np.fft.rfftfreq(len(z), 1 / sr)
        return S[(fr > 4000) & (fr < 8000)].sum() / (S.sum() + 1e-20)
    return snr, 20 * np.log10((hf(b) + 1e-20) / (hf(a) + 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--model", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "ai_music_detector.onnx"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    det = FakeprintDetector(args.model)
    srcs = {"AI": args.ai, "control": args.control}

    base = {k: det.predict(v)[0] for k, v in srcs.items()}
    print(f"baseline:  AI p={base['AI']:.4f}   control p={base['control']:.4f}")
    print()
    print(f"{'denoiser':<28} {'AI p':>8} {'SNR dB':>8} {'HFΔ dB':>8}   "
          f"{'ctrl p':>8} {'SNR dB':>8}")
    print("-" * 78)

    for name, af in TREATMENTS:
        row = {}
        for k, src in srcs.items():
            out = os.path.join(args.outdir,
                               f"{k}_{name.split()[0]}_{abs(hash(name)) % 9999}.wav")
            try:
                if af is None and name.startswith("DeepFilterNet"):
                    deepfilternet_file(src, out)
                elif af is None:
                    noisereduce_file(src, out, 0.5 if "0.5" in name else 1.0)
                else:
                    ff(src, out, af)
                p, _ = det.predict(out)
                snr, hfd = quality(src, out)
                row[k] = (p, snr, hfd)
            except Exception as e:
                row[k] = (float("nan"), float("nan"), float("nan"))
                print(f"  ! {name} on {k}: {type(e).__name__}: {e}")
        a = row.get("AI", (float("nan"),) * 3)
        c = row.get("control", (float("nan"),) * 3)
        flag = ""
        if a[0] == a[0] and a[0] < 0.5:
            flag = "   <-- AI no longer detected"
        elif c[0] == c[0] and c[0] > 0.5:
            flag = "   <-- control now flagged"
        print(f"{name:<28} {a[0]:>8.4f} {a[1]:>8.1f} {a[2]:>8.1f}   "
              f"{c[0]:>8.4f} {c[1]:>8.1f}{flag}")

    print()
    print("SNR = signal-to-distortion vs the untouched input (higher = gentler).")
    print("HFd = change in 4-8 kHz energy share; large negative = HF gutted.")


if __name__ == "__main__":
    main()
