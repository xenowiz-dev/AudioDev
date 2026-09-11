"""
Round-trip audio through ACE-Step's VAE and look for its frame-rate comb.

The comb is stamped by the decoder, not by the diffusion model, so the whole
text-to-music pipeline is unnecessary to test the prediction: encode real audio
to the latent, decode it back, and measure.

ACE-Step 1.5 uses AutoencoderOobleck with
    downsampling_ratios = [2, 4, 4, 6, 10]   product 1920
    sampling_rate       = 48000
so the latent runs at 48000/1920 = 25 Hz, and the decoder's transposed
convolutions should stamp a comb at 25 Hz (plus harmonics).

Prediction registered before running: a 25 Hz comb, present in the output and
absent from the input.

Usage:
    python vae_roundtrip.py --input control.wav --out out.wav
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vae", default=os.path.join(ROOT, "acestep", "checkpoints", "vae"))
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    from diffusers import AutoencoderOobleck

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderOobleck.from_pretrained(args.vae).to(dev).eval()

    ratios = getattr(vae.config, "downsampling_ratios", None)
    sr_cfg = getattr(vae.config, "sampling_rate", 48000)
    total = int(np.prod(ratios)) if ratios else None
    if total:
        print(f"VAE: ratios {ratios} -> total {total}, sr {sr_cfg}")
        print(f"     predicted latent frame rate = {sr_cfg/total:.4f} Hz")

    x, sr = sf.read(args.input, always_2d=True, dtype="float32")
    x = x[: int(args.seconds * sr)]
    if sr != sr_cfg:
        import librosa
        x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=sr_cfg)
                      for c in range(x.shape[1])], axis=1)
        sr = sr_cfg
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)

    # Trim to a whole number of latent frames so no partial frame is padded.
    if total:
        n = (len(x) // total) * total
        x = x[:n]

    t = torch.from_numpy(x.T).unsqueeze(0).to(dev)
    with torch.no_grad():
        posterior = vae.encode(t).latent_dist
        z = posterior.mode()
        y = vae.decode(z).sample
    y = y.squeeze(0).cpu().numpy().T

    print(f"latent shape {tuple(z.shape)}  -> "
          f"{z.shape[-1]/(len(x)/sr):.2f} latent frames per second")

    peak = np.abs(y).max()
    if peak > 0.999:
        y = y / peak * 0.999
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, y, sr, subtype="PCM_24")
    print(f"wrote {args.out}  ({sr} Hz, {y.shape[1]} ch, {len(y)/sr:.1f}s)")


if __name__ == "__main__":
    main()
