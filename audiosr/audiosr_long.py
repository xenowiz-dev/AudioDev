"""
Chunked AudioSR runner for full-length tracks.

Stock `audiosr` builds ONE batch for the whole file, so VRAM scales with
duration and a 10 GB card OOMs past ~30 s. This splits the input into
overlapping windows, runs AudioSR per window, then equal-power crossfades
the results back together.

Usage:
    python audiosr_long.py -i input.wav -o output.wav
    python audiosr_long.py -i input.mp3 -o out.wav --chunk 10.24 --overlap 1.0
    python audiosr_long.py -i input.wav -o out.wav --lowpass 11250
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

OUT_SR = 48000  # AudioSR always emits 48 kHz


def detect_cutoff(x, sr, floor_db=-55.0):
    """Estimate the real bandwidth of a lossy file, in Hz.

    Averages magnitude spectra over several windows and finds the highest
    bin still above `floor_db` relative to peak.
    """
    n = 8192
    if len(x) < n:
        return sr / 2
    step = max(n, (len(x) - n) // 40)
    acc = np.zeros(n // 2 + 1)
    win = np.hanning(n)
    count = 0
    for start in range(0, len(x) - n, step):
        acc += np.abs(np.fft.rfft(x[start:start + n] * win))
        count += 1
    if count == 0:
        return sr / 2
    acc /= count
    db = 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)
    above = np.where(db > floor_db)[0]
    freqs = np.fft.rfftfreq(n, 1 / sr)
    return float(freqs[above[-1]]) if len(above) else sr / 2


def apply_lowpass(x, sr, cutoff):
    """Steep lowpass. AudioSR was trained on lowpassed audio only; feeding it
    raw codec artifacts above the cutoff makes it hallucinate. See repo README.
    """
    from scipy.signal import cheby1, sosfiltfilt
    nyq = sr / 2
    wn = min(cutoff / nyq, 0.999)
    sos = cheby1(8, 0.05, wn, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def crossfade_curves(n):
    """Equal-power fade pair: fade_out**2 + fade_in**2 == 1.

    AudioSR is generative, so each window invents its own high-frequency
    detail. Across a seam those components are largely uncorrelated, which
    makes an equal-gain (a + b == 1) fade dip audibly in the middle.
    """
    theta = np.linspace(0, np.pi / 2, n, endpoint=False)
    return np.cos(theta), np.sin(theta)  # (fade_out, fade_in)


def main():
    ap = argparse.ArgumentParser(description="Chunked AudioSR for long files")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--chunk", type=float, default=5.12,
                    help="window length in seconds (default 5.12, verified on "
                         "a 10 GB card)")
    ap.add_argument("--overlap", type=float, default=1.0,
                    help="crossfade overlap in seconds (default 1.0)")
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("-gs", "--guidance_scale", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_name", default="basic", choices=["basic", "speech"])
    ap.add_argument("-d", "--device", default="auto")
    ap.add_argument("--lowpass", type=float, default=None,
                    help="lowpass the input at this Hz before SR. "
                         "Use 'auto' behaviour via --auto-lowpass instead to detect it.")
    ap.add_argument("--auto-lowpass", action="store_true",
                    help="detect the real cutoff and lowpass there first "
                         "(recommended for MP3/AAC sources)")
    args = ap.parse_args()

    from audiosr import build_model, super_resolution

    x, sr = sf.read(args.input, always_2d=True, dtype="float32")
    mono_probe = x.mean(axis=1)
    print(f"input: {args.input}  sr={sr}  ch={x.shape[1]}  "
          f"dur={len(x)/sr:.1f}s", flush=True)

    cutoff = detect_cutoff(mono_probe, sr)
    print(f"detected bandwidth: ~{cutoff:.0f} Hz", flush=True)

    if args.auto_lowpass or args.lowpass:
        lp = args.lowpass if args.lowpass else cutoff
        print(f"pre-lowpass at {lp:.0f} Hz", flush=True)
        x = apply_lowpass(x, sr, lp)

    # AudioSR reads mono internally; process the downmix and keep it mono-out
    # unless the source was already mono.
    n_ch = x.shape[1]
    if n_ch > 1:
        print("note: AudioSR is mono-in/mono-out -- processing channels "
              "separately to preserve the stereo image", flush=True)

    print(f"loading AudioSR ({args.model_name})...", flush=True)
    model = build_model(model_name=args.model_name, device=args.device)

    chunk_n = int(args.chunk * sr)
    hop_n = int((args.chunk - args.overlap) * sr)
    ov_out = int(args.overlap * OUT_SR)

    tmpdir = tempfile.mkdtemp(prefix="audiosr_chunks_")
    out_channels = []

    try:
        for ch in range(n_ch):
            sig = x[:, ch]
            starts = list(range(0, max(1, len(sig) - (chunk_n - hop_n)), hop_n))
            acc = None
            for k, s in enumerate(starts):
                seg = sig[s:s + chunk_n]
                if len(seg) < chunk_n:
                    seg = np.pad(seg, (0, chunk_n - len(seg)))
                cpath = os.path.join(tmpdir, f"c{ch}_{k:05d}.wav")
                sf.write(cpath, seg, sr, subtype="PCM_16")

                y = super_resolution(
                    model, cpath, seed=args.seed,
                    ddim_steps=args.ddim_steps,
                    guidance_scale=args.guidance_scale,
                )
                y = np.asarray(y).squeeze()
                if y.ndim > 1:
                    y = y[0]
                os.remove(cpath)

                # AudioSR rounds duration up to the next 2.5 s multiple
                # (round_up_duration in pipeline.py), so a 10.24 s window comes
                # back as 12.5 s with padding-generated tail. Trim to the true
                # window length or that garbage gets spliced into every seam and
                # the timeline drifts progressively later.
                expected = int(round(chunk_n / sr * OUT_SR))
                if len(y) > expected:
                    y = y[:expected]
                elif len(y) < expected:
                    y = np.pad(y, (0, expected - len(y)))

                if acc is None:
                    acc = y.astype(np.float32)
                else:
                    ov = min(ov_out, len(acc), len(y))
                    if ov > 0:
                        f_out, f_in = crossfade_curves(ov)
                        acc[-ov:] = acc[-ov:] * f_out + y[:ov] * f_in
                        acc = np.concatenate([acc, y[ov:]])
                    else:
                        acc = np.concatenate([acc, y])

                print(f"  ch{ch} chunk {k+1}/{len(starts)}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            target_len = int(len(sig) / sr * OUT_SR)
            acc = acc[:target_len] if len(acc) >= target_len else \
                np.pad(acc, (0, target_len - len(acc)))
            out_channels.append(acc)

        out = np.stack(out_channels, axis=1)
        peak = np.abs(out).max()
        if peak > 0.999:
            out = out / peak * 0.999
            print(f"normalised (peak was {peak:.3f})", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sf.write(args.output, out, OUT_SR, subtype="PCM_24")
        print(f"wrote {args.output}  ({OUT_SR} Hz, {out.shape[1]} ch, "
              f"{len(out)/OUT_SR:.1f}s)", flush=True)
    finally:
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
