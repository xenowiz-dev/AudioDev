"""
Chunked FlashSR runner for full-length tracks.

FlashSR is a one-step distillation of AudioSR: same idea, one sampling step
instead of 50. It only accepts exactly 245760 samples (5.12 s at 48 kHz), so
long input has to be windowed and stitched.

Its forward() takes [batch, time] -- the leading dim is a true batch, not
channels -- so windows AND stereo channels are pushed through together in one
pass. That is where most of the speed comes from.

Unlike AudioSR, FlashSR does its own cutoff detection and lowpass internally
(lowpass_input=True by default), so no pre-filtering is needed here.

Usage:
    python flashsr_long.py -i input.wav -o output.wav
    python flashsr_long.py -i in.wav -o out.wav --batch 8 --overlap 0.5
    python flashsr_long.py -i in.wav -o out.wav --no-lowpass
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch

SR = 48000
CHUNK = 245760          # the only length FlashSR accepts
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FlashSR_Inference")


def crossfade_curves(n):
    """Equal-power pair: fade_out**2 + fade_in**2 == 1."""
    theta = np.linspace(0, np.pi / 2, n, endpoint=False)
    return np.cos(theta), np.sin(theta)


def detect_cliff(x, sr):
    """Locate the codec cutoff as the steepest high-frequency rolloff.

    FlashSR ships find_cutoff_freq(), which takes a fixed 98.3rd-percentile of
    spectral energy. On real music that lands well BELOW the actual cliff
    (measured: 12820 Hz vs a true 15305 Hz cliff, and 15773 vs 20010 on a
    320k-class file). Since the cutoff is used to lowpass the input before
    super-resolution, under-estimating it throws away real audio and asks the
    model to re-invent it. Finding the actual edge keeps that content.
    """
    n = 8192
    if len(x) < n * 2:
        return None
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    cnt = 0
    for s in range(0, len(x) - n, max(n, (len(x) - n) // 300)):
        acc += np.abs(np.fft.rfft(x[s:s + n] * win))
        cnt += 1
    acc /= max(cnt, 1)
    fr = np.fft.rfftfreq(n, 1 / sr)
    db = 20 * np.log10(acc / (acc.max() + 1e-12) + 1e-12)
    sm = np.convolve(db, np.ones(9) / 9, mode="same")
    d = np.diff(sm)
    searchable = fr[:-1] > 6000
    if not searchable.any():
        return None
    i = int(np.argmin(np.where(searchable, d, 0)))
    cliff = fr[i]
    lo = db[(fr >= cliff - 1500) & (fr < cliff)]
    hi = db[(fr > cliff) & (fr <= cliff + 1500)]
    drop = (lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0
    # No real cliff -> nothing was cut off; don't filter.
    return float(cliff) if drop > 8 else None


def load_48k(path):
    x, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != SR:
        import librosa
        x = np.stack([librosa.resample(x[:, c], orig_sr=sr, target_sr=SR)
                      for c in range(x.shape[1])], axis=1)
        print(f"resampled {sr} -> {SR} Hz", flush=True)
    return x


def main():
    ap = argparse.ArgumentParser(description="Chunked FlashSR for long files")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="crossfade overlap in seconds (default 0.5)")
    ap.add_argument("--batch", type=int, default=4,
                    help="windows per forward pass (default 4)")
    ap.add_argument("--steps", type=int, default=1,
                    help="sampling steps (FlashSR is distilled for 1)")
    ap.add_argument("--no-lowpass", action="store_true",
                    help="skip FlashSR's internal lowpass of the input")
    ap.add_argument("--cutoff", type=int, default=None,
                    help="force the lowpass cutoff in Hz instead of detecting")
    ap.add_argument("--detect", choices=["cliff", "flashsr"], default="cliff",
                    help="cutoff detector: 'cliff' finds the steepest rolloff "
                         "(default, more accurate on music); 'flashsr' uses the "
                         "model's own percentile method")
    args = ap.parse_args()

    sys.path.insert(0, REPO)
    from FlashSR.FlashSR import FlashSR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = os.path.join(REPO, "ModelWeights")
    print("loading FlashSR...", flush=True)
    model = FlashSR(
        os.path.join(weights, "student_ldm.pth"),
        os.path.join(weights, "sr_vocoder.pth"),
        os.path.join(weights, "vae.pth"),
    ).to(device)

    x = load_48k(args.input)
    n_samples, n_ch = x.shape
    print(f"input: {args.input}  {n_ch} ch  {n_samples/SR:.1f}s", flush=True)

    # FlashSR's built-in lowpass asserts batch is 1 or 2, which would cap the
    # batch size and make it re-detect the cutoff from whichever window happens
    # to be first in each batch. Instead: detect the cutoff ONCE over the whole
    # track and apply their exact filter up front, then run with
    # lowpass_input=False so batching is unconstrained.
    if not args.no_lowpass:
        from FlashSR.Util.UtilAudioSR import UtilAudioSR
        from FlashSR.Util.UtilAudioLowPassFilter import UtilAudioLowPassFilter

        cutoff = args.cutoff
        if cutoff is None:
            probe = x.mean(axis=1)
            # Detect on the middle 60 s of long tracks -- cheaper, and avoids
            # fades at the head/tail skewing the estimate.
            if len(probe) > 60 * SR:
                mid = len(probe) // 2
                probe = probe[mid - 30 * SR: mid + 30 * SR]

            if args.detect == "flashsr":
                cutoff = int(UtilAudioSR.find_cutoff_freq(
                    torch.from_numpy(probe.copy()).float().unsqueeze(0)))
            else:
                c = detect_cliff(probe, SR)
                if c is None:
                    print("no hard cliff found -- skipping lowpass", flush=True)
                    cutoff = 24000
                else:
                    cutoff = int(c)
            print(f"detected cutoff ({args.detect}): {cutoff} Hz", flush=True)

        if cutoff < 23999:
            x = UtilAudioLowPassFilter.lowpass(
                x.T.copy(), SR, filter_name="cheby",
                filter_order=8, cutoff_freq=cutoff).T.copy()
            print(f"pre-lowpassed at {cutoff} Hz", flush=True)
        else:
            print("cutoff at/above Nyquist -- skipping lowpass", flush=True)

    ov = int(args.overlap * SR)
    hop = CHUNK - ov
    if hop <= 0:
        raise SystemExit("--overlap must be smaller than 5.12 s")

    starts = list(range(0, max(1, n_samples - ov), hop))

    # Build every (channel, window) segment up front, then push them through in
    # batches. Padding the tail is fine -- it gets trimmed after stitching.
    # Laid out channel-major, so segment (ch, k) is at ch * len(starts) + k.
    segs = []
    for ch in range(n_ch):
        for s in starts:
            seg = x[s:s + CHUNK, ch]
            if len(seg) < CHUNK:
                seg = np.pad(seg, (0, CHUNK - len(seg)))
            segs.append(seg)

    print(f"{len(starts)} windows x {n_ch} ch = {len(segs)} segments, "
          f"batch {args.batch}", flush=True)

    out_segs = [None] * len(segs)
    with torch.no_grad():
        for b in range(0, len(segs), args.batch):
            block = np.stack(segs[b:b + args.batch])
            t = torch.from_numpy(block).float().to(device)
            # lowpass already applied above (or deliberately skipped)
            y = model(t, num_steps=args.steps, lowpass_input=False)
            y = y.detach().float().cpu().numpy()
            if y.ndim == 3:
                y = y.squeeze(1)
            for j in range(len(block)):
                out_segs[b + j] = y[j][:CHUNK]
            print(f"  {min(b+args.batch, len(segs))}/{len(segs)}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Stitch each channel back together with equal-power crossfades.
    channels = []
    for ch in range(n_ch):
        acc = None
        for k in range(len(starts)):
            y = out_segs[ch * len(starts) + k]
            if acc is None:
                acc = y.astype(np.float32).copy()
                continue
            o = min(ov, len(acc), len(y))
            if o > 0:
                f_out, f_in = crossfade_curves(o)
                acc[-o:] = acc[-o:] * f_out + y[:o] * f_in
                acc = np.concatenate([acc, y[o:]])
            else:
                acc = np.concatenate([acc, y])
        acc = acc[:n_samples] if len(acc) >= n_samples else \
            np.pad(acc, (0, n_samples - len(acc)))
        channels.append(acc)

    out = np.stack(channels, axis=1)
    peak = np.abs(out).max()
    if peak > 0.999:
        out = out / peak * 0.999
        print(f"normalised (peak was {peak:.3f})", flush=True)

    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    sf.write(args.output, out, SR, subtype="PCM_24")
    print(f"wrote {args.output}  ({SR} Hz, {out.shape[1]} ch, "
          f"{len(out)/SR:.1f}s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
