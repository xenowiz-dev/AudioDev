"""Transcribe sung lyrics: isolate the vocal, then run Whisper on it.

Whisper on a full mix is markedly worse for singing -- the instrumental bed is
in-band with the voice, and ASR trained mostly on speech mistakes it for noise
or invents words from it. Separating the vocal stem first is the single biggest
quality lever available here, which is why Demucs runs by default.

The two models never share the GPU: Demucs is freed before Whisper loads, so
peak VRAM is the larger of the two, not their sum. That matters on a 10 GB card
already shared with the generators.

Output is plain lines, paragraph-broken on long silences. It deliberately does
NOT invent `[verse]` / `[chorus]` tags -- ACE-Step conditions on those, and
wrong structure is worse than none. Add them by hand if you want them.

Usage:
    python transcribe.py --input song.wav
    python transcribe.py --input song.wav --no-separate --model medium
    python transcribe.py --input song.wav --start 30 --seconds 60
"""

import argparse
import json
import os
import sys
import tempfile
import time


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def excerpt(path, start, seconds):
    """Cut a window with ffmpeg so long tracks can be sampled cheaply."""
    if not start and not seconds:
        return path, None
    out = os.path.join(tempfile.gettempdir(),
                       f"tx_{os.getpid()}_{int(start)}.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", out]
    import subprocess
    subprocess.run(cmd, check=True)
    return out, out


def separate(path, device):
    """Demucs vocal stem. Returns (wav_path, cleanup_path)."""
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    log("loading demucs (htdemucs)")
    model = get_model("htdemucs")
    model.to(device).eval()

    # soundfile rather than torchaudio.load: torchaudio 2.7 removed its
    # built-in I/O backends (moved to torchcodec), so `load` raises
    # "Couldn't find appropriate backend" on a plain wav.
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != model.samplerate:
        from fractions import Fraction
        from scipy.signal import resample_poly
        f = Fraction(model.samplerate, sr)
        data = resample_poly(data, f.numerator, f.denominator,
                             axis=0).astype("float32")
        sr = model.samplerate
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    log("separating vocals")
    with torch.no_grad():
        # split=True streams the track in windows; a full-length tensor would
        # not fit alongside the model on a 10 GB card.
        est = apply_model(model, wav[None], device=device, split=True,
                          overlap=0.25, progress=False)[0]
    est = est * (ref.std() + 1e-8) + ref.mean()
    voc = est[model.sources.index("vocals")]

    out = os.path.join(tempfile.gettempdir(), f"vocals_{os.getpid()}.wav")
    sf.write(out, voc.cpu().numpy().T, sr)

    # Free BEFORE whisper loads: peak = max(demucs, whisper), not the sum.
    del model, est, wav, voc
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, out


def transcribe(path, model_name, device, use_vad):
    from faster_whisper import WhisperModel

    compute = "int8_float16" if device == "cuda" else "int8"
    log(f"loading whisper {model_name} ({compute})")
    model = WhisperModel(model_name, device=device, compute_type=compute)

    log("transcribing")
    segments, info = model.transcribe(
        path,
        beam_size=5,
        # Lyrics repeat by design. With conditioning on, Whisper latches onto a
        # repeated chorus and loops it for the rest of the track.
        condition_on_previous_text=False,
        vad_filter=bool(use_vad),
        temperature=[0.0, 0.2, 0.4],
    )
    return list(segments), info


def to_lines(segments, gap=2.5):
    """Plain lines, blank line where a long silence suggests a section break."""
    out, prev_end = [], None
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        if prev_end is not None and (s.start - prev_end) >= gap and out:
            out.append("")
        out.append(text)
        prev_end = s.end
    return "\n".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="large-v3",
                    help="faster-whisper model (large-v3, medium, small...)")
    ap.add_argument("--no-separate", dest="separate", action="store_false",
                    help="run Whisper on the full mix (notably worse on sung "
                         "material -- for comparison, not for real use)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 = whole file")
    ap.add_argument("--vad", dest="vad", action="store_true", default=False,
                    help="Whisper VAD gate. Measured to FRAGMENT sung lines "
                         "on this material, so it is off by default.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON on stdout instead of plain text")
    args = ap.parse_args()

    t0 = time.time()
    temps = []
    try:
        src, tmp = excerpt(args.input, args.start, args.seconds)
        if tmp:
            temps.append(tmp)

        if args.separate:
            src, tmp = separate(src, args.device)
            temps.append(tmp)

        segments, info = transcribe(src, args.model, args.device, args.vad)
        text = to_lines(segments)
        elapsed = time.time() - t0

        log(f"done in {elapsed:.0f}s  language={info.language} "
            f"({info.language_probability:.2f})  segments={len(segments)}")
        if args.json:
            print(json.dumps({
                "ok": True, "text": text, "separated": args.separate,
                "model": args.model, "language": info.language,
                "language_probability": info.language_probability,
                "elapsed": elapsed, "segments": len(segments),
            }, ensure_ascii=False))
        else:
            print(text)
        return 0
    finally:
        for p in temps:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
