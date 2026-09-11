"""Bandwidth extension as a post step, over the three installed upscalers.

These are one-shot CLI tools, not stateful models, so there is no warm worker
and no JSON protocol here -- just a subprocess per invocation with stderr merged
into stdout, which is both simpler and deadlock-free.

The wrappers are called rather than the raw tools, deliberately: `audiosr.ps1`
handles ffmpeg transcoding and the chunking that stock AudioSR needs to avoid
OOM past ~20 s, and `apollo.ps1` handles MSST's folder-only input contract.

Ranking is from this repo's ground-truth test (real track -> 96 kbps -> restore
-> compare against the original), not from the tools' marketing:

    AudioSR   4.37 dB mean error   most accurate, ~10x slower
    Apollo    7.00 dB              fast, trained on real codec artifacts
    FlashSR  11.34 dB              fastest, worst

Worth knowing before using any of them on generated audio: **neither generator
here produces a codec cliff** (measured: MiniMax 0.6 dB, ACE-Step 3.9 dB), and
upresing material with no cliff makes it worse. The honest use for this module
is preparing a lossy *source* track before covering it.
"""

import glob
import os
import subprocess
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

PS = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

WATERMARK_PY = os.path.join(ROOT, "watermark", ".venv", "Scripts", "python.exe")
CHECK_BW = os.path.join(ROOT, "check_bandwidth.py")
FLASHSR_PY = os.path.join(ROOT, "flashsr", ".venv", "Scripts", "python.exe")
FLASHSR_SCRIPT = os.path.join(ROOT, "flashsr", "flashsr_long.py")

UPSCALERS = {
    "none": "None",
    "apollo": "Apollo — fast, trained on codec artifacts (7.00 dB)",
    "audiosr": "AudioSR — most accurate, ~10x slower (4.37 dB)",
    "flashsr": "FlashSR — fastest, least accurate (11.34 dB)",
}

FINGERPRINT_NOTE = (
    "AudioSR and FlashSR stamp a 100 Hz neural-vocoder comb on their output "
    "(hop 480 @ 48 kHz, confirmed in their source) — upscaling adds a "
    "detectable AI fingerprint that was not there before."
)


SPECTROGRAM = os.path.join(ROOT, "spectrogram.py")
REGION_FILL = os.path.join(ROOT, "region_fill.py")
NATURALIZE = os.path.join(ROOT, "naturalize.py")

_FFMPEG = None

# ffmpeg's banner and stream dump are 30 lines per invocation, and `degrade`
# invokes it up to three times -- 90 lines of build flags burying the two lines
# that matter. `error` still surfaces a real failure, and a non-zero exit is
# checked separately anyway.
FF_QUIET = ["-y", "-hide_banner", "-loglevel", "error", "-nostats"]


def ffmpeg():
    """Resolve ffmpeg once. On PATH here, but do not assume it always is."""
    global _FFMPEG
    if _FFMPEG is None:
        import shutil
        _FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
    return _FFMPEG


# ---------------------------------------------------------------- degrade
#
# The module docstring above says upresing material with no codec cliff makes
# it worse, and that neither generator here produces one. This is the other
# half of that finding put to work: MAKE a cliff, then let the upscaler rebuild
# it. The rebuilt band is synthesised rather than merely sharpened, which is a
# different -- and often better-sounding -- result than upresing a clean file.
#
# Each upscaler was trained on a particular kind of damage, and giving it the
# damage it knows is worth more than giving it more damage:
#   Apollo            real codec artifacts  -> an MP3 round-trip
#   AudioSR, FlashSR  lowpass-filtered audio -> a lowpass
# `auto` picks on that basis.

DEGRADES = {
    "off":   "Off — upscale the file exactly as it is",
    "auto":  "Auto — the damage this upscaler was trained on",
    "cut14": "Cut above 14 kHz — gentle, leaves most of the air",
    "cut11": "Cut above 11 kHz — a big band to rebuild",
    "mp3":   "MP3 128k round-trip — real codec artifacts",
    "both":  "MP3 128k + cut above 11 kHz — the most to rebuild",
}

_AUTO = {"apollo": "mp3", "audiosr": "cut11", "flashsr": "cut11"}

# mode -> (intermediate sample rate, mp3 kbps)
#
# The band cut is a RESAMPLE round trip, not a lowpass filter. ffmpeg's
# `lowpass` is a biquad capped at 2 poles; chaining it to 4th order still only
# reaches ~24 dB/oct, and measured on a real file that left content only 7 dB
# down an octave above the corner. That is a slope, not a cliff, and the
# upscalers are trained on cliffs. Resampling to 2*fc and back puts soxr's
# anti-alias filter at Nyquist, which is a genuine brick wall -- and for
# AudioSR in particular it is literally the training condition, since it is a
# sample-rate super-resolution model.
_PLANS = {"cut14": (28000, None), "cut11": (22050, None),
          "mp3": (None, 128), "both": (22050, 128)}


def degrade_plan(mode, kind="apollo"):
    """(intermediate_sr, mp3_kbps, label). ValueError on an unknown mode."""
    if mode in (None, "", "off"):
        return None, None, "off"
    if mode == "auto":
        mode = _AUTO.get(kind, "cut11")
    plan = _PLANS.get(mode)
    if plan is None:
        raise ValueError(f"unknown degrade mode {mode!r}")
    return plan[0], plan[1], mode


def sample_rate(path):
    """The file's sample rate via ffprobe, or None."""
    probe = (ffmpeg()[:-len("ffmpeg.exe")] + "ffprobe.exe"
             if ffmpeg().lower().endswith("ffmpeg.exe") else "ffprobe")
    try:
        r = subprocess.run([probe, "-v", "error", "-select_streams", "a:0",
                            "-show_entries", "stream=sample_rate",
                            "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True, timeout=30,
                           creationflags=NOWIN)
        return int((r.stdout or "").strip())
    except Exception:
        return None


def degrade(src, out, mode, kind="apollo", on_progress=None, on_proc=None):
    """Damage `src` into `out` so the upscaler has something to restore.

    Returns the output path, or `src` unchanged when the mode is off. Never
    touches `src` -- the pristine file is what the hybrid splices against.
    """
    lo_sr, kbps, label = degrade_plan(mode, kind)
    if label == "off":
        return src

    if on_progress:
        on_progress(f"degrading: {DEGRADES.get(label, label)}")
    orig_sr = sample_rate(src) or 44100
    stem = os.path.splitext(out)[0]
    step = src
    tmps = []

    if lo_sr and lo_sr < orig_sr:
        low = f"{stem}.{lo_sr}.wav"
        cmd = [ffmpeg(), *FF_QUIET, "-i", step, "-af",
               f"aresample={lo_sr}:resampler=soxr:precision=28",
               "-ar", str(lo_sr), "-c:a", "pcm_s16le", low]
        if _stream(cmd, on_progress, on_proc=on_proc) != 0:
            raise RuntimeError(f"degrade: downsample to {lo_sr} failed")
        step, _ = low, tmps.append(low)

    if kbps:
        # Two passes, because the point is the codec's OWN artifacts: encoding
        # and decoding inside one filter graph keeps float samples throughout
        # and produces none of them.
        mp3 = f"{stem}.{kbps}k.mp3"
        enc = [ffmpeg(), *FF_QUIET, "-i", step, "-c:a", "libmp3lame",
               "-b:a", f"{kbps}k", mp3]
        if _stream(enc, on_progress, on_proc=on_proc) != 0:
            raise RuntimeError("degrade: mp3 encode failed")
        step, _ = mp3, tmps.append(mp3)

    # Back to the original rate, so everything downstream (and the splice
    # against the pristine original) still lines up sample for sample.
    back = [ffmpeg(), *FF_QUIET, "-i", step, "-af",
            f"aresample={orig_sr}:resampler=soxr:precision=28",
            "-ar", str(orig_sr), "-c:a", "pcm_s16le", out]
    if _stream(back, on_progress, on_proc=on_proc) != 0:
        raise RuntimeError("degrade: restore to source rate failed")

    for t in tmps:
        try:
            os.remove(t)
        except OSError:
            pass
    if not os.path.exists(out):
        raise RuntimeError("degrade produced no output")
    return out


# ------------------------------------------------------------- naturalize

NATURALIZERS = {
    "off": "Off",
    "subtle": "Subtle — you will only hear it in an A/B",
    "medium": "Medium — the default analogue pass",
    "strong": "Strong — obvious colour",
}


def naturalize(src, out, strength="medium", lufs=-14.0, on_progress=None,
               on_proc=None):
    """Highpass, tape-style saturation, a pink noise floor, loudness match.

    See naturalize.py for what each stage is for and why the saturation is
    oversampled. Returns the output path, or `src` when the strength is off.
    """
    if strength in (None, "", "off"):
        return src
    if strength not in NATURALIZERS:
        raise ValueError(f"unknown naturalize strength {strength!r}")
    cmd = [WATERMARK_PY, NATURALIZE, "-i", src, "-o", out,
           "--strength", strength, "--lufs", str(lufs if lufs else 0)]
    if _stream(cmd, on_progress, on_proc=on_proc) != 0:
        raise RuntimeError(f"naturalize ({strength}) failed")
    if not os.path.exists(out):
        raise RuntimeError("naturalize produced no output")
    return out


def cliff_of(path):
    """(cliff_hz, drop_db) parsed from check_bandwidth's report line."""
    import re
    m = re.search(r"cliff at ~(\d+) Hz\s+\(drop ([-\d.]+) dB",
                  bandwidth(path) or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


SPECVIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "specview.py")


def specview(src, out_png, region=None, marker=None):
    """Render one panel and return its geometry dict (see specview.py).

    The geometry travels with the image so a click can be mapped to (time,
    frequency) by arithmetic -- it must be stored alongside the PNG it came
    from, never cached globally, or a click can map through a stale render.
    """
    cmd = [WATERMARK_PY, SPECVIEW, "--input", src, "--out", out_png]
    if region:
        cmd += ["--region", region]
    if marker:
        cmd += ["--marker", marker]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, creationflags=NOWIN)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "")[-1500:])
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def spectrogram(panels, out_png, regions=(), dur=20.0, fmax=None,
                title="Spectrogram", subtitle=""):
    """Render stacked panels via spectrogram.py. `panels` is [(label, path)]."""
    cmd = [WATERMARK_PY, SPECTROGRAM, "-o", out_png, "--dur", str(dur),
           "--title", title]
    if subtitle:
        cmd += ["--subtitle", subtitle]
    if fmax:
        cmd += ["--fmax", str(fmax)]
    for r in regions:
        cmd += ["--region", r]
    for label, path in panels:
        cmd.append(f"{label}={path}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=600, creationflags=NOWIN)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise RuntimeError((r.stdout or "") + (r.stderr or "") or "render failed")
    return out_png


def region_fill(orig, restored, out_path, regions, on_progress=None,
                on_proc=None):
    """Splice `restored` into `orig` over t0:t1:flo:fhi regions. CPU only."""
    cmd = [WATERMARK_PY, REGION_FILL, "--orig", orig, "--restored", restored,
           "-o", out_path]
    for r in regions:
        cmd += ["--region", r]
    code = _stream(cmd, on_progress, on_proc=on_proc)
    if code != 0:
        raise RuntimeError(f"region_fill failed (exit {code})")
    return out_path


def _audio_files(root):
    """Every audio file under `root`, recursively -- MSST nests its output."""
    out = []
    for ext in ("wav", "flac"):
        out += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    return out


def bandwidth(path):
    """Run check_bandwidth.py and return its report text.

    Uses the watermark venv, which already has numpy/soundfile, so the studio
    venv stays free of any audio stack.
    """
    if not path or not os.path.exists(path):
        return "(no file)"
    try:
        r = subprocess.run([WATERMARK_PY, CHECK_BW, path],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180, creationflags=NOWIN)
        return (r.stdout or r.stderr or "").strip() or "(no output)"
    except Exception as e:
        return f"(bandwidth check failed: {type(e).__name__}: {e})"


def _stream(cmd, on_progress, cwd=None, on_proc=None):
    """Run a command, streaming merged stdout+stderr. Returns the exit code.

    stderr is merged rather than piped separately because these tools emit tqdm
    and DDIM bars continuously; one pipe means no drain thread and no chance of
    a full buffer blocking the child.

    `on_proc(p)` hands the caller the live Popen so a long run can be killed --
    without it there is no way to stop an upscale once it has started.
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1,
                         cwd=cwd, creationflags=NOWIN)
    if on_proc:
        on_proc(p)
    last = ""
    for line in p.stdout:
        line = line.rstrip()
        # Progress bars redraw constantly; forward only meaningful changes.
        if not line or line == last:
            continue
        last = line
        if on_progress:
            on_progress(line[:160])
    p.wait()
    return p.returncode


def upscale(kind, src, outdir, on_progress=None, auto_lowpass=True,
            on_proc=None):
    """Run one upscaler. Returns the path to the produced file."""
    if kind == "none":
        return src
    if kind not in UPSCALERS:
        raise ValueError(f"unknown upscaler {kind!r}")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src))[0]
    t0 = time.time()

    if kind == "apollo":
        # Apollo's -Output is a FOLDER, and MSST nests one level deeper still:
        # it writes <outdir>\<input stem>\restored.wav. So the search has to be
        # recursive, and the result is normalised to a flat name to match the
        # other two tools. Apollo also resamples 48 kHz down to 44.1 kHz.
        before = set(_audio_files(outdir))
        code = _stream(PS + [os.path.join(ROOT, "apollo.ps1"),
                             "-InputPath", src, "-Output", outdir],
                       on_progress, on_proc=on_proc)
        if code != 0:
            raise RuntimeError(f"Apollo failed (exit {code})")
        new = sorted(set(_audio_files(outdir)) - before, key=os.path.getmtime)
        if not new:
            raise RuntimeError("Apollo produced no output file")
        out = os.path.join(outdir, f"{stem}_apollo.wav")
        if os.path.exists(out):
            os.remove(out)
        os.replace(new[-1], out)
        nested = os.path.dirname(new[-1])
        if os.path.normpath(nested) != os.path.normpath(outdir):
            try:
                os.rmdir(nested)
            except OSError:
                pass
    else:
        out = os.path.join(outdir, f"{stem}_{kind}.wav")
        if kind == "audiosr":
            cmd = PS + [os.path.join(ROOT, "audiosr.ps1"),
                        "-InputPath", src, "-Output", out]
            if auto_lowpass:
                # AudioSR was trained on lowpass-filtered audio only; fed a raw
                # lossy file it hallucinates from codec artifacts.
                cmd.append("-AutoLowpass")
        else:
            cmd = [FLASHSR_PY, FLASHSR_SCRIPT, "-i", src, "-o", out]
        code = _stream(cmd, on_progress, on_proc=on_proc)
        if code != 0:
            raise RuntimeError(f"{kind} failed (exit {code})")
        if not os.path.exists(out):
            raise RuntimeError(f"{kind} produced no output at {out}")

    if on_progress:
        on_progress(f"{kind} finished in {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="headless upscaler test")
    ap.add_argument("src")
    ap.add_argument("--kind", choices=list(UPSCALERS), default="apollo")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "Music", "studio", "upscaled"))
    ap.add_argument("--no-lowpass", action="store_true")
    a = ap.parse_args()

    print("=== before ===")
    print(bandwidth(a.src))
    out = upscale(a.kind, a.src, a.outdir, on_progress=lambda m: print(" ..", m),
                  auto_lowpass=not a.no_lowpass)
    print("\n=== after ===")
    print(out)
    print(bandwidth(out))
