"""Lyrics for a cover source: find what already exists, or transcribe it.

Both steps are subprocesses in other venvs -- the probe needs `mutagen` (in the
watermark venv), and transcription needs torch + demucs + faster-whisper (its
own venv). The studio venv keeps no audio stack.
"""

import json
import os
import subprocess

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

WATERMARK_PY = os.path.join(ROOT, "watermark", ".venv", "Scripts", "python.exe")
LYRICS_PY = os.path.join(ROOT, "lyrics", ".venv", "Scripts", "python.exe")
PROBE = os.path.join(ROOT, "lyrics_probe.py")
TRANSCRIBE = os.path.join(ROOT, "transcribe.py")

WHISPER_MODELS = ["large-v3", "medium", "small"]


def probe(path):
    """{'found', 'best': {'source','field','text'}, 'embedded', 'sidecars'}."""
    if not path or not os.path.exists(path):
        return {"ok": False, "found": False, "error": "no such file"}
    try:
        r = subprocess.run([WATERMARK_PY, PROBE, path], capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=120, creationflags=NOWIN)
        if r.returncode != 0:
            return {"ok": False, "found": False,
                    "error": (r.stderr or "")[-400:]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"ok": False, "found": False,
                "error": f"{type(e).__name__}: {e}"}


def describe(res, uploaded_only):
    """One status line the user can act on."""
    if not res.get("ok"):
        return f"Lyrics probe failed: {res.get('error', 'unknown')}"
    if res.get("found"):
        b = res["best"]
        where = ("embedded tag " + b["field"] if b["source"] == "embedded"
                 else "file " + b["field"])
        why = f" — {b['why']}" if b.get("why") else ""
        n = len([l for l in b["text"].splitlines() if l.strip()])
        return f"**Found lyrics** in {where}{why} ({n} lines). Apply to use."
    tail = ""
    if uploaded_only:
        # Browsers never send the original directory, so an uploaded file has
        # no neighbours to search -- say so rather than implying none exist.
        tail = ("  \n*Sidecar files can't be seen for an upload — the browser "
                "only sends a copy. Paste the file's real path to search "
                "beside it.*")
    return "No lyrics found in tags or beside the file. Use Transcribe." + tail


def transcribe(path, separate=True, model="large-v3", seconds=0.0,
               on_progress=None, on_proc=None):
    """Return transcribed text. Streams progress lines to `on_progress`.

    `on_proc(p)` hands the caller the live Popen so the run can be cancelled.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path or "(no file)")
    cmd = [LYRICS_PY, TRANSCRIBE, "--input", path, "--model", model, "--json"]
    if not separate:
        cmd.append("--no-separate")
    if seconds:
        cmd += ["--seconds", str(seconds)]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1, creationflags=NOWIN)
    if on_proc:
        on_proc(p)
    # stderr carries progress, stdout carries the single JSON result, so they
    # are read separately here rather than merged.
    import threading
    err = []

    def drain():
        for line in p.stderr:
            line = line.rstrip()
            if line:
                err.append(line)
                if on_progress:
                    on_progress(line[:160])

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    out = p.stdout.read()
    p.wait()
    t.join(timeout=5)
    if p.returncode != 0:
        raise RuntimeError("\n".join(err[-12:]) or "transcription failed")
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        raise RuntimeError("\n".join(err[-12:]) or "no result from transcriber")
