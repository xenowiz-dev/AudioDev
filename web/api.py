"""AudioDev Studio — HTTP/SSE API (v1).

The replacement view layer for `studio/app.py`. Everything that does real work
already lives in the portable modules under `studio\\`; this file is
transport only — validation, JSON shapes, the one GPU lane, and byte-ranged
media. Nothing here reimplements a generator, an upscaler or a sidecar.

Run it:  .\\web\\serve.ps1   (studio venv, uvicorn on 127.0.0.1:7862)

Three shapes of work, three treatments:

  * instant / slow-sync endpoints are plain `def`, so FastAPI runs them in the
    threadpool and a 6 s matplotlib render cannot freeze the event loop;
  * GPU work (generate / upscale / transcribe) is a job on ONE queue, created
    by a POST that returns 202 and observed over a re-attachable SSE stream;
  * CPU subprocesses in the other venvs (bandwidth, specview, covers, probe,
    titles) deliberately run OFF the GPU lane, so the library stays browsable
    while a generation is in flight.
"""

import glob
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

# The portable modules, imported -- never copied. Order matters: `playlists`
# imports `library`, and both must resolve to the studio copy.
sys.path.insert(0, os.path.join(ROOT, "studio"))
sys.path.insert(0, os.path.join(ROOT, "minimax"))

from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

import ar_cache                                     # noqa: E402
import art                                          # noqa: E402
import library as lib                               # noqa: E402
import loras as lo                                  # noqa: E402
import llm_title                                    # noqa: E402
import lyrics as ly                                 # noqa: E402
import playlists as pls                             # noqa: E402
import postprocess as pp                            # noqa: E402
import titles                                       # noqa: E402
import settings as prof                             # noqa: E402
import workspaces as wsp                            # noqa: E402
import writer as wr                                 # noqa: E402
from supervisor import BACKENDS, Supervisor, gpu_memory   # noqa: E402

import jobs as J                                    # noqa: E402

# --------------------------------------------------------------------- paths

STUDIO = os.path.join(ROOT, "studio")
OUTDIR = lib.OUTDIR
UPDIR = os.path.join(OUTDIR, "upscaled")
COVERDIR = os.path.join(OUTDIR, "covers")
TRASHDIR = os.path.join(OUTDIR, "trash")
UPLOADDIR = os.path.join(OUTDIR, "uploads")
ASSETS = os.path.join(STUDIO, "assets")
WEB = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(WEB, "static")

PREVIEW_SECONDS = 15.0
UPLOAD_MB = 200
UPLOAD_EXT = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
SPEC_TTL = 30 * 60          # _spec_*.png older than this are swept
UPLOAD_TTL = 7 * 24 * 3600

SUP = Supervisor()
REG = J.Registry()
START = time.time()
TEST_HOOKS = os.environ.get("STUDIO_TEST_HOOKS") == "1"

MINIMAX_PROMPT = """Global Metadata
Basic Attributes: bpm is 96. key is C, and scale is major. Acoustic Pop.
Sonics & Production Profile: warm, intimate, lightly compressed, natural room.
Vocal Details
Vocal Gender & Timbre: Singer A (Female). Soft, close, breathy head voice.
Vocal Style: gentle and conversational in the verse, opening up in the chorus.
Arrangement
Primary: fingerpicked steel-string acoustic guitar throughout.
Secondary: soft piano pads; brushed drums and upright bass enter at the chorus."""

ACESTEP_PROMPT = ("upbeat indie pop, jangly electric guitar, live drums, "
                  "warm analog production, 110 bpm")

DEFAULT_LYRICS = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""

PROMPT_META = {
    "minimax": dict(
        requires_lyrics=True, supports_instrumental=False, supports_cover=False,
        default_steps=30, default_prompt=MINIMAX_PROMPT, prompt_lines=10,
        prompt_info="MiniMax was trained on sectioned captions — keep the "
                    "Global Metadata / Vocal Details / Arrangement headings; "
                    "short prompts lose arrangement control.",
        lyrics_info="Required — MiniMax has no instrumental mode."),
    "acestep": dict(
        requires_lyrics=False, supports_instrumental=True, supports_cover=True,
        default_steps=8, default_prompt=ACESTEP_PROMPT, prompt_lines=4,
        prompt_info="Plain keyword-style prompt.",
        lyrics_info="Leave blank, or tick Instrumental above."),
}

NOTES = {
    "post_kind": "Runs after generation. Measured: neither generator leaves a "
                 "codec cliff (MiniMax 0.6 dB, ACE-Step 3.9 dB), and upresing "
                 "audio with no cliff makes it worse — so this is normally for "
                 "lossy source material, on the Restore tab. The bandwidth "
                 "verdict is logged either way.",
    "cover": "Upload a track to regenerate it in the style above. Noise "
             "strength is the parameter that makes it a cover — 0.0 means pure "
             "noise and produces an unrelated song. 0.4–0.8.",
    "auto_lowpass": "AudioSR was trained on lowpass-filtered audio only; fed a "
                    "raw lossy file it hallucinates from codec artifacts. Leave "
                    "on for anything compressed.",
    "region_fill": "An upscaler rewrites the whole file (~13% relative error "
                   "below the crossover); splicing adds only the masked "
                   "correction, so everything outside the box stays "
                   "bit-identical.",
    "sidecar": "Each track keeps a JSON sidecar holding its prompt, lyrics and "
               "settings, so the record travels with the file. Not written into "
               "the audio's tags. Cover art is rendered from each track's own "
               "spectrogram. Trash is a folder move, never a delete.",
}

# ------------------------------------------------- quality (REQUIREMENTS 6)
#
# Everything done to squeeze MiniMax onto a 10 GB card is a QUALITY TRADE, and
# on a bigger card it should simply be switched off. These are the knobs, and
# the arithmetic that says whether a given combination can actually run.
#
# Two of them are not the same kind of setting, and the UI has to say so:
#   * llm / rvq precision is baked into the weights AS THEY LOAD -- changing
#     either means restarting the worker, ~20 s.
#   * kv cache mode and the offload reserve are per request, and free.

# MEASURED footprints in GB, not derived: 8,584,475,648 LLM params and
# 646,025,216 RVQ params, weighed on this checkpoint. The AR stage drives both
# on every frame and refuses to run unless they are co-resident, so their SUM
# is the floor every budget question starts from.
LLM_GB = {"4bit": 6.29, "8bit": 8.47, "bf16": 15.99}
RVQ_GB = {"4bit": 0.33, "8bit": 0.64, "bf16": 1.20}

# The KV cache is a STORAGE decision, not a history one: "quantized" keeps the
# whole song's history and stores what has aged out in 4 bits. "static" and
# "dynamic" differ in how they allocate, not in how much they hold -- both are
# 16-bit, which is why only "quantized" changes the number below.
KV_BITS = {"quantized": 4, "static": 16, "dynamic": 16, "auto": 16}

QUALITY_FIELDS = {
    "llm": ["4bit", "8bit", "bf16"],
    "rvq": ["4bit", "8bit", "bf16"],
    "kv": ["auto", "quantized", "static", "dynamic"],
    "reserve": ["0.5GB", "1GB", "2GB"],
}

QUALITY_COSTS = {
    "llm": "musicality and prompt adherence",
    "rvq": "acoustic detail",
    "kv": "storage only - full history, stored coarser",
}

# What a 4-minute song needs, GB. These are the published measured figures
# (web/REQUIREMENTS.md 6); `budget_for` below recomputes the same arithmetic
# live for whatever duration the user actually picked, and the two agree to
# within a rounding step.
QUALITY_PRESETS = [
    {"id": "smallest", "label": "Smallest",
     "llm": "4bit", "rvq": "4bit", "kv": "quantized", "needs_4min_gb": 7.87,
     "note": "Lowest floor. Both halves of the AR pair quantized — it starts "
             "when nothing else will, at some cost to detail."},
    {"id": "balanced", "label": "Balanced",
     "llm": "4bit", "rvq": "8bit", "kv": "quantized", "needs_4min_gb": 8.17,
     "note": "Half the RVQ trade of Smallest, for 0.3 GB more."},
    {"id": "default10", "label": "Default (10 GB)",
     "llm": "4bit", "rvq": "bf16", "kv": "quantized", "needs_4min_gb": 8.74,
     "note": "The verified 10 GB setting: full-precision acoustics, and the "
             "cache is the only thing given up."},
    {"id": "fullkv", "label": "Full cache (12 GB+)",
     "llm": "4bit", "rvq": "bf16", "kv": "static", "needs_4min_gb": 10.18,
     "note": "Nothing quantized but the LLM. Needs ~1.4 GB more than Default "
             "for a 4-minute song, and more again as songs get longer."},
    {"id": "high", "label": "High (16 GB+)",
     "llm": "8bit", "rvq": "bf16", "kv": "static", "needs_4min_gb": 12.37,
     "note": "8-bit LLM — the first real step up in musicality and prompt "
             "adherence."},
    {"id": "maximum", "label": "Maximum (24 GB+)",
     "llm": "bf16", "rvq": "bf16", "kv": "dynamic", "needs_4min_gb": 19.88,
     "note": "No quantization anywhere. The biggest quality lever is LLM "
             "precision and this is the only preset that pulls it fully."},
]

PRESET_BY_ID = {p["id"]: p for p in QUALITY_PRESETS}

# What the worker does when nobody asks: 4-bit LLM, unquantized RVQ, the cache
# chosen from free VRAM at request time, 1 GB reserve. `default10` in all but
# the cache, which "auto" resolves to `quantized` exactly when it has to.
QUALITY_DEFAULTS = {"llm": "4bit", "rvq": "bf16", "kv": "auto",
                    "reserve": "1GB"}

QUALITY_RESTART_NOTE = (
    "LLM and RVQ precision are applied while the weights load, so changing "
    "either restarts MiniMax — about 20 s before the next song starts. KV "
    "cache mode and the offload reserve are per request and cost nothing.")

# Case-insensitive lookup so "1gb" and "1GB" are the same answer.
_FIELD_LOOKUP = {f: {v.lower(): v for v in vs}
                 for f, vs in QUALITY_FIELDS.items()}


def recommended_preset(total_gb):
    """The preset a card of this size should start on.

    A 24 GB card must not silently inherit a 10 GB compromise -- that is the
    whole point of the item. With no readable GPU we recommend the verified
    10 GB setting, which is the one that runs everywhere.
    """
    if not total_gb:
        return "default10"
    if total_gb < 11:
        return "default10"
    if total_gb < 14:
        return "fullkv"
    if total_gb < 20:
        return "high"
    return "maximum"


def quality_catalogue():
    used, total = gpu_memory()
    free = None if used is None else round(total - used, 2)
    return {
        "detected_total_gb": round(total, 2) if total else None,
        "detected_free_gb": free,
        "recommended": recommended_preset(total),
        "applies_to": ["minimax"],
        "presets": [dict(p) for p in QUALITY_PRESETS],
        "fields": {k: list(v) for k, v in QUALITY_FIELDS.items()},
        "costs": dict(QUALITY_COSTS),
        "defaults": dict(QUALITY_DEFAULTS),
        "restart_note": QUALITY_RESTART_NOTE,
    }


def quality_field(field, value, default=None):
    """One quality setting, validated.

    Silently ignoring a value we do not understand is the one thing this must
    not do: the budget is computed FROM these, so an ignored setting produces a
    confident answer to a question nobody asked.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if not isinstance(value, str):
        raise bad(f"{field} must be a string.")
    got = _FIELD_LOOKUP[field].get(value.strip().lower())
    if got is None:
        raise bad(f"Unknown {field} setting {value!r}.",
                  {"field": field, "allowed": QUALITY_FIELDS[field]})
    return got


def resolve_quality(raw):
    """A request's `quality` object -> the four settled settings, or None.

    None means the client asked for nothing, which is NOT the same as asking
    for the defaults: it means "whatever is loaded is fine", so a plain
    generate never costs a 20 s reload. That is the path every pre-quality
    client -- including the old gradio app -- takes.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise bad("quality must be an object.")
    wanted = dict(QUALITY_DEFAULTS)
    pid = raw.get("preset")
    if isinstance(pid, str) and pid.strip() and pid.strip() != "custom":
        preset = PRESET_BY_ID.get(pid.strip())
        if preset is None:
            raise bad(f"Unknown quality preset {pid!r}.",
                      {"allowed": [p["id"] for p in QUALITY_PRESETS]
                                  + ["custom"]})
        wanted.update(llm=preset["llm"], rvq=preset["rvq"], kv=preset["kv"])
    elif pid is not None and not isinstance(pid, str):
        raise bad("quality.preset must be a string.")

    out = {f: quality_field(f, raw.get(f), wanted[f])
           for f in ("llm", "rvq", "kv", "reserve")}
    # An explicit field that disagrees with the named preset wins, and the
    # answer stops calling itself by that preset's name.
    named = PRESET_BY_ID.get((pid or "").strip() if isinstance(pid, str)
                             else "")
    same = named and all(named[f] == out[f] for f in ("llm", "rvq", "kv"))
    out["preset"] = named["id"] if same else "custom"
    return out


def budget_for(duration, free_gb, llm, rvq, kv):
    """`ar_cache.budget` asked with the user's OWN settings.

    base_gb is the co-resident AR pair at the chosen precisions; kv_bits is 4
    only for the quantized cache. The defaults reproduce the module's own
    (base 7.49 = 4-bit LLM + bf16 RVQ, kv_bits 16), so an unparameterised call
    answers exactly what it answered before this existed.
    """
    base = LLM_GB[llm] + RVQ_GB[rvq]
    need, fits, head = ar_cache.budget(float(duration or 0), free_gb,
                                       base_gb=base, kv_bits=KV_BITS[kv])
    return need, fits, head, base


MEDIA_TYPES = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4", ".aac": "audio/mp4", ".ogg": "audio/ogg",
    ".opus": "audio/ogg", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".webp": "image/webp", ".json": "application/json",
    ".txt": "text/plain",
}


# ------------------------------------------------------------------- errors

class ApiError(Exception):
    def __init__(self, status, code, message, detail=None):
        super().__init__(message)
        self.status, self.code, self.message, self.detail = (
            status, code, message, detail)


def _envelope(status, code, message, detail=None):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message,
                           "detail": detail}})


def bad(message, detail=None):
    return ApiError(400, "validation", message, detail)


def missing(message="Not found.", detail=None):
    return ApiError(404, "not_found", message, detail)


# ------------------------------------------------------------------ helpers

def media_url(path):
    if not path:
        return None
    from urllib.parse import quote
    return "/api/media?p=" + quote(str(path), safe="")


def _norm(p):
    return os.path.normcase(os.path.realpath(p))


def _under(path, root):
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + os.sep)


def playable(path):
    """The /api/media allowlist: gradio's `allowed_paths=[OUTDIR, ASSETS]`."""
    try:
        return bool(path) and (_under(path, OUTDIR) or _under(path, ASSETS))
    except Exception:
        return False


def resolve_id(track_id):
    """id (a basename) -> absolute path, or 404. Never escapes OUTDIR."""
    tid = (track_id or "").strip()
    if (not tid or "/" in tid or "\\" in tid or ".." in tid or ":" in tid
            or tid in (".", "..")):
        raise missing("Unknown track.")
    p = os.path.join(OUTDIR, tid)
    if not os.path.isfile(p):
        raise missing(f"No track named {tid}.")
    return p


def path_arg(v):
    """A JSON `path` field as a string.

    A client that sends a number, a list or an object has not sent a path, and
    that is a 400 from the caller's own "no file" branch -- never a 500 out of
    `int.strip`. Every path-taking endpoint funnels through here.
    """
    return v.strip().strip('"') if isinstance(v, str) else ""


def text_arg(v, field):
    """A JSON free-text field. Absent reads as empty; wrong type is a 400."""
    if v is None:
        return ""
    if not isinstance(v, str):
        raise bad(f"{field} must be a string.")
    return v


def num_arg(v, default, cast=float):
    """A JSON number field, or `default`. Garbage is `Out of range.`, not a 500."""
    if v is None:
        return default
    if isinstance(v, bool):
        raise bad("Out of range.")
    try:
        return cast(v)
    except (TypeError, ValueError):
        raise bad("Out of range.")


def region_spec(r):
    """The four UI numbers -> one `t0:t1:flo:fhi` string (spec 6.1).

    Validated here rather than in the job: a bad number must be a 400 at the
    POST, not a 202 that dies on the GPU lane a minute later.
    """
    r = r if isinstance(r, dict) else {}
    t0 = num_arg(r.get("t0") or 0, 0.0)
    t1 = r.get("t1")
    flo = num_arg(r.get("flo") or 0, 0.0)
    fhi = r.get("fhi")
    end = "*" if not t1 or num_arg(t1, 0.0) <= 0 else f"{num_arg(t1, 0.0):.3f}"
    hi = "*" if not fhi or num_arg(fhi, 0.0) <= 0 else f"{num_arg(fhi, 0.0):.0f}"
    return f"{t0:.3f}:{end}:{flo:.0f}:{hi}"


def spec_png():
    # A fresh clone has no Music\studio yet: create the library root before
    # anything lists it, or the first request on a new box is a 500.
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(UPDIR, exist_ok=True)
    return os.path.join(UPDIR, f"_spec_{uuid.uuid4().hex[:12]}.png")


def vram_snapshot():
    used, total = gpu_memory()
    loaded = SUP.current()
    if used is None:
        return {"available": False, "used_gb": None, "total_gb": None,
                "free_gb": None, "loaded": loaded,
                "loaded_label": BACKENDS[loaded]["label"] if loaded else None,
                "warn_low_free": False}
    free = total - used
    return {"available": True,
            "used_gb": round(used, 2), "total_gb": round(total, 2),
            "free_gb": round(free, 2),
            "loaded": loaded,
            "loaded_label": BACKENDS[loaded]["label"] if loaded else None,
            "warn_low_free": bool(loaded is None and free < 8.0)}


J.VRAM_PROBE = vram_snapshot

_llm_cache = {"t": 0.0, "ready": False}


def llm_ready():
    now = time.time()
    if now - _llm_cache["t"] > 60:
        try:
            _llm_cache["ready"] = bool(llm_title.is_ready())
        except Exception:
            _llm_cache["ready"] = False
        _llm_cache["t"] = now
    return _llm_cache["ready"]


# --------------------------------------------------------- library entries

def rating_of(entry):
    """Tri-state rating, from rowview._rating -- kept in exactly one place."""
    if entry.get("shared_rec"):
        return 0        # a verdict must not ride on a lent record
    r = (entry.get("rec") or {}).get("rating")
    if isinstance(r, bool):
        val = 1 if r else 0
    elif isinstance(r, (int, float)):
        val = 1 if r > 0 else (-1 if r < 0 else 0)
    elif isinstance(r, str):
        s = r.strip().lower()
        val = 1 if s in ("up", "like", "liked", "+1", "1") else (
            -1 if s in ("down", "dislike", "disliked", "-1") else 0)
    else:
        val = 0
    if val == 0 and (entry.get("rec") or {}).get("favorite"):
        val = 1
    return val


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def api_entry(e):
    """The list/detail object of spec 8.1 -- complete enough to render both."""
    rec = e.get("rec") or {}
    path = e["path"]
    inherited = bool(rec.get("inherited_from"))
    shared = bool(e.get("shared_rec")) or inherited
    recorded = bool(e.get("recorded")) and not inherited
    fav = (pls.is_favorite(path) if e.get("shared_rec")
           else bool(rec.get("favorite")))
    title = titles.derive(rec, e["name"])
    cover = art.cover_path(path)
    seconds = _num(rec.get("seconds"))
    bits = []
    if (e.get("takes") or 1) > 1:
        bits.append(f"take {e['take']}/{e['takes']}")
    if seconds:
        bits.append(f"{seconds:.0f}s")
    if rec.get("model"):
        bits.append(str(rec["model"]))
    cap_title = ("★ " if fav else "") + title
    caption = f"{cap_title}\n{' · '.join(bits)}" if bits else cap_title
    try:
        created = rec.get("created") or datetime.fromtimestamp(
            e["mtime"]).isoformat(timespec="seconds")
    except Exception:
        created = None
    return {
        "id": e["name"],
        "name": e["name"],
        "path": path,
        "title": title,
        "audio_url": media_url(path),
        "cover_url": media_url(cover) if os.path.exists(cover) else None,
        "mtime": e["mtime"],
        "created": created,

        "recorded": recorded,
        "shared_rec": shared,
        "take": e.get("take", 1),
        "takes": e.get("takes", 1),

        "favorite": fav,
        "rating": rating_of(e),

        "workspace": rec.get("workspace") or None,
        "model": rec.get("model"),
        "prompt": rec.get("prompt"),
        "lyrics": rec.get("lyrics"),
        "instrumental": bool(rec.get("instrumental")),
        "duration": _num(rec.get("duration")),
        "steps": _num(rec.get("steps")),
        "seed": _num(rec.get("seed")),
        "seconds": seconds,
        "sampling_rate": _num(rec.get("sampling_rate")),
        "elapsed": _num(rec.get("elapsed")),
        "vram_peak": _num(rec.get("vram_peak")),
        "preview": bool(rec.get("preview")),
        "post_kind": rec.get("post_kind"),
        "cover_src": rec.get("cover_src"),
        "cover_strength": _num(rec.get("cover_strength")),
        "noise_strength": _num(rec.get("noise_strength")),

        # Provenance (REQUIREMENTS 11, 13). Missing MUST arrive as null, never
        # "" and never "none": 91 tracks predate these features, and the record
        # sheet omits null values -- a printed "none" would assert something
        # false about how one of them was made. `quality` is the object the
        # generate request sent; the client reads `.preset`.
        "lora": rec.get("lora") or None,
        "lora_name": rec.get("lora_name") or None,
        "lora_scale": _num(rec.get("lora_scale")),
        "variant": rec.get("variant") or None,
        # Tri-state on purpose: None means "this track predates the field",
        # False means "thinking was deliberately off". `bool(x) or None`
        # collapsed both to null, which is information loss in the one
        # payload whose job is provenance.
        "thinking": (None if rec.get("thinking") is None
                     else bool(rec.get("thinking"))),
        "lm_temperature": _num(rec.get("lm_temperature")),
        "quality": rec.get("quality") or None,

        "title_override": rec.get("title", "") or "",
        "style": rec.get("style", "") or "",
        "caption": caption,

        # Extension (web/REQUIREMENTS.md 2): played state lives in the sidecar
        # so the phone and the desktop agree about what has been heard.
        "played": bool(rec.get("played")),
        "plays": int(rec.get("plays") or 0),
        "last_played": rec.get("last_played"),
    }


def find_entry(track_id, entries=None):
    es = entries if entries is not None else lib.entries()
    for e in es:
        if e["name"] == track_id:
            return es, e
    raise missing(f"No track named {track_id}.")


def donor_name(entries, entry):
    """The sibling whose record this take is borrowing.

    `group_takes` assigns the DONOR'S dict object to the borrower, so identity
    finds it exactly -- no re-derivation of the take window here.
    """
    for e in entries:
        if e is not entry and e.get("recorded") and e.get("rec") is entry.get("rec"):
            return e["name"]
    return None


def ensure_own_record(entries, entry):
    """Give a take that is borrowing its sibling's record a record of its own.

    Spec 8.8: any write to a shared_rec row creates a sidecar, after which
    `group_takes` stops lending -- and the borrowed prompt would vanish from
    that row. Seeding it from the donor first is what keeps the prompt visible.
    """
    if not entry.get("shared_rec") or lib.read(entry["path"]):
        return False
    donor = dict(entry.get("rec") or {})
    for k in ("audio", "audio_path", "favorite", "rating"):
        donor.pop(k, None)
    donor["inherited_from"] = donor_name(entries, entry) or ""
    lib.write(entry["path"], **donor)
    return True


def entry_by_id(track_id):
    resolve_id(track_id)
    es, e = find_entry(track_id)
    return es, e


def fresh(track_id):
    """Re-read one row after a mutation."""
    _es, e = find_entry(track_id)
    return api_entry(e)


# ------------------------------------------------------- cover backfill

_cover_lock = threading.Lock()


def kick_covers(paths, force=False, cap=60):
    """Render missing covers in the background. Never blocks a response."""
    if not paths or _cover_lock.locked():
        return 0
    todo = list(paths)[:cap]

    def work():
        with _cover_lock:
            for p in todo:
                try:
                    art.ensure_cover(p, force=force)
                except Exception:
                    pass
    threading.Thread(target=work, daemon=True).start()
    return len(todo)


# ------------------------------------------------------------------- app

@asynccontextmanager
async def lifespan(_app):
    for d in (OUTDIR, UPDIR, COVERDIR, UPLOADDIR, STATIC):
        os.makedirs(d, exist_ok=True)
    sweep_uploads()
    sweep_specs()
    REG.start()
    threading.Thread(target=_sweeper, daemon=True).start()
    try:
        yield
    finally:
        # A worker left alive holds 8 GB. This is the one teardown that matters.
        try:
            SUP.unload()
        except Exception:
            pass
        REG.stop()


app = FastAPI(title="AudioDev Studio API", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(ApiError)
async def _api_error(_request, exc):
    return _envelope(exc.status, exc.code, exc.message, exc.detail)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request, exc):
    code = {400: "bad_request", 404: "not_found", 409: "gpu_busy",
            413: "too_large", 422: "unmeasurable"}.get(exc.status_code,
                                                       "internal")
    return _envelope(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(_request, exc):
    return _envelope(400, "validation", "Malformed request body.",
                     json.loads(json.dumps(exc.errors(), default=str)))


@app.exception_handler(Exception)
async def _unhandled(_request, exc):
    return _envelope(500, "internal", f"{type(exc).__name__}: {exc}")


def _sweeper():
    while True:
        time.sleep(600)
        try:
            sweep_specs()
        except Exception:
            pass


def sweep_specs():
    now = time.time()
    for p in glob.glob(os.path.join(UPDIR, "_spec_*.png")):
        try:
            if now - os.path.getmtime(p) > SPEC_TTL:
                os.remove(p)
        except OSError:
            pass


def sweep_uploads():
    now = time.time()
    for p in glob.glob(os.path.join(UPLOADDIR, "*")):
        try:
            if now - os.path.getmtime(p) > UPLOAD_TTL:
                os.remove(p)
        except OSError:
            pass


# ============================================================ 4. config etc.

# Feature blocks that carry an `applies_to`. Named here so the capability map
# is built from the SAME data the client reads, rather than a second list that
# drifts the first time a feature changes hands.
_GATED_FEATURES = ("thinking", "auto_duration", "takes", "variants",
                   "loras", "quality")


def _capabilities(cfg):
    """What each backend can actually do.

    Composed from the feature blocks in the config, never restated: the answer
    to "can this model take a LoRA" has exactly one source, and adding a
    feature block with an `applies_to` is all it takes to gate a control.
    """
    out = {}
    for name, b in cfg["backends"].items():
        caps = {f: (name in (cfg[f].get("applies_to") or []))
                for f in _GATED_FEATURES}
        caps.update(cover=bool(b.get("supports_cover")),
                    instrumental=bool(b.get("supports_instrumental")),
                    lyrics_required=bool(b.get("requires_lyrics")))
        out[name] = caps
    return out


@app.get("/api/config")
def get_config():
    backends = {}
    for name, cfg in BACKENDS.items():
        meta = PROMPT_META.get(name, {})
        backends[name] = {"label": cfg["label"], "note": cfg["note"],
                          "vram_gb": cfg["vram_gb"], **meta}
    phone = None
    url_file = os.path.join(ASSETS, "phone_url.txt")
    qr_file = os.path.join(ASSETS, "phone_qr.png")
    if os.path.exists(url_file):
        try:
            with open(url_file, encoding="utf-8") as fh:
                phone = {"url": fh.read().strip(),
                         "qr_url": media_url(qr_file)
                         if os.path.exists(qr_file) else None}
        except OSError:
            phone = None
    cfg = {
        "version": 1,
        "outdir": OUTDIR,
        "default_model": "minimax",
        "backends": backends,
        "default_lyrics": DEFAULT_LYRICS,
        "upscalers": dict(pp.UPSCALERS),
        "degrades": dict(pp.DEGRADES),
        "naturalizers": dict(pp.NATURALIZERS),
        "fingerprint_note": pp.FINGERPRINT_NOTE,
        "whisper_models": list(ly.WHISPER_MODELS),
        "defaults": {"duration": 30, "steps": 30, "seed": 7,
                     "instrumental": False, "post_kind": "none",
                     "cover_strength": 1.0, "noise_strength": 0.75,
                     "upscaler": "apollo", "auto_lowpass": True, "fill": True,
                     "degrade": "off", "naturalize": "off", "lufs": -14.0,
                     "region": {"t0": 0, "t1": 0, "flo": 16000, "fhi": 0},
                     "whisper_model": "large-v3", "separate": True,
                     "seconds": 0},
        "limits": {"duration": [10, 300, 5], "steps": [4, 60, 1],
                   "cover_strength": [0.0, 1.0, 0.05],
                   "noise_strength": [0.0, 1.0, 0.05],
                   "preview_seconds": PREVIEW_SECONDS,
                   "upload_mb": UPLOAD_MB},
        # The initial list, so Create can build its dropdown without a second
        # round trip. /api/loras rescans when the user asks for a refresh.
        "auto_duration": {
            "applies_to": ["acestep"], "requires": "thinking",
            "note": "The model sizes the song to the lyrics and caption. "
                    "Measured 13 s for a two-line sketch, 75 s for a "
                    "verse/chorus song, 192 s for a six-section epic. Needs "
                    "thinking mode: without the LM there is nothing to make "
                    "the decision and it is a flat 120 s.",
        },
        "thinking": {
            "default": False, "applies_to": ["acestep"],
            "model": "acestep-5Hz-lm-1.7B",
            "temperature": [0.0, 2.0, 0.05],
            "note": "ACE-Step's 5Hz language model reasons over your prompt "
                    "before any audio is made — it picks bpm, key and time "
                    "signature, and can rewrite the caption. Costs about 7 GB "
                    "of system RAM (not VRAM) and ~20 s the first time. "
                    "Temperature is the weirdness dial and only applies here.",
        },
        "takes": {"default": 1, "max": max_takes(), "applies_to": ["acestep"],
                  "note": "Each take is a different song from the same prompt "
                          "(ACE-Step gives every take after the first its own "
                          "seed), not a copy. The ceiling comes from the card, "
                          "so a bigger GPU raises it on its own."},
        "variants": {"active": ACESTEP_VARIANT, "items": acestep_variants(),
                     "applies_to": ["acestep"],
                     "note": "The DiT checkpoint. Switching restarts the "
                             "worker (~40 s) because the model is chosen at "
                             "load. A LoRA only matches the variant it was "
                             "trained on."},
        "loras": {"items": lo.scan(), "applies_to": ["acestep"],
                  "drop_path": lo.ROOTS[0][1],
                  "note": "A LoRA nudges ACE-Step toward the style it was "
                          "trained on. Scale is how hard: 1.0 is as trained, "
                          "lower blends back toward the base model."},
        "writer": {"ready": wr.is_ready(), "model_id": wr.model_id(),
                   "targets": list(wr.TARGETS), "modes": list(wr.MODES),
                   "mode_info": dict(wr.MODE_INFO),
                   "note": "Takes the GPU when it is idle (~30 s for a full "
                           "lyric) and falls back to the CPU when it is not "
                           "(a minute or two). It never waits for the card, "
                           "so writing is always possible mid-render."},
        "llm_title": {"ready": llm_ready(), "model_id": llm_title.MODEL_ID,
                      "hint": f"{llm_title.MODEL_ID} on CPU, ~9 s. Off uses "
                              f"the instant heuristic."},
        "phone": phone,
        "notes": NOTES,
        "quality": quality_catalogue(),
        # Features that are meaningless unless ANOTHER feature is on:
        # automatic duration has nothing to make the decision without the LM.
        "requires": {"auto_duration": "thinking"},
    }
    cfg["capabilities"] = _capabilities(cfg)
    return cfg


@app.get("/api/health")
def get_health():
    return {"ok": True, "uptime_s": round(time.time() - START, 1),
            "jobs_active": len(REG.list(active_only=True))}


@app.get("/api/vram")
def get_vram():
    return vram_snapshot()


@app.get("/api/budget")
def get_budget(model: str = "minimax", duration: float = 30.0,
               llm: str = "", rvq: str = "", kv: str = ""):
    """Does this song fit, AT THE SETTINGS THE USER PICKED.

    llm/rvq/kv are optional; omitted, they fall back to what the worker does
    unasked (4-bit LLM, bf16 RVQ, 16-bit cache) -- i.e. exactly the answer this
    endpoint gave before the quality panel existed. A wrong warning is worse
    than none, so an unrecognised value is a 400 rather than a silent default.
    """
    llm = quality_field("llm", llm, QUALITY_DEFAULTS["llm"])
    rvq = quality_field("rvq", rvq, QUALITY_DEFAULTS["rvq"])
    kv = quality_field("kv", kv, QUALITY_DEFAULTS["kv"])
    if model != "minimax":
        return {"applies": False, "reason": "model"}
    used, total = gpu_memory()
    if used is None:
        return {"applies": False, "reason": "gpu_unavailable"}
    free = total - used
    need, fits, head, base = budget_for(duration, free, llm, rvq, kv)
    return {"applies": True, "need_gb": round(need, 2),
            "free_gb": round(free, 2), "headroom_gb": round(head, 2),
            "fits": bool(fits),
            # Echoed so a fit-note can say WHICH settings it is describing --
            # the same numbers with a different floor are a different answer.
            "llm": llm, "rvq": rvq, "kv": kv,
            "base_gb": round(base, 2), "kv_bits": KV_BITS[kv]}


@app.get("/api/gpu/state")
def get_gpu_state():
    """What is loaded RIGHT NOW, and at which precisions.

    `quality` is the worker's own report of the weights it holds, so the UI can
    say whether the next generation pays a reload -- not what was last asked
    for. Null when nothing is loaded, and for ACE-Step, which has no such
    settings.
    """
    loaded = SUP.current()
    q = SUP.loaded_quality() if loaded else None
    return {"loaded": loaded,
            "loaded_label": BACKENDS[loaded]["label"] if loaded else None,
            "quality": q,
            "applies_to": ["minimax"],
            "restart_note": QUALITY_RESTART_NOTE}


@app.post("/api/gpu/unload")
def post_unload(body: dict = Body(default=None)):
    """Free the GPU. `{"force": true}` is the get-out-of-jail path.

    Without force this refuses while anything is on the GPU queue, which is
    right for the normal case -- but it also means that a worker wedged in
    `running` can never be cleared, and the user is stuck with no way to
    recover short of killing the server. That is exactly what happened.

    The forced path deliberately does NOT call `SUP.unload()` up front: unload
    takes the same RLock `SUP.generate()` holds for the whole generation, so on
    a truly stuck job it would block for minutes -- the exact wedge it is meant
    to rescue. Killing the executing process is what actually returns the VRAM,
    and it takes no lock; unload() runs afterwards, once the lock is free.
    """
    force = bool((body or {}).get("force"))

    if not force:
        # Holding the registry lock across the check closes the race with the
        # drain thread promoting a queued job to `starting` moments later.
        with REG.lock:
            running = REG.active_gpu()
            if running is not None and running.state in ("starting", "running"):
                raise ApiError(409, "gpu_busy", "A job is running.",
                               {"job_id": running.id, "kind": running.kind,
                                "can_force": True})
            if not REG.gpu_idle():
                raise ApiError(409, "gpu_busy", "Jobs are queued on the GPU.",
                               {"can_force": True})
            SUP.unload()
        return {"forced": False, "unloaded": True, "killing": None,
                "cancelled": [], "vram": vram_snapshot()}

    # -- forced ------------------------------------------------------------
    # Drop the queue first so nothing is promoted into the slot being cleared.
    # No kind filter: every kind rides the one GPU lane, and REG.cancel()
    # already no-ops on anything that is not still `queued`.
    cancelled = [jb.id for jb in REG.list(active_only=True) if REG.cancel(jb)]

    running = REG.active_gpu()
    if running is None:
        # The lock is free, so unload directly. This is also the path that
        # clears a worker left resident after a job that already ended -- the
        # case where the GPU is full but the registry looks idle.
        try:
            SUP.unload()
        except Exception as exc:                                # noqa: BLE001
            raise ApiError(500, "internal", f"Unload failed: {exc}") from exc
        return {"forced": True, "unloaded": True, "killing": None,
                "cancelled": cancelled, "vram": vram_snapshot()}

    # cancel_requested is what makes the job record as `cancelled` rather than
    # `error` once its process dies.
    running.cancel_requested = True
    threading.Thread(target=_force_kill, args=(running,), daemon=True).start()

    # Deliberately NOT reporting the job as cancelled or the VRAM as freed:
    # neither has happened yet. The kill is asynchronous -- the client watches
    # the job's SSE `cancelled` event, or polls /api/vram.
    return {"forced": True, "unloaded": False, "killing": running.id,
            "killing_kind": running.kind, "cancelled": cancelled,
            "vram": vram_snapshot()}


# ============================================================== 5. generate

def _f(body, key, default=None):
    v = body.get(key, default)
    return default if v is None else v


_name_lock = threading.Lock()


def free_out_name(model):
    """A `<stamp>_<model>.wav` nobody else is going to write.

    The stamp has ONE-SECOND resolution and the GPU queue is four deep, so two
    Generate taps inside the same second -- a phone double-tap, or Reuse
    followed straight away by Generate -- were handed the SAME name. Both
    workers then wrote that one file: the second overwrote the first's audio
    AND its sidecar, and the first job's `result.path` played the second job's
    song. The old app never hit this because it named the file when the
    serialized job RAN; naming at POST time is what reopened it.

    Called under `_name_lock`, which `post_generate` holds until the job is
    registered -- the name is only visibly reserved once the job is on the
    queue, so the lock is what closes the window between the two.
    """
    taken = {j.request.get("out_name") for j in REG.list(active_only=True)
             if j.kind == "generate"}
    base = f"{datetime.now():%Y%m%d-%H%M%S}_{model}"
    name, n = f"{base}.wav", 1
    while name in taken or os.path.exists(os.path.join(OUTDIR, name)):
        name, n = f"{base}-{n}.wav", n + 1
    return name


@app.post("/api/generate", status_code=202)
def post_generate(body: dict = Body(default=None)):
    b = body or {}
    model = str(_f(b, "model", "minimax"))
    prompt = str(_f(b, "prompt", "") or "")
    lyrics = str(_f(b, "lyrics", "") or "")
    instrumental = bool(_f(b, "instrumental", False))
    post_kind = str(_f(b, "post_kind", "none") or "none")
    preview = bool(_f(b, "preview", False))
    src_path = _f(b, "src_path") or None

    # First, and before the name lock and the enqueue: a typo'd preset must be
    # a 400 from this POST, never a 202 that dies on the GPU lane. Checked even
    # for ACE-Step, which cannot use these settings -- a request that says
    # something impossible should hear about it either way.
    quality = resolve_quality(b.get("quality"))

    if model not in BACKENDS:
        raise bad("Unknown model.")
    if not prompt.strip():
        raise bad("A style description is required.")
    if model == "minimax" and not lyrics.strip():
        raise bad("MiniMax Music 3 requires lyrics — it has no instrumental "
                  "mode. Use ACE-Step for instrumentals.")
    if instrumental and model != "acestep":
        raise bad("Instrumental is ACE-Step only.")
    if post_kind not in pp.UPSCALERS:
        raise bad("Unknown upscaler.")
    # num_arg, not int()/float(): JSON `true` is not the number 1, and letting
    # it through would put a bool on the GPU lane as a seed.
    duration = num_arg(_f(b, "duration", 30), 30.0)
    steps = num_arg(_f(b, "steps", 30), 30, int)
    seed = num_arg(_f(b, "seed", 7), 7, int)
    # -1 is the "let the model decide" signal, not a length. ACE-Step's own
    # docstring: "If <0 or None, model chooses automatically." It only means
    # anything with thinking on -- the length comes from the LM's CoT metadata
    # (`params.cot_duration`) and there is nothing to supply it otherwise.
    auto_duration = duration == -1
    if auto_duration and model != "acestep":
        raise bad("Automatic duration is an ACE-Step feature.",
                  {"field": "duration", "model": model})
    if not auto_duration and (not (10 <= duration <= 300)):
        raise bad("Out of range.")
    if not (4 <= steps <= 60):
        raise bad("Out of range.")
    if src_path:
        # Read the capability, do not name a model: the moment a second
        # backend can cover, a hard-coded name here is the thing that says no.
        if not PROMPT_META.get(model, {}).get("supports_cover"):
            raise bad(f"{BACKENDS.get(model, {}).get('label', model)} cannot "
                      f"cover an existing track.",
                      {"field": "src_path", "model": model})
        if not os.path.isfile(str(src_path)):
            raise bad("No such file.")

    if model != "minimax":
        quality = None      # ACE-Step has none of these knobs

    cover_strength = num_arg(_f(b, "cover_strength", 1.0), 1.0)
    noise_strength = num_arg(_f(b, "noise_strength", 0.75), 0.75)

    # -- LoRA ---------------------------------------------------------------
    # Resolved here, at POST time, so an id that no longer exists is a 400 and
    # not a job that dies four minutes in. The adapter is ACE-Step's DiT, so
    # asking for one on MiniMax is a mistake worth naming rather than ignoring.
    lora_id = str(_f(b, "lora", "") or "").strip()
    lora = None
    if lora_id:
        if model != "acestep":
            raise bad("LoRAs apply to ACE-Step only — they are adapters on its "
                      "DiT. MiniMax has no LoRA support here.",
                      {"field": "lora", "model": model})
        lora = lo.by_id(lora_id)
        if lora is None:
            raise bad(f"No LoRA {lora_id!r}. It may have been moved or deleted.",
                      {"field": "lora"})
    lora_scale = num_arg(_f(b, "lora_scale", 1.0), 1.0)
    lora_scale = max(0.0, min(2.0, lora_scale))

    # Takes. 1 by default -- ACE-Step's own default of 2 quietly rendered a
    # second song per request and stranded it -- but the knob is real, and the
    # ceiling comes from the card rather than from a constant, so a bigger GPU
    # simply allows more without an edit here.
    takes = int(num_arg(_f(b, "takes", 1), 1))
    if takes != 1 and model != "acestep":
        raise bad("Multiple takes are an ACE-Step feature.",
                  {"field": "takes", "model": model})
    takes = max(1, min(takes, max_takes()))

    # Thinking: ACE-Step's 5Hz LM reasons over the prompt before the DiT runs.
    # Opt-in, because it costs ~7 GB of system RAM and ~20 s on first use, and
    # because it CHANGES the result -- it may rewrite the caption and invent
    # bpm/key/time-signature. lm_temperature only means anything with it on.
    # Which DiT checkpoint. Load-time, like quantization: the supervisor
    # restarts the worker when it differs from what is running.
    variant = str(_f(b, "variant", "") or "").strip()
    if variant:
        if model != "acestep":
            raise bad("Model variants are an ACE-Step setting.",
                      {"field": "variant", "model": model})
        known = {v["name"] for v in acestep_variants()}
        if variant not in known:
            raise bad(f"No checkpoint {variant!r} installed.",
                      {"field": "variant", "installed": sorted(known)})
    else:
        variant = ACESTEP_VARIANT if model == "acestep" else None

    # The workspace a song is BORN into. Resolved here, from the server's
    # registry, rather than taken from the client: two browsers with different
    # localStorage would otherwise tag the same project inconsistently. A
    # request may override it explicitly; otherwise it is whatever is active.
    workspace = _f(b, "workspace", None)
    workspace = (wsp.clean(workspace) or None) if workspace else wsp.active()

    thinking = bool(_f(b, "thinking", False))
    if auto_duration and not thinking:
        # Measured: duration=-1 with no LM is a flat 120 s every time, not a
        # decision. Refusing beats shipping a control that means two different
        # things depending on a checkbox elsewhere in the form.
        raise bad("Automatic duration needs thinking mode — the length comes "
                  "from the 5Hz LM. Without it, -1 is a fixed 120 s.",
                  {"field": "duration", "needs": "thinking"})
    if thinking and model != "acestep":
        raise bad("Thinking mode is ACE-Step's 5Hz LM — MiniMax has no "
                  "equivalent here.", {"field": "thinking", "model": model})
    lm_temperature = num_arg(_f(b, "lm_temperature", 1.0), 1.0)
    lm_temperature = max(0.0, min(2.0, lm_temperature))

    if preview:
        duration = min(duration, PREVIEW_SECONDS)
        post_kind = "none"

    # Held until the job is on the queue: the name is not reserved against a
    # second POST until that POST can see the job carrying it.
    with _name_lock:
        out_name = free_out_name(model)
        params = dict(prompt=prompt.strip(),
                      lyrics="" if instrumental else lyrics.strip(),
                      duration=float(duration), steps=int(steps),
                      seed=int(seed), instrumental=bool(instrumental),
                      out_name=out_name)
        if model == "acestep" and src_path:
            # The worker's key is `src_audio`, not `src_path`. Do not rename it.
            params.update(src_audio=str(src_path),
                          cover_strength=cover_strength,
                          noise_strength=noise_strength)
        if quality:
            # Per-request settings ride with the job; the two load-time ones
            # (llm, rvq) are handled by _run_generate, which may have to
            # restart the worker to honour them.
            params.update(kv_cache=quality["kv"], reserve=quality["reserve"])
        if takes > 1:
            params.update(takes=takes)
        if thinking:
            params.update(thinking=True, lm_temperature=lm_temperature)
        if lora:
            # An explicit adapter name, because ACE-Step would otherwise name
            # it after its directory and every `.../final` would collide.
            params.update(lora_path=lora["path"],
                          lora_name=lo.adapter_name(lora["id"]),
                          lora_scale=lora_scale)

        req = {"model": model, "prompt": prompt, "lyrics": lyrics,
               "duration": duration, "steps": steps, "seed": seed,
               "instrumental": instrumental, "post_kind": post_kind,
               "preview": preview, "src_path": src_path,
               "cover_strength": cover_strength,
               "noise_strength": noise_strength, "out_name": out_name,
               "quality": quality,
               "takes": takes,
               "variant": variant,
               "workspace": workspace,
               "thinking": thinking,
               "lm_temperature": lm_temperature if thinking else None,
               "lora": lora["id"] if lora else None,
               "lora_name": lora["name"] if lora else None,
               "lora_scale": lora_scale if lora else None}
        job = J.Job("generate", req)
        job.extra["params"] = params
        _enqueue(job, _run_generate)
    return {"job_id": job.id, "kind": "generate", "state": job.state,
            "stream_url": f"/api/jobs/{job.id}/events", "out_name": out_name,
            "quality": quality}


def _enqueue(job, fn):
    try:
        REG.submit_gpu(job, fn)
    except J.GpuBusy as e:
        raise ApiError(409, "gpu_busy", str(e))


def _run_generate(job):
    r = job.request
    params = job.extra["params"]
    model = r["model"]
    if job.cancel_requested:
        job.cancel()
        return

    # Quantization happens as the weights load, so llm/rvq are not something a
    # request can carry -- they are a property of the process. `quality=` lets
    # the supervisor compare them against the live worker and restart it when
    # they differ; it emits the "reloading … ~20 s" line itself, at the moment
    # it actually happens, so the wait is never a mystery.
    q = r.get("quality")
    env = worker_quality = None
    if model == "acestep" and r.get("variant"):
        # The worker reads ACESTEP_VARIANT and passes it as config_path.
        env = {"ACESTEP_VARIANT": r["variant"]}
    if q:
        env = {"QUANT_LLM": q["llm"], "QUANT_RVQ": q["rvq"],
               "OFFLOAD_RESERVE": q["reserve"]}
        worker_quality = {"llm": q["llm"], "rvq": q["rvq"]}
        job.line(f"quality {q['preset']}: LLM {q['llm']} · RVQ {q['rvq']} · "
                 f"KV cache {q['kv']} · reserve {q['reserve']}")
    res = SUP.generate(model, params, on_progress=job.emit, env=env,
                       quality=worker_quality)
    if job.cancel_requested:
        job.cancel()
        return
    if not res or not res.get("path"):
        job.fail("worker_died", "The worker returned no result.")
        return

    path = res["path"]
    job.artifact("audio", "generated", path, media_url(path))

    # Every take gets the same sidecar, so a second take is a first-class
    # library member rather than the orphan ACE-Step's batch_size=2 used to
    # strand. `path` remains the primary; the siblings differ only by seed.
    takes = [p for p in (res.get("takes") or []) if p and p != path]

    # The sidecar is written for the GENERATED file, before any post step: the
    # upscaled derivative lives in upscaled\ and is reachable via post_kind.
    lib.write(path, model=model, prompt=r["prompt"],
              lyrics=None if r["instrumental"] else (r["lyrics"] or None),
              instrumental=bool(r["instrumental"]),
              duration=float(r["duration"]), steps=int(r["steps"]),
              seed=int(r["seed"]), post_kind=r["post_kind"],
              preview=bool(r["preview"]),
              cover_src=r["src_path"] if (model == "acestep" and r["src_path"])
              else None,
              cover_strength=r["cover_strength"] if r["src_path"] else None,
              noise_strength=r["noise_strength"] if r["src_path"] else None,
              seconds=res.get("seconds"), elapsed=res.get("elapsed"),
              sampling_rate=res.get("sampling_rate"),
              vram_peak=res.get("vram_peak"),
              # The settings that made this song travel with it, like the
              # prompt does. None for a request that named none.
              quality=q,
              # Read back from the worker (what was actually in the forward
              # pass), not from the request (what was asked for).
              lora=r.get("lora"), lora_name=r.get("lora_name"),
              lora_scale=res.get("lora_scale") if res.get("lora_path")
              else None,
              thinking=bool(res.get("thinking")) or None,
              lm_temperature=r.get("lm_temperature"),
              workspace=r.get("workspace"), variant=r.get("variant"))
    for extra in takes:
        try:
            lib.write(extra, model=model, prompt=r["prompt"],
                      lyrics=None if r["instrumental"] else (r["lyrics"] or None),
                      instrumental=bool(r["instrumental"]),
                      duration=float(r["duration"]), steps=int(r["steps"]),
                      seed=int(r["seed"]), post_kind="none",
                      preview=bool(r["preview"]), quality=q,
                      lora=r.get("lora"), lora_name=r.get("lora_name"),
                      workspace=r.get("workspace"),
                      sibling_of=os.path.basename(path))
            art.ensure_cover(extra)
        except Exception:
            pass

    try:
        art.ensure_cover(path)
    except Exception:
        pass
    cover = art.cover_path(path)
    if os.path.exists(cover):
        job.artifact("image", "cover", cover, media_url(cover))

    final = path
    post = None
    kind = r["post_kind"]
    if kind and kind != "none":
        job.phase("post", f"post-processing — {pp.UPSCALERS[kind]}")
        before = pp.bandwidth(final)
        job.line(before)
        post = {"kind": kind, "ok": False, "bandwidth_before": before,
                "bandwidth_after": None}
        try:
            out = pp.upscale(kind, final, UPDIR, on_progress=job.emit,
                             auto_lowpass=True)
            job.artifact("audio", "upscaled", out, media_url(out))
            after = pp.bandwidth(out)
            job.line(after)
            post.update(ok=True, bandwidth_after=after)
            final = out
        except Exception as e:
            # A bad post step degrades to "you still have your generation".
            job.line(f"post FAILED: {type(e).__name__}: {e}")

    seconds = res.get("seconds") or 0.0
    elapsed = res.get("elapsed") or 0.0
    job.succeed({
        "path": path, "url": media_url(path),
        "library_id": os.path.basename(path),
        "final_path": final, "final_url": media_url(final),
        "seconds": res.get("seconds"), "elapsed": res.get("elapsed"),
        "sampling_rate": res.get("sampling_rate"),
        "realtime_ratio": round(elapsed / max(seconds, 1e-6), 2),
        "vram_peak": res.get("vram_peak"), "vram_idle": res.get("vram_idle"),
        "preview": bool(r["preview"]),
        "quality": q,
        # Which adapter actually ran, so the result card can say so without
        # re-reading the sidecar. `lora_scale` comes off the worker's reply,
        # not the request, for the same reason the sidecar does.
        "lora": r.get("lora"), "lora_name": r.get("lora_name"),
        "lora_scale": res.get("lora_scale") if res.get("lora_path") else None,
        "takes": [{"path": p, "url": media_url(p)} for p in takes],
        # Read back from the worker: what the LM actually did, not what was
        # asked for. Same rule as the LoRA fields directly above.
        "workspace": r.get("workspace"),
        "variant": r.get("variant"),
        "thinking": bool(res.get("thinking")),
        "lm_temperature": r.get("lm_temperature"),
        "post": post,
    })


# =============================================================== 6. restore

_inspect_locks = {}
_inspect_guard = threading.Lock()


def _inspect_lock(path):
    key = os.path.normcase(path)
    with _inspect_guard:
        lk = _inspect_locks.get(key)
        if lk is None:
            lk = _inspect_locks[key] = threading.Lock()
        return lk


def _inspect(path, region, marker=None):
    # Formatted BEFORE the try below: a malformed region is a 400 from the
    # caller, not a "(spectrogram failed: …)" note wrapped in a 200.
    rspec = region_spec(region)
    txt = pp.bandwidth(path)
    out = {"bandwidth": txt, "image_url": None, "geometry": None,
           "note": "Click a corner of the region you want to fill.",
           "truncated_note": None}
    mk = None
    if isinstance(marker, dict):
        try:
            mk = f"{float(marker['t']):.4f}:{float(marker['f']):.1f}"
        except (KeyError, TypeError, ValueError):
            mk = None
    with _inspect_lock(path):
        try:
            g = pp.specview(path, spec_png(), region=rspec, marker=mk)
        except Exception as e:
            # A broken render must not hide the bandwidth verdict.
            out["bandwidth"] = txt + f"\n\n(spectrogram failed: {e})"
            return out
    out["geometry"] = g
    out["image_url"] = media_url(g.get("png"))
    if g.get("truncated"):
        out["truncated_note"] = (f"showing the first {g['t1']:.0f}s of "
                                 f"{g['full_duration']:.0f}s")
    return out


@app.post("/api/restore/inspect")
def post_inspect(body: dict = Body(default=None)):
    b = body or {}
    path = path_arg(b.get("path"))
    if not path:
        raise bad("Load a file first.")
    if not os.path.isfile(path):
        raise bad("No such file.")
    return _inspect(path, b.get("region"), b.get("marker"))


@app.post("/api/restore/cliff")
def post_cliff(body: dict = Body(default=None)):
    b = body or {}
    path = path_arg(b.get("path"))
    if not path:
        raise bad("Pick a file first.")
    if not os.path.isfile(path):
        raise bad("No such file.")
    hz, drop = pp.cliff_of(path)
    if hz is None:
        raise ApiError(422, "unmeasurable",
                       "Could not measure a cliff for that file.")
    weak = bool(drop is not None and drop < 8)
    msg = None
    if weak:
        msg = (f"Cliff {hz:.0f} Hz but only {drop:.1f} dB — that is not a real "
               f"codec cliff, so there may be nothing to restore.")
    out = {"cliff_hz": hz, "drop_db": drop, "weak": weak, "message": msg,
           "inspect": None}
    if b.get("render"):
        raw = b.get("region")
        region = dict(raw) if isinstance(raw, dict) else {}
        region["flo"] = hz
        out["inspect"] = _inspect(path, region, b.get("marker"))
    return out


@app.post("/api/upscale", status_code=202)
def post_upscale(body: dict = Body(default=None)):
    b = body or {}
    path = path_arg(b.get("path"))
    kind = str(b.get("kind") or "")
    if not path:
        raise bad("Pick a file to upscale.")
    if not os.path.isfile(path):
        raise bad("No such file.")
    if kind == "none" or kind not in pp.UPSCALERS:
        raise bad("Unknown upscaler.")
    region = b.get("region") or {"t0": 0, "t1": 0, "flo": 16000, "fhi": 0}
    region_spec(region)         # 400 here rather than a job that dies later
    # Same rule for the two new stages: reject the setting now, not four
    # minutes into a job.
    degrade = str(b.get("degrade") or "off")
    try:
        pp.degrade_plan(degrade, kind)
    except ValueError:
        raise bad(f"Unknown degrade mode {degrade!r}.",
                  {"field": "degrade", "allowed": list(pp.DEGRADES)})
    naturalize = str(b.get("naturalize") or "off")
    if naturalize not in pp.NATURALIZERS:
        raise bad(f"Unknown naturalize strength {naturalize!r}.",
                  {"field": "naturalize", "allowed": list(pp.NATURALIZERS)})
    req = {"path": path, "kind": kind,
           "auto_lowpass": bool(b.get("auto_lowpass", True)),
           "fill": bool(b.get("fill", True)),
           "region": region,
           "compare": bool(b.get("compare", True)),
           "degrade": degrade, "naturalize": naturalize,
           "lufs": float(b.get("lufs", -14.0))}
    job = J.Job("upscale", req)
    _enqueue(job, _run_upscale)
    return {"job_id": job.id, "kind": "upscale", "state": job.state,
            "stream_url": f"/api/jobs/{job.id}/events"}


def _keep_proc(job):
    """Register the running child so a cancel can kill it (REQUIREMENTS §8)."""
    return lambda p: job.extra.__setitem__("proc", p)


def _run_upscale(job):
    r = job.request
    src, kind = r["path"], r["kind"]
    spec = region_spec(r["region"])
    job.line(f"--- {pp.UPSCALERS[kind]} ---")
    before = pp.bandwidth(src)
    job.line(before)

    # -- optional: damage it first ---------------------------------------
    # `src` stays the pristine original throughout: it is what the compare
    # figure and the hybrid splice measure against. Only the upscaler input
    # changes.
    feed, degraded, bw_degraded = src, None, None
    if r.get("degrade", "off") != "off":
        _, _, label = pp.degrade_plan(r["degrade"], kind)
        job.phase("degrade", pp.DEGRADES.get(label, label))
        degraded = os.path.join(
            UPDIR, os.path.splitext(os.path.basename(src))[0]
            + f"_deg_{label}.wav")
        try:
            feed = pp.degrade(src, degraded, r["degrade"], kind=kind,
                              on_progress=job.emit, on_proc=_keep_proc(job))
        except Exception as e:
            if job.cancel_requested:
                job.cancel()
                return
            job.fail("degrade_failed", f"{type(e).__name__}: {e}")
            return
        job.artifact("audio", "degraded", degraded, media_url(degraded))
        bw_degraded = pp.bandwidth(degraded)
        job.line(bw_degraded)

    job.phase("upscale", pp.UPSCALERS[kind])
    try:
        out = pp.upscale(kind, feed, UPDIR, on_progress=job.emit,
                         auto_lowpass=bool(r["auto_lowpass"]),
                         on_proc=_keep_proc(job))
    except Exception as e:
        # A killed child raises here first, so the drain loop never gets to
        # convert it: a cancellation would otherwise be recorded as a failure.
        if job.cancel_requested:
            job.cancel()
            return
        job.fail("upscale_failed", f"{type(e).__name__}: {e}")
        return
    job.artifact("audio", "upscaled", out, media_url(out))
    after = pp.bandwidth(out)
    job.line(after)

    upscaled, hybrid, spliced = out, None, False
    if r["fill"]:
        job.phase("splice", "splice into selected region")
        hybrid = os.path.join(
            UPDIR, os.path.splitext(os.path.basename(out))[0] + "_hybrid.wav")
        try:
            # `src`, never `feed`: splicing the restored band back into the
            # DEGRADED file would bake the damage into the result. The whole
            # point of the hybrid is that everything outside the box is
            # bit-identical to the original.
            pp.region_fill(src, out, hybrid, [spec], on_progress=job.emit,
                           on_proc=_keep_proc(job))
            out, spliced = hybrid, True
            job.artifact("audio", "hybrid", hybrid, media_url(hybrid))
        except Exception as e:
            if job.cancel_requested:
                job.cancel()
                return
            # The upscaled file is still a result.
            job.line(f"splice FAILED: {type(e).__name__}: {e}")
            hybrid = None

    # -- optional: the analogue pass -------------------------------------
    natural, pre_natural = None, out
    if r.get("naturalize", "off") != "off":
        job.phase("naturalize", pp.NATURALIZERS[r["naturalize"]])
        natural = os.path.join(
            UPDIR, os.path.splitext(os.path.basename(out))[0] + "_nat.wav")
        try:
            pp.naturalize(out, natural, strength=r["naturalize"],
                          lufs=r.get("lufs", -14.0), on_progress=job.emit,
                          on_proc=_keep_proc(job))
            job.artifact("audio", "naturalized", natural, media_url(natural))
            out = natural
        except Exception as e:
            if job.cancel_requested:
                job.cancel()
                return
            # The upscaled/hybrid file is still a result.
            job.line(f"naturalize FAILED: {type(e).__name__}: {e}")
            natural = None

    compare_png = None
    if r["compare"]:
        job.phase("compare", "before / after")
        panels = [("1. INPUT", src)]
        if degraded:
            panels.append((f"2. DEGRADED ({r['degrade']})", degraded))
        panels.append((f"{len(panels) + 1}. {kind.upper()} (whole file)",
                       upscaled))
        if spliced:
            panels.append((f"{len(panels) + 1}. HYBRID (spliced)", hybrid))
        try:
            compare_png = pp.spectrogram(
                panels, spec_png(), regions=[spec], title="Before / after",
                subtitle="Box = the selected region. On the hybrid, everything "
                         "outside it is bit-identical to the input.")
            job.artifact("image", "compare", compare_png,
                         media_url(compare_png))
        except Exception as e:
            job.line(f"spectrogram failed: {e}")
            compare_png = None

    job.succeed({"path": out, "url": media_url(out), "kind": kind,
                 "spliced": spliced,
                 "upscaled_path": upscaled, "upscaled_url": media_url(upscaled),
                 "compare_url": media_url(compare_png) if compare_png else None,
                 "bandwidth_before": before, "bandwidth_after": after,
                 "region": spec,
                 "degrade": r.get("degrade", "off"),
                 "degraded_path": degraded,
                 "degraded_url": media_url(degraded) if degraded else None,
                 "bandwidth_degraded": bw_degraded,
                 "naturalize": r.get("naturalize", "off"),
                 "natural_path": natural,
                 "natural_url": media_url(natural) if natural else None,
                 # The A/B partner for the analogue pass: whatever the chain
                 # produced just before it. Without this the un-naturalized
                 # version is unreachable once `out` has moved on.
                 "pre_natural_path": pre_natural if natural else None,
                 "pre_natural_url": media_url(pre_natural) if natural else None})


# ================================================================ 7. lyrics

@app.post("/api/lyrics/probe")
def post_probe(body: dict = Body(default=None)):
    b = body or {}
    path = path_arg(b.get("path"))
    if not path:
        return {"ok": False, "found": False, "text": None,
                "message": "Pick a source track to search for its lyrics."}
    uploaded = bool(b.get("uploaded")) or _under(path, UPLOADDIR)
    res = ly.probe(path)
    msg = ly.describe(res, uploaded)
    found = bool(res.get("found"))
    return {"ok": bool(res.get("ok")), "found": found,
            "text": (res.get("best") or {}).get("text") if found else None,
            "message": msg,
            "best": res.get("best"),
            "embedded": res.get("embedded", []),
            "sidecars": res.get("sidecars", [])}


# The DiT variant this studio runs. ACE-Step resolves `config_path=None` to
# "acestep-v15-turbo", and the worker passes None -- so this is that choice
# written down where the rest of the code can compare against it.
ACESTEP_ROOT = os.path.join(ROOT, "acestep")
ACESTEP_VARIANT = os.environ.get("ACESTEP_VARIANT", "acestep-v15-turbo")

# Every DiT checkpoint present, with what it can do. Discovered from disk, so
# downloading one is all it takes to offer it. `complete` is the interesting
# difference: turbo cannot continue a track, base can.
# acestep/constants.py is the source of truth here, not this file's memory:
#   TASK_TYPES_TURBO = text2music, repaint, cover, cover-nofsq
#   TASK_TYPES_BASE  = ... plus extract, lego, complete
# So COVER works on turbo -- it is `complete` (continue a track), `extract`
# (stems) and `lego` (layers) that are base-only, along with real CFG.
_TASKS_TURBO = ["text2music", "repaint", "cover", "cover-nofsq"]
_TASKS_BASE = _TASKS_TURBO + ["extract", "lego", "complete"]

ACESTEP_VARIANTS = {
    "acestep-v15-turbo": dict(
        tasks=_TASKS_TURBO, label="Turbo (2B) - fast", steps=8, needs_gb=7.0,
        note="8 steps, no CFG. What this box has been shipping."),
    "acestep-v15-base": dict(
        tasks=_TASKS_BASE, label="Base (2B) - continue, extract, guidance", steps=32, needs_gb=7.3,
        note="CFG and 32-100 steps, so several times slower than turbo - and "
             "the only 2B variant that can CONTINUE a track (`complete`), "
             "pull stems (`extract`) and honour guidance_scale."),
    "acestep-v15-sft": dict(
        tasks=_TASKS_BASE, label="SFT (2B)", steps=32, needs_gb=7.3, note="Base, further tuned."),
    "acestep-v15-turbo-continuous": dict(
        tasks=_TASKS_TURBO, label="Turbo continuous (2B)", steps=8, needs_gb=7.0,
        note="NOT song continuation, despite the name. It reports "
             "is_turbo=True and its config is identical to plain turbo bar a "
             "missing model_version key; every use of 'continuous' in "
             "ACE-Step's code means continuous (non-quantized) latents, which "
             "is also what `cover-nofsq` refers to. Same speed as turbo, "
             "different training. Use base if you want to CONTINUE a track."),
    "acestep-v15-xl-turbo": dict(
        tasks=_TASKS_TURBO, label="XL Turbo (4B)", steps=8, needs_gb=11.5,
        note="4B decoder. Needs ~12 GB - a 16 GB card or better."),
    "acestep-v15-xl-base": dict(
        tasks=_TASKS_BASE, label="XL Base (4B)", steps=32, needs_gb=12.0,
        note="4B with CFG. Needs ~12 GB and is the slowest of the set."),
    "acestep-v15-xl-sft": dict(
        tasks=_TASKS_BASE, label="XL SFT (4B)", steps=32, needs_gb=12.0, note="XL, further tuned."),
}


def acestep_variants():
    """Installed DiT checkpoints, newest facts first. Reads the folder."""
    root = os.path.join(ACESTEP_ROOT, "checkpoints")
    out = []
    try:
        present = sorted(d for d in os.listdir(root)
                         if d.startswith("acestep-v15-")
                         and os.path.isdir(os.path.join(root, d)))
    except OSError:
        present = []
    _used, total = gpu_memory()
    for name in present:
        meta = dict(ACESTEP_VARIANTS.get(name) or
                    {"label": name, "note": "Unknown variant.",
                     "needs_gb": None, "steps": 8, "tasks": list(_TASKS_TURBO)})
        need = meta.get("needs_gb")
        tasks = list(meta.get("tasks") or _TASKS_TURBO)
        meta.update(name=name, installed=True, tasks=tasks,
                    # What the CHECKPOINT adds on top of what the backend can
                    # do. A control gated on one of these has to consult the
                    # active variant, not just the model.
                    supports={"cover": "cover" in tasks,
                              "repaint": "repaint" in tasks,
                              "complete": "complete" in tasks,
                              "extract": "extract" in tasks,
                              "lego": "lego" in tasks,
                              "guidance": "complete" in tasks},
                    fits=(None if (not need or not total) else need <= total),
                    active=(name == ACESTEP_VARIANT))
        out.append(meta)
    return out


def max_takes():
    """How many ACE-Step takes this card can be asked for.

    Derived, not constant. The 10 GB card is the floor this was developed on,
    and every limit tied to it should read the hardware instead of hardcoding
    the compromise -- otherwise a bigger GPU inherits a small card's ceiling
    and nobody notices. ACE-Step's own `_vram_guard_reduce_batch` will still
    reduce the batch if it disagrees, so this is a UI ceiling, not a promise.
    """
    _used, total = gpu_memory()
    if not total:
        return 2
    if total < 11:
        return 2          # 10 GB: a second take fits, a third is optimistic
    if total < 20:
        return 4
    return 8


@app.get("/api/loras")
def get_loras():
    """Every adapter on disk, rescanned per call.

    Deliberately not cached: the whole point of the drop folder is that you
    put a file in it and it appears. A scan is a handful of stat calls and one
    small JSON read per adapter.
    """
    lo.ensure_roots()
    items = lo.scan()
    loaded = SUP.current()
    # An adapter is rank-r weights shaped for ONE decoder. ACE-Step ships
    # 2B variants (turbo, base, sft) and 4B XL variants, and a LoRA trained on
    # turbo will load against XL and produce noise -- PEFT matches by module
    # name, not by shape provenance. "contains acestep" was the first version
    # of this check and it would call every adapter compatible with every
    # variant, which is exactly the silently-wrong-model failure.
    current = ACESTEP_VARIANT
    for e in items:
        base = (e.get("base_model_name") or "").lower()
        if not base:
            e["compatible"] = None          # unrecorded: unknown, not "fine"
            e["compat_note"] = "No base model recorded — try it and listen."
        elif base == current.lower():
            e["compatible"] = True
            e["compat_note"] = None
        else:
            e["compatible"] = False
            e["compat_note"] = (f"Trained against {e['base_model_name']}, but "
                                f"{current} is loaded. Sizes differ between "
                                f"the 2B and XL decoders; expect noise.")
    return {"loras": items,
            "variant": current,
            "roots": [{"key": k, "path": p, "exists": os.path.isdir(p)}
                      for k, p in lo.ROOTS],
            "applies_to": ["acestep"],
            "loaded_model": loaded,
            "note": "Drop an adapter folder (adapter_config.json + "
                    "adapter_model.safetensors) into the loras folder and it "
                    "appears here. Anything trained locally is picked up "
                    "automatically, per-epoch checkpoints included."}


@app.post("/api/write", status_code=202)
def post_write(body: dict = Body(default=None)):
    """Ask the local LLM for lyrics or a style prompt.

    A job, not a plain POST: this is 30-90 s on CPU, and a request that long
    dies with the phone's screen. Going through the lane machinery means it
    streams, survives a reconnect, and is cancellable -- the same guarantees
    everything else has.

    It does NOT go on the GPU lane. Nothing here touches the card, and queueing
    a lyric behind a four-minute render (or worse, in front of one) would be a
    self-inflicted wound.
    """
    b = body or {}
    target = str(b.get("target") or "lyrics")
    mode = str(b.get("mode") or "generate")
    model = str(b.get("model") or "minimax")
    if target not in wr.TARGETS:
        raise bad(f"Unknown target {target!r}.",
                  {"field": "target", "allowed": list(wr.TARGETS)})
    if mode not in wr.MODES:
        raise bad(f"Unknown mode {mode!r}.",
                  {"field": "mode", "allowed": list(wr.MODES)})
    if model not in BACKENDS:
        raise bad(f"Unknown model {model!r}.",
                  {"field": "model", "allowed": list(BACKENDS)})
    if not wr.is_ready():
        raise ApiError(503, "model_missing",
                       f"{wr.model_id()} is not downloaded. Run "
                       f"`songwriter.py --fetch` once.", None)

    brief = str(b.get("brief") or "").strip()
    existing = str(b.get("existing") or "").strip()
    if mode in ("extend", "edit") and not existing:
        raise bad(f"{mode.title()} needs the current text to work from.",
                  {"field": "existing"})
    if mode == "edit" and not brief:
        raise bad("Edit needs an instruction — say what to change.",
                  {"field": "brief"})
    if mode == "generate" and not brief:
        raise bad("Say what the song is about.", {"field": "brief"})

    # Measured: 1.5B is 2-4 tok/s on this CPU and 20 tok/s on the card, so a
    # full lyric is ~3 minutes against ~25 seconds. Take the GPU when it is
    # genuinely free; never WAIT for it, because waiting behind a four-minute
    # render costs more than the CPU run it replaced. `gpu_idle()` under the
    # registry lock is what makes "free" mean free rather than "free a moment
    # ago" -- a job queued between the check and the submit would otherwise
    # find us already in the lane.
    v = vram_snapshot()
    with REG.lock:
        use_gpu = (REG.gpu_idle()
                   and float(v.get("free_gb") or 0) >= wr.VRAM_NEEDED_GB)
        seed = b.get("seed")
        req = {"target": target, "mode": mode, "model": model,
               "brief": brief[:4000], "existing": existing[:20000],
               "seed": (int(seed) if seed not in (None, "") else None),
               "temperature": float(b.get("temperature", 0.9)),
               "device": "cuda" if use_gpu else "cpu"}
        job = J.Job("write", req)
        if use_gpu:
            _enqueue(job, _run_write)          # the one GPU lane
        else:
            REG.submit_cpu(job, _run_write)    # its own thread, off the lane
    return {"job_id": job.id, "kind": "write", "state": job.state,
            "device": req["device"],
            "stream_url": f"/api/jobs/{job.id}/events"}


def _run_write(job):
    r = job.request
    dev = r.get("device", "cpu")
    job.line(f"{r['mode']} {r['target']} for {r['model']} — "
             + ("on the GPU (idle), about 30 s"
                if dev == "cuda" else "on the CPU, a minute or two"))
    job.phase("write", f"{r['mode'].title()} {r['target']}")
    try:
        res = wr.write(r["target"], r["mode"], brief=r["brief"],
                       existing=r["existing"], model=r["model"],
                       seed=r["seed"], temperature=r["temperature"],
                       device=dev,
                       on_progress=job.emit, on_proc=_keep_proc(job))
    except Exception:
        # Same as the other subprocess runners: a kill reads as cancelled.
        if job.cancel_requested:
            job.cancel()
            return
        raise
    if res.get("scoped_to"):
        job.line(f"rewrote [{res['scoped_to']}] only — the rest was spliced "
                 f"back unchanged")
    if res.get("warning"):
        job.line("NOTE: " + res["warning"])
    job.line(f"{len(res.get('text') or '')} characters, seed {res.get('seed')}")
    job.succeed({"text": res.get("text", ""), "target": res.get("target"),
                 "mode": res.get("mode"), "model": res.get("model"),
                 "seed": res.get("seed"), "tokens": res.get("tokens"),
                 "retried": bool(res.get("retried")),
                 "scoped_to": res.get("scoped_to"),
                 "device": dev, "warning": res.get("warning")})


@app.post("/api/lyrics/transcribe", status_code=202)
def post_transcribe(body: dict = Body(default=None)):
    b = body or {}
    path = path_arg(b.get("path"))
    model = str(b.get("model") or "large-v3")
    if not path:
        raise bad("Pick a source track first.")
    if not os.path.isfile(path):
        raise bad("No such file.")
    if model not in ly.WHISPER_MODELS:
        raise bad("Unknown whisper model.")
    seconds = num_arg(b.get("seconds") or 0, 0.0)
    req = {"path": path, "separate": bool(b.get("separate", True)),
           "model": model, "seconds": seconds}
    job = J.Job("transcribe", req)
    _enqueue(job, _run_transcribe)
    return {"job_id": job.id, "kind": "transcribe", "state": job.state,
            "stream_url": f"/api/jobs/{job.id}/events"}


def _run_transcribe(job):
    r = job.request
    job.line("transcribing " + os.path.basename(r["path"]))
    try:
        res = ly.transcribe(r["path"], separate=r["separate"],
                            model=r["model"],
                            seconds=float(r["seconds"] or 0),
                            on_progress=job.emit,
                            on_proc=lambda p: job.extra.__setitem__("proc", p))
    except Exception:
        # Same as _run_upscale: a kill must read as cancelled, not failed.
        if job.cancel_requested:
            job.cancel()
            return
        raise
    summary = (f"{res.get('segments')} segments, language "
               f"{res.get('language')} "
               f"({float(res.get('language_probability') or 0):.2f}), "
               f"{float(res.get('elapsed') or 0):.0f}s"
               f"{' · vocals isolated' if res.get('separated') else ' · full mix'}")
    job.line(summary)
    job.succeed({"text": res.get("text", ""), "segments": res.get("segments"),
                 "language": res.get("language"),
                 "language_probability": res.get("language_probability"),
                 "elapsed": res.get("elapsed"),
                 "separated": bool(res.get("separated")),
                 "summary": summary,
                 "note": "Structure tags are NOT invented — add [verse] / "
                         "[chorus] yourself if you want ACE-Step to follow "
                         "them."})


# =============================================================== 8. library

@app.get("/api/library")
def get_library(q: str = "", fav: int = 0, playlist: str = "",
                workspace: str = "", limit: int = 500, offset: int = 0):
    """`workspace` filters by project tag. Two special values:

       ""          no filtering at all -- every track, tagged or not. This is
                   the default ON PURPOSE: 91 tracks predate workspaces and
                   silently hiding them behind a filter reads as data loss.
       "(none)"    only the untagged ones.
    """
    es = lib.entries()
    if playlist:
        want = pls.tracks(playlist)
        order = {n: i for i, n in enumerate(want)}
        es = [e for e in es if e["name"] in order]
        es.sort(key=lambda e: order[e["name"]])     # playlist order, not date
    if workspace:
        # The tag lives in each sidecar, so this reads the record rather than
        # an index -- nothing to keep in sync, and a restored track brings its
        # workspace back with it.
        want = wsp.clean(workspace).lower()
        if want == "(none)":
            es = [e for e in es if not (lib.read(e["path"]) or {}).get("workspace")]
        else:
            es = [e for e in es
                  if str((lib.read(e["path"]) or {}).get("workspace") or "").lower() == want]
    if fav:
        es = [e for e in es if pls.is_favorite(e["path"])]
    needle = (q or "").strip().lower()
    if needle:
        def hay(e):
            r = e["rec"]
            # `x or ""` because a sidecar can hold an explicit null: str(None)
            # would put "none" in every haystack and q=none would match all.
            return " ".join(str(x or "") for x in (
                titles.derive(r, e["name"]), e["name"], r.get("prompt", ""),
                r.get("lyrics", ""), r.get("model", ""),
                # style is what the Edit sheet writes, so hand-typed tags have
                # to be searchable or tagging buys nothing; workspace makes
                # a workspace name find the project rather than only prompts
                # that say the word; lora_name and variant make q=epoch10 pull
                # up the A/B pair REQUIREMENTS 11 exists to enable.
                r.get("style", ""), r.get("workspace", ""),
                r.get("lora_name", ""), r.get("variant", ""))).lower()
        # Every term must appear, in any order: as one literal substring
        # "doom folk" could not match a style tagged "doom, folk", so word
        # order and punctuation had to be guessed exactly. Nothing a saved
        # query used to find is lost -- a contiguous substring already
        # contains all of its terms -- and each further term only narrows what
        # the first matched. Measured here: q="dark pop" 8 rows -> 10, none
        # dropped; q="acestep 110 bpm" 0 -> 14.
        terms = needle.split()
        pairs = [(e, hay(e)) for e in es]      # hay() reads the record already
        es = [e for e, h in pairs if all(t in h for t in terms)]   # in hand

    total = len(es)
    with_prompt = sum(1 for e in es if e["recorded"] or e["shared_rec"])
    missing_covers = [e["path"] for e in es
                      if not os.path.exists(art.cover_path(e["path"]))]
    kick_covers(missing_covers)

    off = max(0, int(offset or 0))
    lim = max(1, min(2000, int(limit or 500)))
    page = es[off:off + lim]
    out = [api_entry(e) for e in page]
    status = f"{total} tracks · {with_prompt} with a prompt"
    if missing_covers:
        status += f"  ·  {len(missing_covers)} awaiting cover art"
    return {"entries": out, "total": total, "returned": len(out),
            "offset": off, "with_prompt": with_prompt,
            "awaiting_cover": len(missing_covers), "status": status,
            "playlists": pls.names(),
            "workspaces": wsp.load(),
            "workspace": workspace or None}


@app.get("/api/library/{track_id}")
def get_track(track_id: str):
    _es, e = entry_by_id(track_id)
    entry = api_entry(e)
    note = None
    if entry["shared_rec"]:
        other = 1 if entry["take"] == 2 else 2
        note = (f"*Prompt inherited from take {other} of the same run — "
                f"ACE-Step writes two takes and only one carries the record.*")
    return {"entry": entry, "detail_md": lib.detail(e), "shared_note": note}


@app.patch("/api/library/{track_id}")
def patch_track(track_id: str, body: dict = Body(default=None)):
    b = body or {}
    es, e = entry_by_id(track_id)
    ensure_own_record(es, e)
    fields = {}
    if "title" in b:
        fields["title"] = text_arg(b.get("title"), "title").strip() or None
    if "style" in b:
        fields["style"] = text_arg(b.get("style"), "style").strip() or None
    if fields:
        lib.update(e["path"], **fields)
    return {"entry": fresh(track_id),
            "message": f"Saved details for {track_id}"}


@app.post("/api/library/{track_id}/retitle")
def post_retitle(track_id: str, body: dict = Body(default=None)):
    b = body or {}
    es, e = entry_by_id(track_id)
    ensure_own_record(es, e)
    rec = e.get("rec") or {}
    how = "heuristic"
    title = None
    if b.get("use_llm"):
        try:
            got = llm_title.suggest(lyrics=rec.get("lyrics"),
                                    prompt=rec.get("prompt"), n=1, timeout=30)
        except Exception:
            got = []
        if got:
            title, how = got[0], "LLM"
        else:
            # Absent model, too slow, or nothing with substance to work from --
            # every one of those falls back rather than failing the action.
            how = "heuristic (LLM declined)"
    if not title:
        title = titles.derive({**rec, "title": None}, e["name"])
    lib.update(e["path"], title=title)
    return {"title": title, "how": how, "entry": fresh(track_id),
            "message": f"Titled **{title}** ({how})"}


@app.put("/api/library/{track_id}/favorite")
def put_favorite(track_id: str, body: dict = Body(default=None)):
    on = bool((body or {}).get("on", True))
    es, e = entry_by_id(track_id)
    ensure_own_record(es, e)
    pls.set_favorite(e["path"], on)
    return {"favorite": on, "entry": fresh(track_id)}


@app.put("/api/library/{track_id}/rating")
def put_rating(track_id: str, body: dict = Body(default=None)):
    b = body or {}
    raw = b.get("rating")
    if isinstance(raw, bool):
        raise bad("rating must be one of 1, 0, -1.")
    try:
        rating = int(raw)
    except (TypeError, ValueError):
        raise bad("rating must be one of 1, 0, -1.")
    if rating not in (1, 0, -1):
        raise bad("rating must be one of 1, 0, -1.")
    es, e = entry_by_id(track_id)
    ensure_own_record(es, e)
    prev = rating_of(e)
    path = e["path"]
    # The ★ and the 👍 are the same gesture -- except clearing a DISLIKE, which
    # must not touch a favourite the user set independently.
    if rating == 1:
        lib.update(path, rating=1)
        pls.set_favorite(path, True)
    elif rating == -1:
        lib.update(path, rating=-1)
        pls.set_favorite(path, False)
    else:
        lib.update(path, rating=None)
        if prev == 1:
            pls.set_favorite(path, False)
    entry = fresh(track_id)
    word = {1: "Liked", -1: "Disliked", 0: "Rating cleared —"}[rating]
    return {"rating": rating, "favorite": entry["favorite"], "entry": entry,
            "message": f"{word} — {track_id}"}


@app.post("/api/library/{track_id}/played")
def post_played(track_id: str, body: dict = Body(default=None)):
    """Extension (web/REQUIREMENTS.md 2) -- played state in the sidecar.

    Server-side rather than localStorage so the phone and the desktop agree
    about what has already been heard.
    """
    es, e = entry_by_id(track_id)
    ensure_own_record(es, e)
    rec = lib.read(e["path"]) or {}
    asked = num_arg((body or {}).get("plays") or 0, 0, int)
    try:
        stored = int(rec.get("plays") or 0)
    except (TypeError, ValueError):
        stored = 0          # a hand-edited sidecar must not 500 the endpoint
    plays = asked or stored + 1
    lib.update(e["path"], played=True, plays=plays,
               last_played=datetime.now().isoformat(timespec="seconds"))
    return {"played": True, "plays": plays, "entry": fresh(track_id)}


@app.post("/api/library/{track_id}/trash")
def post_trash(track_id: str):
    _es, e = entry_by_id(track_id)
    dest = pls.trash(e["path"])
    if not dest:
        raise ApiError(409, "move_failed",
                       "Could not move that file to the trash folder.")
    return {"trashed": True, "dest": dest,
            "message": f"Moved **{track_id}** to trash\\ (recoverable)."}


@app.get("/api/library/{track_id}/reuse")
def get_reuse(track_id: str):
    _es, e = entry_by_id(track_id)
    entry = api_entry(e)
    if not entry["recorded"]:
        raise ApiError(409, "no_record",
                       "That track has no saved prompt — it predates the "
                       "library, or is an alternate take.")
    rec = e["rec"] or {}
    return {"model": rec.get("model") or "minimax",
            "prompt": rec.get("prompt") or "",
            "lyrics": rec.get("lyrics") or "",
            "duration": float(rec.get("duration") or 30),
            "steps": int(rec.get("steps") or 30),
            "seed": int(rec.get("seed") or 7),
            "instrumental": bool(rec.get("instrumental")),
            "message": f"Loaded **{track_id}** into Generate ✓"}


@app.get("/api/library/{track_id}/cover-source")
def get_cover_source(track_id: str):
    _es, e = entry_by_id(track_id)
    entry = api_entry(e)
    rec = e["rec"] or {}
    probe = post_probe({"path": e["path"], "uploaded": False})
    return {"model": "acestep",
            "prompt": rec.get("prompt") or "",
            "lyrics": rec.get("lyrics") or "",
            "instrumental": bool(rec.get("instrumental")),
            "src_path": e["path"], "src_url": media_url(e["path"]),
            "cover_strength": 1.0, "noise_strength": 0.75,
            "lyrics_probe": probe,
            "message": f"Covering **{entry['title']}** — source loaded in "
                       f"Generate. Edit the style, then Generate."}


@app.post("/api/library/covers")
def post_covers(body: dict = Body(default=None)):
    b = body or {}
    force = bool(b.get("force"))
    cap = max(1, num_arg(b.get("max") or 60, 60, int))
    es = lib.entries()
    todo = [e["path"] for e in es
            if force or not os.path.exists(art.cover_path(e["path"]))]
    n = kick_covers(todo, force=force, cap=cap)
    return {"queued": n, "running": bool(n)}


# ==================================================== 9. playlists / trash

def _pname(name):
    n = name.strip() if isinstance(name, str) else ""
    if not n:
        raise bad("Type a playlist name, or pick an existing one.")
    return n


@app.get("/api/playlists")
def get_playlists():
    data = pls.load()
    return {"playlists": [{"name": n, "count": len(data[n])}
                          for n in sorted(data, key=lambda s: (s.lower(), s))]}


@app.post("/api/playlists", status_code=201)
def post_playlist(body: dict = Body(default=None)):
    name = _pname((body or {}).get("name"))
    created = bool(pls.create(name))
    return {"name": name, "created": created}


@app.delete("/api/playlists/{name}")
def delete_playlist(name: str):
    if not pls.delete_playlist(name):
        raise missing(f"No playlist named {name}.")
    return {"deleted": True}


@app.get("/api/playlists/{name}")
def get_playlist(name: str):
    if _pname(name) not in pls.load():
        raise missing(f"No playlist named {name}.")
    return {"name": name, "tracks": pls.tracks(name)}


@app.post("/api/playlists/{name}/tracks")
def post_playlist_track(name: str, body: dict = Body(default=None)):
    name = _pname(name)
    track_id = (body or {}).get("id")
    path = resolve_id(track_id)
    pls.create(name)                    # "+ Add" creates the playlist too
    added = bool(pls.add(name, path))
    count = len(pls.tracks(name))
    return {"added": added, "count": count,
            "message": f"Added to **{name}** ({count} tracks)" if added
            else f"Already in **{name}** ({count} tracks)"}


@app.delete("/api/playlists/{name}/tracks/{track_id}")
def delete_playlist_track(name: str, track_id: str):
    if not pls.remove(_pname(name), track_id):
        raise missing("That track is not in that playlist.")
    return {"removed": True}


@app.post("/api/playlists/{name}/reorder")
def post_reorder(name: str, body: dict = Body(default=None)):
    b = body or {}
    name = _pname(name)
    delta = num_arg(b.get("delta", 0), 0, int)
    if not pls.reorder(name, b.get("id"), delta):
        raise missing("That track is not in that playlist.")
    return {"ok": True, "tracks": pls.tracks(name)}


@app.get("/api/profile")
def get_profile():
    """Studio-wide settings. Today that is the artist name credited on
    playback.

    An empty artist is the honest unset state, not a value to fill in: the
    lock screen carried a hard-coded name until now, which credited every
    track to a label nobody had typed.
    """
    return prof.load()


@app.put("/api/profile")
def put_profile(body: dict = Body(default=None)):
    """Set the artist name. Clearing it is a legitimate edit, so "" is
    accepted and stored rather than rejected as missing."""
    b = body or {}
    if "artist" not in b:
        raise bad("Nothing to change.", {"field": "artist"})
    raw = str(b.get("artist") or "")
    d = prof.set_artist(raw)
    # Say so when the stored value is not what was typed, rather than
    # letting the field silently disagree with the box it came from.
    note = None
    if raw.strip() and not d["artist"]:
        note = "That name had nothing storable in it."
    elif len(raw.strip()) > prof.MAX_ARTIST:
        note = f"Shortened to {prof.MAX_ARTIST} characters."
    return {"artist": d["artist"], "note": note}


@app.get("/api/workspaces")
def get_workspaces():
    """The workspaces, which is active, and how many tracks each holds.

    A workspace is the project a generation was made FOR, not a list you curate
    afterwards -- that is what playlists are. Membership is a sidecar field, so
    the counts are derived from the library rather than from an index that
    could drift.
    """
    d = wsp.load()
    counts = wsp.counts(lib.entries())
    return {"active": d["active"], "names": d["names"],
            "counts": counts,
            "untagged": counts.get("", 0),
            "note": "New songs are tagged with the active workspace as they "
                    "start. Switching mid-render does not move a song that is "
                    "already generating."}


@app.post("/api/workspaces")
def post_workspace(body: dict = Body(default=None)):
    name = wsp.create(str((body or {}).get("name") or ""))
    if not name:
        raise bad("That workspace name is unusable.", {"field": "name"})
    return {"created": name, "names": wsp.names(), "active": wsp.active()}


@app.put("/api/workspaces/active")
def put_workspace_active(body: dict = Body(default=None)):
    """Switch context. null/"" means new songs are untagged."""
    name = (body or {}).get("name")
    return {"active": wsp.set_active(name), "names": wsp.names()}


@app.delete("/api/workspaces/{name}")
def delete_workspace(name: str):
    """Forget the NAME. Tracks keep their tag and their audio -- see
    workspaces.delete on why this is not a cascade."""
    if not wsp.delete(name):
        raise missing(f"No workspace {name!r}.")
    return {"deleted": name, "names": wsp.names(), "active": wsp.active(),
            "note": "The tracks still carry this tag and are untouched; "
                    "recreating the name brings them back into view."}


@app.get("/api/trash")
def get_trash():
    """What is in the trash, with enough to recognise it.

    The bare filename is a UUID for anything ACE-Step made, which is useless
    for deciding whether to restore something. The sidecar travels into the
    trash with its track, so read it when it is there.

    It is NOT always there: tracks binned before sidecars were written for
    every take have audio and nothing else, so every field below is optional
    and the caller must render a row from `name` and `mtime` alone.
    """
    items = []
    for n in pls.trashed():
        p = os.path.join(TRASHDIR, n)
        try:
            mtime = os.path.getmtime(p)
            size = os.path.getsize(p)
        except OSError:
            mtime, size = None, None

        title = prompt = model = None
        rec = None
        side = os.path.splitext(p)[0] + ".json"
        if os.path.isfile(side):
            try:
                with open(side, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except (OSError, json.JSONDecodeError):
                rec = None
        if rec:
            prompt = (rec.get("prompt") or "").strip() or None
            model = rec.get("model")
            # derive() takes the filename as a last resort and never returns
            # empty, so it handles a record with no title and no prompt too.
            title = rec.get("title_override") or titles.derive(rec, n)

        items.append({
            "name": n,
            "url": media_url(p),
            "mtime": mtime,
            "size": size,
            "title": title,
            "prompt": (prompt[:160] if prompt else None),
            "model": model,
            "has_record": bool(rec),
        })
    return {"items": items, "count": len(items),
            "note": "Restoring moves the file back into the library. Emptying "
                    "the trash is not offered here — delete from the folder if "
                    "that is what you want."}


@app.post("/api/trash/{name}/restore")
def post_restore(name: str):
    dest = pls.restore(name)
    if not dest:
        raise ApiError(409, "exists",
                       "Could not restore — a live track of that name is "
                       "already there, or it is not in the trash.")
    return {"restored": True, "path": dest}


# =========================================== 10. media, upload, filesystem

@app.api_route("/api/media", methods=["GET", "HEAD"])
def get_media(p: str = ""):
    # Report anything outside the roots as not-found: confirming existence
    # would leak the filesystem to a page proxied over the tailnet.
    if not p or not playable(p) or not os.path.isfile(p):
        raise missing("No such file.")
    ext = os.path.splitext(p)[1].lower()
    base = os.path.basename(p)
    cache = ("no-store" if base.startswith("_spec_")
             else "private, max-age=300")
    return FileResponse(
        p, media_type=MEDIA_TYPES.get(ext, "application/octet-stream"),
        headers={"Cache-Control": cache})


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


@app.post("/api/upload", status_code=201)
async def post_upload(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(name)[1].lower()
    if ext not in UPLOAD_EXT:
        raise bad(f"Unsupported file type {ext or '(none)'}. "
                  f"Use {', '.join(UPLOAD_EXT)}.")
    safe = _SAFE_NAME.sub("_", name).strip() or ("upload" + ext)
    os.makedirs(UPLOADDIR, exist_ok=True)
    dest = os.path.join(UPLOADDIR, f"{uuid.uuid4().hex[:8]}_{safe}")
    limit = UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    fh.close()
                    os.remove(dest)
                    raise ApiError(413, "too_large",
                                   f"Files over {UPLOAD_MB} MB are refused.")
                fh.write(chunk)
    except ApiError:
        raise
    except Exception as e:
        try:
            os.remove(dest)
        except OSError:
            pass
        raise ApiError(500, "internal", f"{type(e).__name__}: {e}")
    return {"path": dest, "url": media_url(dest), "name": name, "size": size,
            "uploaded_only": True}


@app.get("/api/fs/stat")
def get_fs_stat(p: str = ""):
    path = (p or "").strip().strip('"')
    if not path:
        return {"exists": False, "is_file": False, "size": None,
                "playable": False, "name": None}
    try:
        is_file = os.path.isfile(path)
        exists = is_file or os.path.isdir(path)
        size = os.path.getsize(path) if is_file else None
    except OSError:
        exists, is_file, size = False, False, None
    return {"exists": exists, "is_file": is_file, "size": size,
            "playable": bool(is_file and playable(path)),
            "name": os.path.basename(path) or None}


# =================================================================== jobs

@app.get("/api/jobs")
def get_jobs(active: int = 0):
    return {"jobs": [j.snapshot() for j in REG.list(active_only=bool(active))]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = REG.get(job_id)
    if job is None:
        raise missing(f"No job {job_id}.")
    return job.snapshot()


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str, request: Request):
    job = REG.get(job_id)
    if job is None:
        raise missing(f"No job {job_id}.")
    last = request.headers.get("last-event-id")
    if last is None:
        last = request.query_params.get("last_event_id")
    return StreamingResponse(
        J.event_stream(job, request, last),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store",
                 "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"})


@app.delete("/api/jobs/{job_id}", status_code=202)
def delete_job(job_id: str):
    job = REG.get(job_id)
    if job is None:
        raise missing(f"No job {job_id}.")
    if job.terminal:
        raise ApiError(409, "not_cancellable", "That job has already finished.")
    # A queued job of ANY kind comes off the lane cleanly -- no process exists
    # yet. Only the running case needs a kind that we know how to kill.
    if REG.cancel(job):
        return {"cancelling": True}
    job.cancel_requested = True
    threading.Thread(target=_force_kill, args=(job,), daemon=True).start()
    return {"cancelling": True}


def _force_kill(job):
    """Kill whatever process is executing `job`, then free the GPU.

    Two shapes of executor, two ways to reach the process:
      * generate/test run inside the persistent model worker  -> SUP.worker.proc
      * upscale/transcribe spawn their own venv child          -> job.extra["proc"]
    """
    if job.kind in ("generate", "test"):
        _kill_worker(job)
        return

    deadline = time.time() + 30
    while time.time() < deadline and not job.terminal:
        p = job.extra.get("proc")
        if p is not None:
            _kill_tree(p)
            break
        time.sleep(0.25)                 # the child may not be spawned yet
    # Nothing to unload: these jobs use their own process, and the resident
    # model worker (if any) is a separate tenant the user did not ask to evict.


def _kill_tree(p):
    """Kill a child AND its descendants.

    Apollo and AudioSR are launched as `powershell.exe script.ps1`, which then
    spawns the real python process. Killing only the shell orphans the python,
    which keeps the GPU -- the opposite of what a cancel is for. taskkill /T
    walks the tree; p.kill() is the fallback and the plain-child case.
    """
    try:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       capture_output=True, timeout=20)
    except Exception:
        pass
    try:
        p.kill()
    except Exception:
        pass


def _kill_worker(job):
    """Kill the worker process directly -- NEVER SUP.unload().

    unload() takes the same RLock the running generation holds, so it would
    block until that generation finished. Closing the worker's stdout makes
    `Worker._read_until` raise WorkerDied, which propagates out of
    SUP.generate and releases the lock; only then is unload() safe.

    During ensure() the new worker is not assigned to SUP.worker until
    Worker.start() returns (~20 s of model loading), so there is a window with
    no handle to kill -- hence the retry.
    """
    deadline = time.time() + 30
    killed = False
    while time.time() < deadline and not job.terminal:
        w = getattr(SUP, "worker", None)
        p = getattr(w, "proc", None)
        if p is not None:
            try:
                p.kill()
                killed = True
            except Exception:
                pass
            break
        time.sleep(0.25)
    if killed:
        time.sleep(1.0)
        try:
            SUP.unload()        # safe now: the lock has been released
        except Exception:
            pass


# ============================================================== test hooks

if TEST_HOOKS:
    @app.post("/api/_test/job", status_code=202)
    def post_test_job(body: dict = Body(default=None)):
        """Synthetic GPU-lane job -- exercises the queue and SSE, no GPU.

        Enabled only with STUDIO_TEST_HOOKS=1; inert in production.
        """
        b = body or {}
        job = J.Job("test", {"steps": int(b.get("steps") or 3),
                             "delay": float(b.get("delay") or 0.2),
                             "lead": float(b.get("lead") or 0.0),
                             "fail": bool(b.get("fail"))})
        _enqueue(job, _run_test)
        return {"job_id": job.id, "kind": "test", "state": job.state,
                "stream_url": f"/api/jobs/{job.id}/events"}

    @app.get("/api/_test/subs")
    def get_test_subs(job_id: str = ""):
        """Live SSE subscriber count -- proves a vanished client is detached."""
        job = REG.get(job_id)
        if job is None:
            raise missing(f"No job {job_id}.")
        with job.lock:
            return {"job_id": job.id, "subs": len(job.subs),
                    "dropped": sum(1 for s in job.subs if s.dropped)}

    def _run_test(job):
        r = job.request
        # `lead` gives a test client time to attach before the first event, so
        # it can observe them on the wire instead of only in the hello.
        time.sleep(r["lead"])
        job.line("synthetic job starting")
        job.phase("generate", "synthetic phase")
        for i in range(r["steps"]):
            if job.cancel_requested:
                job.cancel()
                return
            time.sleep(r["delay"])
            job.emit({"msg": f"working {int((i + 1) / r['steps'] * 100)}%",
                      "stage": "work", "frac": (i + 1) / r["steps"]})
        if r["fail"]:
            raise RuntimeError("synthetic failure")
        icon = os.path.join(ASSETS, "icon.png")
        job.artifact("image", "test", icon, media_url(icon))
        job.succeed({"ok": True, "steps": r["steps"]})


# ================================================================= static

mimetypes.add_type("application/manifest+json", ".webmanifest")


@app.get("/manifest.webmanifest")
def get_manifest():
    p = os.path.join(STATIC, "manifest.webmanifest")
    if os.path.exists(p):
        return FileResponse(p, media_type="application/manifest+json")
    return JSONResponse({
        "name": "AudioDev Studio", "short_name": "Studio",
        "start_url": "/", "display": "standalone",
        "background_color": "#141416", "theme_color": "#141416",
        "icons": [{"src": "/static/icon.png", "sizes": "512x512",
                   "type": "image/png", "purpose": "any maskable"}],
    }, media_type="application/manifest+json")


# The app shell must never be served from cache without asking first.
#
# StaticFiles sends ETag and Last-Modified but no Cache-Control, which leaves
# the browser to guess -- and the heuristic (roughly 10% of the age of the
# file) means a script edited an hour ago can be served from cache for minutes
# without a request reaching us. On a desktop reload that is invisible; on an
# iOS home-screen app it is not, and it cost real debugging time: index.html
# updated while create.js did not, so a <select> the new markup had added sat
# there empty because the old script never populated it. "Works on the PC,
# says no options on the phone" is exactly that shape.
#
# `no-cache` is not `no-store`: the file is still cached, the browser just has
# to revalidate. The ETag then answers 304 with no body, so the cost is one
# conditional request per asset per load on a tailnet.
SHELL_EXT = (".html", ".js", ".css", ".webmanifest", ".json")


@app.middleware("http")
async def revalidate_app_shell(request, call_next):
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith(SHELL_EXT):
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


# ---------------------------------------------------------------------------
# Versioned module graph.
#
# `Cache-Control: no-cache` fixes the ENTRY point, and is still not enough. An
# ES module graph is cached by URL in the browser's module map, and on iOS
# Safari that map survives an ordinary reload: `app.js` gets re-fetched while
# `./create.js` -- which it imports -- is served from the map, unasked-for and
# arbitrarily old. Observed exactly that: a phone showing a control that new
# markup had added, sitting empty because the script that fills it was weeks
# stale, through repeated reloads.
#
# So the URLs change whenever the files do. Every relative import inside a
# served .js gains `?v=<build>`, and index.html's entry <script> gains it too.
# A new build is a new URL, and a new URL cannot be answered from the old map.
_IMPORT_RE = re.compile(
    r"""((?:^|\s)(?:import|export)\b[^;'"]*?['"])(\.{1,2}/[^'"?]+?\.js)(['"])""",
    re.M)


def _build_id():
    """Newest mtime across the shell. Cheap, and changes exactly when we edit."""
    newest = 0.0
    try:
        for fn in os.listdir(STATIC):
            if fn.endswith((".js", ".html", ".css")):
                newest = max(newest, os.path.getmtime(os.path.join(STATIC, fn)))
    except OSError:
        pass
    return f"{int(newest):x}"


def _shell_headers():
    return {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/{name}.js")
def get_module(name: str):
    p = os.path.normpath(os.path.join(STATIC, name + ".js"))
    if not p.startswith(STATIC) or not os.path.isfile(p):
        raise missing(f"No module {name}.js")
    with open(p, encoding="utf-8") as fh:
        src = fh.read()
    v = _build_id()
    src = _IMPORT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}?v={v}{m.group(3)}",
                         src)
    return Response(src, media_type="text/javascript; charset=utf-8",
                    headers=_shell_headers())


@app.get("/")
def get_index():
    p = os.path.join(STATIC, "index.html")
    if not os.path.isfile(p):
        raise missing("index.html")
    with open(p, encoding="utf-8") as fh:
        html = fh.read()
    v = _build_id()
    html = re.sub(r'(<script[^>]*\ssrc=")(/[^"?]+\.js)(")',
                  lambda m: f"{m.group(1)}{m.group(2)}?v={v}{m.group(3)}", html)
    html = re.sub(r'(<link[^>]*\shref=")(/[^"?]+\.css)(")',
                  lambda m: f"{m.group(1)}{m.group(2)}?v={v}{m.group(3)}", html)
    return Response(html, media_type="text/html; charset=utf-8",
                    headers=_shell_headers())


os.makedirs(STATIC, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static-files")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
