"""A record of everything generated, as a JSON sidecar next to each track.

One `<track>.json` per audio file, rather than a central index: the record
travels with the file if it is moved or copied, cannot drift out of sync with
an index, and is greppable. The library view is just a scan of the folder, so
it is idempotent and self-healing -- files that appear later (CLI runs, tests)
show up on their own, and nothing has to be migrated.

Deliberately NOT embedded in the audio's RIFF/ID3 tags. This whole toolkit
exists partly because a Suno track carried "made with suno" in its LIST/INFO
chunk; baking prompts into the file would replay that on our own output. A
sidecar is provenance you keep, not provenance you ship.

The record holds the lyrics too, so the library doubles as a lyrics store --
`lyrics_probe.py` reads these sidecars, which means covering one of our own
generations finds its lyrics automatically.
"""

import glob
import json
import os
from datetime import datetime

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

OUTDIR = os.path.join(ROOT, "Music", "studio")
AUDIO_EXT = (".wav", ".flac")
SCHEMA = 1


def sidecar_path(audio_path):
    return os.path.splitext(audio_path)[0] + ".json"


def write(audio_path, **fields):
    """Write the sidecar for a generated track. Returns its path or None."""
    try:
        rec = {"schema": SCHEMA,
               "created": datetime.now().isoformat(timespec="seconds"),
               "audio": os.path.basename(audio_path)}
        rec.update({k: v for k, v in fields.items() if v is not None})
        p = sidecar_path(audio_path)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        return p
    except Exception:
        # Never let bookkeeping fail a generation the user already waited for.
        return None


def update(audio_path, **fields):
    """Merge fields into a track's sidecar, creating it if absent.

    Read-modify-write rather than write(): an edit must not drop the prompt,
    lyrics or settings the track was generated with. Passing None for a field
    REMOVES it, which is how a user clears a title override and falls back to
    the derived one.
    """
    try:
        rec = read(audio_path) or {"schema": SCHEMA,
                                   "audio": os.path.basename(audio_path)}
        for k, v in fields.items():
            if v is None:
                rec.pop(k, None)
            else:
                rec[k] = v
        rec["edited"] = datetime.now().isoformat(timespec="seconds")
        p = sidecar_path(audio_path)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)      # atomic: a torn sidecar reads as "no record"
        return True
    except Exception:
        return False


def read(audio_path):
    p = sidecar_path(audio_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None      # corrupt or half-written: treat as no record


def entries(outdir=OUTDIR):
    """Every track in the folder, newest first, with its record if it has one.

    Top level only -- `upscaled\\` holds derivatives, which are reachable from
    the record that produced them.
    """
    out = []
    for ext in AUDIO_EXT:
        for a in glob.glob(os.path.join(outdir, f"*{ext}")):
            try:
                mtime = os.path.getmtime(a)
            except OSError:
                continue
            rec = read(a) or {}
            out.append({
                "path": a,
                "name": os.path.basename(a),
                "mtime": mtime,
                "recorded": bool(rec),
                "rec": rec,
            })
    out.sort(key=lambda e: -e["mtime"])
    return group_takes(out)


TAKE_WINDOW = 2.5      # seconds


def group_takes(entries_, window=TAKE_WINDOW):
    """Fold each generation's multiple takes into one group, in place.

    ACE-Step emits TWO clips per run and the worker keeps one, so the sibling
    would otherwise sit in the library as an orphan with no prompt. Measured
    across 18 runs on this machine, the pair is always written 0-1 s apart,
    while separate generations are minutes apart -- so mtime proximity
    recovers the relationship, retroactively, for tracks that predate the
    sidecars entirely.

    Conservative on purpose: same file extension and inside `window`, since a
    wrong pairing would attach one track's prompt to unrelated audio. The GPU
    slot serialises generations, so two real runs cannot land this close.

    Adds `take` (1-based), `takes` (group size) and `shared_rec` (True when the
    record was inherited from a sibling) to each entry.
    """
    by_time = sorted(entries_, key=lambda e: e["mtime"])
    group, groups = [], []
    for e in by_time:
        if (group and e["mtime"] - group[-1]["mtime"] <= window
                and os.path.splitext(e["name"])[1]
                == os.path.splitext(group[-1]["name"])[1]):
            group.append(e)
        else:
            if group:
                groups.append(group)
            group = [e]
    if group:
        groups.append(group)

    for g in groups:
        donor = next((x for x in g if x["recorded"]), None)
        for i, e in enumerate(g, 1):
            e["take"], e["takes"] = i, len(g)
            e["shared_rec"] = False
            if donor is not None and not e["recorded"]:
                # Same run, same prompt -- but keep `recorded` False so the
                # UI never claims a sidecar exists, and Reuse still refuses.
                e["rec"] = donor["rec"]
                e["shared_rec"] = True
    return entries_


def _short(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def rows(entries_):
    """Table rows for the browser. Order matches `entries_` exactly."""
    out = []
    for e in entries_:
        r = e["rec"]
        when = datetime.fromtimestamp(e["mtime"]).strftime("%m-%d %H:%M")
        if not e["recorded"]:
            # Honest rather than blank: these predate the sidecars, or are the
            # second variant ACE-Step emits per run, or are test output.
            out.append([when, "—", "—", "(no record — found on disk)"])
            continue
        secs = r.get("seconds")
        length = f"{secs:.0f}s" if isinstance(secs, (int, float)) else "—"
        tag = r.get("model", "?")
        if r.get("preview"):
            tag += " · preview"
        if r.get("cover_src"):
            tag += " · cover"
        out.append([when, tag, length, _short(r.get("prompt"), 90)])
    return out


HEADERS = ["When", "Model", "Length", "Prompt"]


def detail(entry):
    """Markdown detail for one entry."""
    if not entry:
        return "Select a track above."
    r = entry["rec"]
    if not entry["recorded"]:
        return (f"### {entry['name']}\n\n*No sidecar record.* This file "
                f"predates the library, is the alternate take ACE-Step emits "
                f"alongside each generation, or came from a CLI/test run.")
    bits = []
    for label, key, fmt in (
            ("model", "model", str), ("seed", "seed", str),
            ("steps", "steps", str), ("duration", "duration", str),
            ("length", "seconds", lambda v: f"{v:.1f}s"),
            ("rate", "sampling_rate", lambda v: f"{v} Hz"),
            ("took", "elapsed", lambda v: f"{v:.0f}s")):
        v = r.get(key)
        if v is not None:
            try:
                bits.append(f"{label} **{fmt(v)}**")
            except Exception:
                pass
    head = f"### {entry['name']}\n\n" + " · ".join(bits)
    if r.get("cover_src"):
        head += (f"\n\ncover of `{os.path.basename(r['cover_src'])}` "
                 f"(strength {r.get('cover_strength')}, "
                 f"noise {r.get('noise_strength')})")
    if r.get("post_kind") and r["post_kind"] != "none":
        head += f"\n\npost-processed with **{r['post_kind']}**"
    return head
