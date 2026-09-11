"""Workspaces: which project a generation belongs to.

A workspace and a playlist are not the same thing, and conflating them is the
mistake this module exists to avoid:

  * A **playlist** is a relationship BETWEEN tracks with a goal -- an album, a
    set. You curate it afterwards, from anywhere, and order matters. It lives
    in `playlists.json` (see playlists.py).
  * A **workspace** is a fact ABOUT one track: the project it was made for.
    Cinematic attempts should not be mixed in with pop attempts. You do not
    add to it afterwards -- a track is born into whichever workspace was
    active when you pressed Generate.

playlists.py already states the rule that decides where each kind of state
lives: a relationship between tracks goes in a shared index, a fact about one
track goes in that track's own sidecar. So workspace membership is a sidecar
field, which means it survives the file being moved, copied or restored from
the trash, and no index can drift out of sync with the audio.

What this file holds instead is the small part membership cannot express: the
LIST of workspaces (so an empty new one can exist before anything is in it)
and which one is ACTIVE. Active lives on the server, not in a browser, because
the phone and the desktop both generate and two localStorage copies would
disagree about where tonight's songs went.

`active = None` means untagged, which is also the migration story: every track
that predates this file has no workspace and must stay visible. Any view that
filters by workspace needs an "Everything" option for exactly that reason.

Pure stdlib, degrades to empty on a missing or corrupt file, atomic writes --
same contract as playlists.py.
"""

import json
import os
import re
import tempfile
import threading

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

OUTDIR = os.environ.get("STUDIO_OUTDIR", os.path.join(ROOT, "Music", "studio"))
FILE = "workspaces.json"
MAX_NAME = 48

_lock = threading.RLock()
_BAD = re.compile(r"[^\w \-&'.,()]+", re.UNICODE)


def _path():
    return os.path.join(OUTDIR, FILE)


def clean(name):
    """A storable workspace name, or "" if nothing usable is left."""
    name = _BAD.sub(" ", str(name or "")).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name[:MAX_NAME]


def load():
    """{"active": str|None, "names": [str]}. Never raises."""
    try:
        with open(_path(), encoding="utf-8") as fh:
            d = json.load(fh)
        if not isinstance(d, dict):
            raise ValueError
    except (OSError, json.JSONDecodeError, ValueError):
        return {"active": None, "names": []}
    names = [clean(n) for n in (d.get("names") or []) if clean(n)]
    # De-duplicate while keeping order; a hand-edited file is fair game.
    seen, out = set(), []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    active = clean(d.get("active") or "")
    if active and active.lower() not in seen:
        out.append(active)          # active must always be a member
    return {"active": active or None, "names": out}


def _save(d):
    try:
        os.makedirs(OUTDIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=OUTDIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path())     # atomic; a torn index is worse than none
        return True
    except OSError:
        return False


def names():
    return load()["names"]


def active():
    return load()["active"]


def create(name):
    """Add a workspace. Returns its stored name, or None if unusable."""
    name = clean(name)
    if not name:
        return None
    with _lock:
        d = load()
        if name.lower() not in {n.lower() for n in d["names"]}:
            d["names"].append(name)
            _save(d)
    return name


def set_active(name):
    """Switch the active workspace. `None` or "" means untagged."""
    name = clean(name) if name else ""
    with _lock:
        d = load()
        if not name:
            d["active"] = None
        else:
            if name.lower() not in {n.lower() for n in d["names"]}:
                d["names"].append(name)
            # Return the stored spelling, not the caller's casing.
            name = next(n for n in d["names"] if n.lower() == name.lower())
            d["active"] = name
        _save(d)
        return d["active"]


def delete(name):
    """Forget a workspace NAME. Tracks keep their tag and their audio.

    Deliberately not a cascade: the tag is in each track's sidecar, and
    rewriting 90 sidecars because a label was removed would be a destructive
    answer to a cosmetic request. The tracks simply stop being reachable by
    that chip until the name is recreated.
    """
    name = clean(name)
    with _lock:
        d = load()
        keep = [n for n in d["names"] if n.lower() != name.lower()]
        if len(keep) == len(d["names"]):
            return False
        d["names"] = keep
        if d["active"] and d["active"].lower() == name.lower():
            d["active"] = None
        return _save(d)


def rename(old, new):
    """Rename in the registry only; see `delete` on why tracks are untouched.

    Returns the new stored name, or None. The caller may choose to retag
    tracks -- `library.update(path, workspace=new)` per track -- but that is a
    decision about the user's data, not a side effect of editing a label.
    """
    old, new = clean(old), clean(new)
    if not old or not new:
        return None
    with _lock:
        d = load()
        idx = next((i for i, n in enumerate(d["names"])
                    if n.lower() == old.lower()), None)
        if idx is None:
            return None
        d["names"][idx] = new
        if d["active"] and d["active"].lower() == old.lower():
            d["active"] = new
        _save(d)
        return new


def counts(entries):
    """{workspace: n} over library entries, plus "" for untagged."""
    out = {}
    for e in entries or []:
        k = (e.get("workspace") or "") if isinstance(e, dict) else ""
        out[k] = out.get(k, 0) + 1
    return out
