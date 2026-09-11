"""Who the output is credited to.

A studio-wide setting, not a fact about one track: it names the person at the
keyboard, so it belongs on the server rather than in a browser. The phone and
the desktop both play, and two localStorage copies would disagree about the
credit shown on a lock screen.

**There is no default artist and there must not be one.** The lock screen
previously carried a hard-coded name, which credited every track to a label
nobody had typed. Unset means unset: the player omits the field, iOS shows the
title alone, and nothing is invented on the user's behalf. `artist == ""` is
the honest empty state, not a migration to fix.

Deliberately NOT written into the audio's tags, for the reason library.py
gives: a Suno track arriving with "made with suno" in its LIST/INFO chunk is
what this toolkit exists to avoid, and baking a credit into the file replays
that on our own output. Provenance you keep, not provenance you ship.

One flat object rather than a bare string, so the next studio-wide setting has
somewhere to go without a migration. Pure stdlib, degrades to empty on a
missing or corrupt file, atomic writes -- same contract as workspaces.py.

Named `settings`, not `profile`: `profile` is a stdlib module (the profiler),
and api.py puts the studio folder at the FRONT of sys.path, so a file of that
name here would shadow it for the whole server process.
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
FILE = "settings.json"
MAX_ARTIST = 64

_lock = threading.RLock()
# Control characters only. A name is the user's to choose -- punctuation,
# accents, "!", "&" and non-Latin scripts all belong to somebody -- so this
# strips what would break a JSON round trip or a lock-screen line, and nothing
# else. Newlines go because the credit renders on one line.
_BAD = re.compile(r"[\x00-\x1f\x7f]+")

DEFAULT = {"artist": ""}


def _path():
    return os.path.join(OUTDIR, FILE)


def clean(name):
    """A storable artist name, or "" if nothing usable is left."""
    name = _BAD.sub(" ", str(name or ""))
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name[:MAX_ARTIST]


def load():
    """{"artist": str}. Never raises -- a broken file reads as unset."""
    with _lock:
        try:
            with open(_path(), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return dict(DEFAULT)
        if not isinstance(d, dict):
            return dict(DEFAULT)
        return {"artist": clean(d.get("artist"))}


def _save(d):
    try:
        os.makedirs(OUTDIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=OUTDIR, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(d, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _path())     # atomic; a torn file is worse than none
        return True
    except OSError:
        return False


def artist():
    return load()["artist"]


def set_artist(name):
    """Store a name. Returns the stored value, which may be "" -- clearing the
    field is a legitimate edit, not a failure."""
    with _lock:
        d = load()
        d["artist"] = clean(name)
        _save(d)
        return d
