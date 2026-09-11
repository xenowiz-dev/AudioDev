"""Named playlists, favourites, and a delete that can be undone.

Three kinds of state, kept three different ways, because each has a different
natural home:

* A playlist is a relationship BETWEEN tracks, so it has no per-file home --
  it lives in one small `playlists.json` beside the audio.
* A favourite is a fact ABOUT one track, so it goes in that track's own
  sidecar (see `library.py`), where the rest of the record already lives and
  where it survives the file being moved or copied.
* A delete is a move into `trash\\`. `library.entries()` globs the top level
  only, so anything in a subfolder is already invisible to the library -- the
  same mechanism that hides `upscaled\\` is what makes delete reversible here.
  Nothing in this module ever calls `os.remove` on a track.

Playlists hold basenames, not paths: the folder is the namespace, and a name
survives the whole tree being moved. They are deliberately NOT pruned when a
track is trashed -- the entry going dangling for a while is the price of
`restore()` putting the track back where the playlist still expects it.

Pure stdlib, so this imports in the studio venv (no numpy/soundfile). Nothing
here raises on a missing or corrupt file: a UI is not the place to find out
the index went bad. Reads degrade to empty, writes that cannot happen return
False/None.
"""

import json
import os
import tempfile
import threading
from datetime import datetime

import library as lib

# Derived at call time, not bound at import: one knob (OUTDIR) then redirects
# the playlist file, the trash and the sidecars together, which is what a test
# needs to stay off the user's real generations.
OUTDIR = lib.OUTDIR
PLAYLISTS_NAME = "playlists.json"
TRASH_NAME = "trash"

# Every mutator is load-modify-save on one shared file, and Gradio dispatches
# event handlers on a thread pool -- two edits in flight at once otherwise read
# the same snapshot and the second one silently drops the first. In-process
# only: two studio instances on one folder would still race, which is not a
# thing the UI can do to itself.
_LOCK = threading.Lock()


def _playlists_path():
    return os.path.join(OUTDIR, PLAYLISTS_NAME)


def _trash_dir():
    return os.path.join(OUTDIR, TRASH_NAME)


def _clean(name):
    return name.strip() if isinstance(name, str) else ""


def _path(p):
    """Whatever a UI event handed us, as a string path -- "" if it is not one.

    The int is the trap worth guarding: `os.path.isfile()` reads a bare int as
    a file DESCRIPTOR, so a stray row index (exactly what a Gradio select event
    carries) sails straight through an existence check and only blows up later
    inside basename(). Anything that is not a path reads as absent instead.
    """
    if isinstance(p, str):
        return p
    try:
        p = os.fspath(p)                # pathlib.Path and friends are fine
    except TypeError:
        return ""
    return p if isinstance(p, str) else ""


# --------------------------------------------------------------- playlists

def load():
    """Every playlist: `{name: [basename, ...]}`. Absent or corrupt reads {}."""
    try:
        with open(_playlists_path(), encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for name, items in data.items():
        # Salvage rather than reject: the file is small enough to hand-edit,
        # and one bad entry should not cost the user every other playlist.
        if not isinstance(name, str) or not isinstance(items, list):
            continue
        seen, keep = set(), []
        for it in items:
            if isinstance(it, str) and it and it not in seen:
                seen.add(it)
                keep.append(it)
        out[name] = keep
    return out


def _save(data):
    p = _playlists_path()
    tmp = None
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(p),
                                   prefix=".playlists-", suffix=".tmp")
        # Temp file in the same directory so os.replace stays a same-volume
        # rename, and closed before the replace -- Windows refuses to replace
        # a file that is still open.
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except Exception:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def names():
    """Playlist names, sorted for display (case-insensitively)."""
    return sorted(load(), key=lambda s: (s.lower(), s))


def create(name):
    name = _clean(name)
    if not name:
        return False
    with _LOCK:
        data = load()
        if name in data:
            return False    # already exists: the caller made no new playlist
        data[name] = []
        return _save(data)


def delete_playlist(name):
    """Forget a playlist. The tracks themselves are untouched."""
    name = _clean(name)
    with _LOCK:
        data = load()
        if name not in data:
            return False
        del data[name]
        return _save(data)


def add(name, audio_path):
    """Append a track. False if the playlist is unknown or already has it."""
    name = _clean(name)
    b = os.path.basename(_path(audio_path))
    if not name or not b:
        return False
    with _LOCK:
        data = load()
        if name not in data or b in data[name]:
            return False
        data[name].append(b)
        return _save(data)


def remove(name, audio_path):
    name = _clean(name)
    b = os.path.basename(_path(audio_path))
    with _LOCK:
        data = load()
        if name not in data or b not in data[name]:
            return False
        data[name].remove(b)
        return _save(data)


def tracks(name):
    """Basenames in playlist order. Unknown playlist reads as empty.

    Entries are not checked against the disk: a trashed track is coming back
    if the user restores it, and silently dropping it here would lose its
    place in the running order.
    """
    return list(load().get(_clean(name), []))


def reorder(name, basename, delta):
    """Move one track `delta` places (negative is earlier). Clamped at the ends.

    True whenever the track is there -- nudging the first track up is a no-op,
    not an error, which is what a pair of up/down buttons wants.
    """
    name = _clean(name)
    b = os.path.basename(_path(basename))
    try:
        delta = int(delta)
    except (TypeError, ValueError):
        return False
    with _LOCK:
        data = load()
        items = data.get(name)
        if not items or b not in items:
            return False
        i = items.index(b)
        j = max(0, min(len(items) - 1, i + delta))
        if j == i:
            return True                 # already as far as it can go
        items.insert(j, items.pop(i))
        return _save(data)


# -------------------------------------------------------------- favourites

def _record(audio_path):
    """The sidecar as a mapping, or None if there is nothing usable.

    `lib.read()` only guards against a PARSE failure: a sidecar holding valid
    JSON that is not an object (a bare list, string or number -- a hand-edit
    gone wrong, or a half-written file that happens to parse) comes straight
    back, and every caller here would then treat it as a mapping and raise.
    Same verdict as a corrupt file: no record.
    """
    rec = lib.read(audio_path)
    return rec if isinstance(rec, dict) else None


def set_favorite(audio_path, on=True):
    """Star or unstar a track, in its own sidecar."""
    audio_path = _path(audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return False
    on = bool(on)
    rec = _record(audio_path)
    if rec is None:
        if not on:
            return True     # no record, nothing starred: already in that state
        # First record for a track that never had one (a CLI run, or the
        # alternate take ACE-Step emits). Seed `created` from the file's mtime
        # -- write() would otherwise stamp now, and the record would claim the
        # track was made at the moment someone starred it.
        try:
            when = datetime.fromtimestamp(os.path.getmtime(audio_path))
            rec = {"created": when.isoformat(timespec="seconds")}
        except OSError:
            rec = {}
    # write()'s own first parameter is named `audio_path`, so a record that
    # carries that key -- nothing we write does, but a hand-edit or a later
    # schema could -- would blow the splat up with "multiple values". Drop it:
    # the real basename is already stored under `audio`.
    rec = {k: v for k, v in rec.items() if k != "audio_path"}
    rec["favorite"] = on
    # The whole record goes back through write(), which rebuilds it from
    # scratch: any field not passed here would be dropped on the floor.
    return lib.write(audio_path, **rec) is not None


def is_favorite(audio_path):
    audio_path = _path(audio_path)
    if not audio_path:
        return False
    return bool((_record(audio_path) or {}).get("favorite"))


# ------------------------------------------------------------------- trash

def _free_name(dirpath, basename):
    """A free stem in `dirpath` -- audio and sidecar move together, so both
    names have to be clear before the stem can be used."""
    stem, ext = os.path.splitext(basename)
    cand, n = stem, 1
    while (os.path.exists(os.path.join(dirpath, cand + ext))
           or os.path.exists(os.path.join(dirpath, cand + ".json"))):
        cand = f"{stem}-{n}"
        n += 1
    return os.path.join(dirpath, cand + ext)


def trash(audio_path):
    """Move a track and its sidecar into `trash\\`. Returns the new path.

    Reversible on purpose: `restore()` is the whole point of not deleting.
    """
    audio_path = _path(audio_path)
    if not audio_path or not os.path.isfile(audio_path):
        return None
    d = _trash_dir()
    try:
        os.makedirs(d, exist_ok=True)   # lazily -- an empty trash\ is clutter
        dest = _free_name(d, os.path.basename(audio_path))
        os.replace(audio_path, dest)
    except OSError:
        return None
    # The sidecar follows under the SAME stem, so the pair stays discoverable
    # and restore() only ever has to be told the audio name.
    src_json = lib.sidecar_path(audio_path)
    if os.path.isfile(src_json):
        try:
            os.replace(src_json, lib.sidecar_path(dest))
        except OSError:
            # A record left behind at the top level is worse than a trash that
            # did not happen: the next generation of the same name inherits it
            # and the library reports a brand-new track's provenance as the
            # dead one's. Half a move is the one outcome we cannot leave, so
            # put the audio back -- a refused trash is retryable and visible,
            # a lying record is neither.
            try:
                os.replace(dest, audio_path)
            except OSError:
                pass    # rollback failed too; still refuse, the pair is split
            return None
    return dest


def trashed():
    """Basenames waiting in `trash\\` -- the restore menu.

    Ordered by the track's own mtime (os.replace keeps it), so the listing
    matches the library's ordering rather than the order things were binned.
    """
    d = _trash_dir()
    out = []
    for name in os.listdir(d) if os.path.isdir(d) else []:
        if not name.lower().endswith(lib.AUDIO_EXT):
            continue
        try:
            out.append((os.path.getmtime(os.path.join(d, name)), name))
        except OSError:
            continue
    return [n for _, n in sorted(out, key=lambda t: -t[0])]


def restore(basename):
    """Move a trashed track (and its sidecar) back. Returns the new path."""
    b = os.path.basename(_path(basename))
    if not b:
        return None
    src = os.path.join(_trash_dir(), b)
    if not os.path.isfile(src):
        return None
    dest = os.path.join(OUTDIR, b)
    if os.path.exists(dest):
        # Refuse rather than rename around it: a restore must never land on
        # top of a live track, and renaming would only move the confusion.
        return None
    try:
        os.replace(src, dest)
    except OSError:
        return None
    src_json = lib.sidecar_path(src)
    if os.path.isfile(src_json) and not os.path.exists(lib.sidecar_path(dest)):
        try:
            os.replace(src_json, lib.sidecar_path(dest))
        except OSError:
            pass
    return dest
