"""Turn a folder of Suno downloads into an ACE-Step training dataset.

Suno writes a `<track>.wav.txt` sidecar next to each file. Most of it is a
human-readable summary, but the last section is `--- Raw API Response ---`
followed by the clip's real JSON, and THAT is what this reads: the summary
loses the newlines inside the style prompt and carries no play count. The
prose sections are only a fallback, for a download interrupted mid-write.

What comes out is the schema ACE-Step's preprocessor wants:

    {"metadata": {"custom_tag": ..., "tag_position": "prepend", ...},
     "samples": [{"filename", "audio_path", "caption", "lyrics",
                  "duration", "bpm", "keyscale", ...}, ...]}

The interesting work is deduplication. Suno hands back two clips per prompt
and a re-download repeats them, so a library accumulates two different things:

  * exact repeats -- same clip `id`, byte-identical audio. Keep one.
  * sibling takes -- same title, DIFFERENT `id`, genuinely different audio
    from one prompt. Keeping both teaches the model that prompt twice and
    over-weights it, so keep the take that was actually played.

`play_count` is the tiebreak, then upvotes, then liked, then the longer take,
then the clip id -- every step deterministic, so two runs over the same folder
produce the same dataset.

    python build_dataset.py --src "C:/Users/Kevin/Downloads/Suno Playlist 8-11-2026" --tag xenowiz
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

RAW_MARK = "--- Raw API Response ---"
AUDIO_EXT = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")

# Suno puts the whole style description into `tags`, newlines and all. Collapse
# it: ACE-Step's caption is one text field, and a caption with blank lines in
# it reads as several captions once tokenized.
WS = re.compile(r"\s*\n\s*")
MULTISPACE = re.compile(r"[ \t]{2,}")


def clean(s):
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = WS.sub(" ", s).strip()
    return MULTISPACE.sub(" ", s)


def parse_sidecar(path):
    """Everything known about one clip, or None when the sidecar is unusable."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None

    i = text.find(RAW_MARK)
    raw = {}
    if i >= 0:
        try:
            raw = json.loads(text[i + len(RAW_MARK):])
        except json.JSONDecodeError:
            raw = {}

    meta = raw.get("metadata") or {}
    out = {
        "id": raw.get("id") or "",
        "title": clean(raw.get("title")),
        "caption": clean(meta.get("tags")),
        "lyrics": (meta.get("prompt") or "").strip(),
        "duration": float(meta.get("duration") or 0),
        "play_count": int(raw.get("play_count") or 0),
        "upvote_count": int(raw.get("upvote_count") or 0),
        "liked": bool(raw.get("is_liked")),
        "instrumental": bool(meta.get("make_instrumental")
                             or not meta.get("has_vocal", True)),
        "model": raw.get("model_name") or raw.get("major_model_version") or "",
        "persona": meta.get("persona_id") or "",
    }

    if not out["title"]:
        m = re.search(r"^Title:\s*(.+)$", text, re.M)
        out["title"] = clean(m.group(1)) if m else ""
    if not out["caption"]:
        m = re.search(r"--- Creation Details ---\s*\nPrompt:\s*(.*?)"
                      r"(?=\n--- |\nCover Art URL:|\Z)", text, re.S)
        cand = clean(m.group(1)) if m else ""
        # Suno's "Prompt:" section is the LYRICS field, not the style field --
        # the style lives in metadata.tags. Using it as a caption is only ever
        # right when the user typed a style there, so refuse anything carrying
        # section markers. One track ("Snap~There goes another gear") trained
        # with its own lyrics as its style prompt because of this fallback.
        out["caption"] = "" if cand.count("[") >= 3 else cand
    return out


def probe(path):
    """(seconds, sample_rate, channels) read from the file, or zeros.

    The sidecar's duration is what Suno intended; this is what is on disk. A
    download that stopped halfway still reports its full length in metadata,
    and a truncated file in the training set is a silent quality bug.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "format=duration:stream=sample_rate,channels",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        d = json.loads(r.stdout or "{}")
        st = (d.get("streams") or [{}])[0]
        return (float((d.get("format") or {}).get("duration") or 0),
                int(st.get("sample_rate") or 0), int(st.get("channels") or 0))
    except Exception:
        return (0.0, 0, 0)


def norm_title(t):
    """Titles differing only by take/version markers collapse together."""
    t = unicodedata.normalize("NFKD", (t or "").lower())
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\b(ext|v|ver|version|take|remix|alt)\s*\d*(\.\d+)*\b", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def dedup_key(s):
    """What makes two clips the same song: same title AND same style prompt.

    Title alone is not enough, and getting this wrong is expensive. Suno names
    a clip "Untitled" whenever you never named it, so this library has twelve
    completely different tracks -- different folders, 69 s to 480 s -- sharing
    that title. Keying on title alone silently merged all twelve into one and
    threw away eleven real songs.

    Two takes from ONE generation share the caption exactly, which is precisely
    the case the user wants collapsed. Requiring the caption to match as well
    keeps the rule conservative: it errs toward keeping a near-duplicate rather
    than deleting a distinct song, and a near-duplicate only costs a little
    training time.
    """
    return (norm_title(s["title"]), s["caption"].lower())


def rank(s):
    """Higher is better. The user's rule first: highest play count wins."""
    return (s["play_count"], s["upvote_count"], int(s["liked"]),
            round(s["real_duration"], 1), s["id"])


DUP_SUFFIX = re.compile(r"^(?P<base>.+) \((?P<n>\d+)\)(?P<ext>\.[A-Za-z0-9]+)$")


def sidecar_for(audio):
    """The .txt beside `audio`, across the three naming schemes seen here.

    The third one is the browser's duplicate-download rule, and it is not what
    you would guess. Chrome inserts " (1)" before the LAST extension, and the
    sidecar's name already ends in ".wav" -- so a second copy of
    `Song.wav` + `Song.wav.txt` lands as:

        Song (1).wav          <- " (1)" before ".wav"
        Song.wav (1).txt      <- " (1)" before ".txt", after ".wav"

    Neither of the obvious patterns finds that partner, which silently cost 47
    of 215 tracks -- 22% of the library -- before this existed.
    """
    cands = [audio + ".txt", os.path.splitext(audio)[0] + ".txt"]
    m = DUP_SUFFIX.match(os.path.basename(audio))
    if m:
        root = os.path.dirname(audio)
        cands.append(os.path.join(
            root, f"{m.group('base')}{m.group('ext')} ({m.group('n')}).txt"))
    return next((c for c in cands if os.path.exists(c)), None)


def scan(srcs, drops):
    found = []
    for prio, src in enumerate(srcs):
        if not os.path.isdir(src):
            print(f"!! not a directory: {src}", file=sys.stderr)
            continue
        for root, _, files in os.walk(src):
            for fn in files:
                if not fn.lower().endswith(AUDIO_EXT):
                    continue
                audio = os.path.join(root, fn)
                side = sidecar_for(audio)
                if side is None:
                    drops["no sidecar"] += 1
                    continue
                s = parse_sidecar(side)
                if s is None:
                    drops["unreadable sidecar"] += 1
                    continue
                s["audio_path"] = audio
                s["filename"] = fn
                s["priority"] = prio
                found.append(s)
    return found


LABELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.json")


def load_labels():
    """Measured bpm / keyscale / timesignature, keyed by filename.

    These three used to be written as the literal string "N/A" for every
    sample. They are NOT inert: acestep/training_v2/preprocess_prompt.py:63-73
    formats them into the `# Metas` block of SFT_GEN_PROMPT, which reaches the
    text encoder in training AND at inference
    (core/generation/handler/conditioning_text.py:116,168). Writing "N/A" 184
    times taught the adapter that the trigger word co-occurs with unspecified
    tempo, key and meter -- and the 5Hz planner supplies real values when you
    generate, which is off-distribution.

    Absent or unmeasurable stays "N/A" ON PURPOSE. A track whose tempo we could
    not establish is better described as unspecified than as a number we made
    up: a wrong value binds the trigger word to a fact the audio does not have.
    Run label_audio.py to (re)generate.
    """
    if not os.path.exists(LABELS):
        print(f"  ! {LABELS} not found — bpm/keyscale/timesignature stay N/A.")
        print("    Run: .venv-label\\Scripts\\python.exe label_audio.py")
        return {}
    with open(LABELS, encoding="utf-8") as fh:
        d = json.load(fh)
    out = {}
    for r in d.get("labels", []):
        out[r["filename"]] = {
            "bpm": r.get("bpm") or "N/A",
            "keyscale": r.get("keyscale") or "N/A",
            "timesignature": r.get("timesignature") or "N/A",
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True,
                    help="folder to scan (repeatable; earlier = higher priority)")
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "dataset.json"))
    ap.add_argument("--tag", default="", help="trigger word for the LoRA")
    ap.add_argument("--min-seconds", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--no-probe", action="store_true",
                    help="trust sidecar durations instead of reading each file")
    ap.add_argument("--report", default=os.path.join(ROOT, "train", "dataset_report.txt"))
    a = ap.parse_args()

    drops = Counter()
    found = scan(a.src, drops)
    print(f"scanned {len(a.src)} folder(s): {len(found)} audio+sidecar pairs")

    for n, s in enumerate(found, 1):
        if a.no_probe:
            s["real_duration"], s["sr"], s["ch"] = s["duration"], 0, 0
        else:
            s["real_duration"], s["sr"], s["ch"] = probe(s["audio_path"])
            if n % 50 == 0:
                print(f"  probed {n}/{len(found)}")

    # exact repeats: same clip id
    by_id, no_id = {}, []
    for s in found:
        if not s["id"]:
            no_id.append(s)
            continue
        prev = by_id.get(s["id"])
        if prev is None or s["priority"] < prev["priority"]:
            if prev is not None:
                drops["repeat (same clip id)"] += 1
            by_id[s["id"]] = s
        else:
            drops["repeat (same clip id)"] += 1
    uniq = list(by_id.values()) + no_id

    # sibling takes: same song, different clip
    by_title = {}
    for s in uniq:
        k = dedup_key(s)
        if not any(k):
            k = ("__", s["id"])
        prev = by_title.get(k)
        if prev is None:
            by_title[k] = s
        elif rank(s) > rank(prev):
            by_title[k] = s
            drops["sibling take (lower play count)"] += 1
        else:
            drops["sibling take (lower play count)"] += 1
    kept = list(by_title.values())

    final = []
    for s in kept:
        if not s["caption"]:
            drops["no caption"] += 1
            continue
        d = s["real_duration"] or s["duration"]
        if d < a.min_seconds:
            drops[f"shorter than {a.min_seconds:.0f}s"] += 1
            continue
        try:
            if not os.path.getsize(s["audio_path"]):
                drops["empty file"] += 1
                continue
        except OSError:
            drops["unreadable audio"] += 1
            continue
        final.append(s)

    final.sort(key=lambda s: (-s["play_count"], s["title"]))

    labels = load_labels()
    if labels:
        have = sum(1 for s in final if labels.get(s["filename"], {}).get("bpm") != "N/A")
        print(f"  labels: {have}/{len(final)} tracks carry a measured bpm")

    samples = [{
        "filename": s["filename"],
        "audio_path": s["audio_path"],
        "caption": s["caption"],
        "lyrics": s["lyrics"] or "[Instrumental]",
        "duration": round(s["real_duration"] or s["duration"], 2),
        # Measured by label_audio.py; "N/A" only where it could not be
        # established. See load_labels() for why that distinction matters.
        "bpm": labels.get(s["filename"], {}).get("bpm", "N/A"),
        "timesignature": labels.get(s["filename"], {}).get("timesignature", "N/A"),
        "keyscale": labels.get(s["filename"], {}).get("keyscale", "N/A"),
        # Written per sample as well as dataset-wide. Preprocess only fills
        # this in as a FALLBACK for samples that lack it, so setting it here
        # makes the trigger word a property of the data rather than of one
        # code path continuing to behave.
        "custom_tag": a.tag,
        # provenance; the trainer ignores underscore keys
        "_id": s["id"], "_title": s["title"], "_play_count": s["play_count"],
        "_sr": s["sr"], "_ch": s["ch"], "_model": s["model"],
        "_instrumental": s["instrumental"],
    } for s in final]

    doc = {"metadata": {"custom_tag": a.tag, "tag_position": "prepend",
                        "genre_ratio": 0},
           "samples": samples}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    total = sum(s["duration"] for s in samples)
    trained = sum(min(s["duration"], a.max_seconds) for s in samples)
    over = [s for s in samples if s["duration"] > a.max_seconds]
    n = max(len(samples), 1)
    lines = [
        "sources          : " + "; ".join(a.src),
        f"pairs found      : {len(found)}",
        f"kept             : {len(samples)}",
        f"total audio      : {total / 3600:.2f} h (mean {total / n:.0f}s)",
        f"trained audio    : {trained / 3600:.2f} h "
        f"(after the {a.max_seconds:.0f}s cap)",
        f"over the cap     : {len(over)}",
        f"instrumental     : {sum(1 for s in samples if s['_instrumental'])}",
        f"trigger word     : {a.tag or '(none)'}",
        "",
        "dropped:",
    ] + [f"  {v:5d}  {k}" for k, v in drops.most_common()] + [
        "",
        "sample rates: " + ", ".join(f"{k}Hz x{v}" for k, v in
                                     Counter(s["_sr"] for s in samples).most_common()),
        "channels    : " + ", ".join(f"{k}ch x{v}" for k, v in
                                     Counter(s["_ch"] for s in samples).most_common()),
        "models      : " + ", ".join(f"{k} x{v}" for k, v in
                                     Counter(s["_model"] for s in samples).most_common()),
        "",
        "top 10 by play count:",
    ] + [f"  {s['_play_count']:4d}  {s['_title'][:52]:52s} {s['duration']:6.1f}s"
         for s in samples[:10]]

    report = "\n".join(lines)
    with open(a.report, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print()
    print(report)
    print()
    print(f"wrote {a.out}  and  {a.report}")


if __name__ == "__main__":
    main()
