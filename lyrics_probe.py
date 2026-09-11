"""Find lyrics that already exist for an audio file: embedded tags or a sidecar.

Two places to look, in this order:

  1. EMBEDDED  ID3 USLT/SYLT/TXXX:LYRICS, Vorbis LYRICS/UNSYNCEDLYRICS,
     MP4 (c)lyr -- and RIFF LIST/INFO for plain WAV, which `mutagen` does NOT
     surface (it handles WAVE-with-ID3 only). That gap is why this imports
     `read_riff_info` from the forensics toolkit rather than trusting mutagen:
     it once hid a literal "made with suno" comment 164 bytes into a file.

  2. SIDECAR   .lrc / .txt / .srt / .vtt beside the audio, or in a lyrics\\
     subfolder. Matching is ranked, never silent: an exact stem beats a prefix
     beats token overlap, and the winner is always named, because a folder can
     easily hold `excerpt_lyrics.txt` next to the file you actually want.

Timestamps are stripped, structure tags are NOT. A greedy `\\[.*?\\]` would eat
`[verse]` and `[chorus]`, which are exactly what ACE-Step conditions on.

Prints one JSON object to stdout.

Usage:
    python lyrics_probe.py "B:\\AudioDev\\Music\\Paycheck to Paycheck.wav"
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "watermark"))

# .json covers the studio's own sidecar records (see studio\library.py), so
# covering one of our own generations finds the lyrics it was made with.
SIDECAR_EXT = (".lrc", ".txt", ".srt", ".vtt", ".json")
MAX_SIDECAR_BYTES = 100_000

# Words that carry no identity, so `paycheck_lyrics.txt` still matches
# `Paycheck to Paycheck.wav` on the token that matters.
NOISE_TOKENS = {"lyrics", "lyric", "text", "words", "transcript", "final",
                "copy", "the", "a", "of", "to", "and", "official"}

# LRC timing tags and LRC metadata keys -- and nothing else.
LRC_TIME = re.compile(r"\[\s*\d{1,3}\s*:\s*\d{1,2}(?:[.:]\d{1,3})?\s*\]")
LRC_META = re.compile(r"^\[(?:ar|ti|al|by|offset|re|ve|length|au|id|encoding)"
                      r"\s*:[^\]]*\]\s*$", re.I)
SRT_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*-->")
WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(name):
    return {t for t in WORD_RE.findall(name.lower()) if t not in NOISE_TOKENS}


def clean_text(raw, ext):
    """Strip timing, keep structure tags like [verse]."""
    out = []
    for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = line.strip()
        if ext == ".lrc":
            if LRC_META.match(s):
                continue
            s = LRC_TIME.sub("", s).strip()
        elif ext in (".srt", ".vtt"):
            if (s.upper().startswith("WEBVTT") or SRT_TIME.match(s)
                    or s.isdigit() or s.upper().startswith("NOTE")):
                continue
        out.append(s)
    # collapse runs of blank lines, drop leading/trailing blanks
    text, blank = [], 0
    for s in out:
        if s:
            blank = 0
            text.append(s)
        else:
            blank += 1
            if blank == 1 and text:
                text.append("")
    return "\n".join(text).strip()


def looks_like_lyrics(text):
    """Cheap guard so a stray README.txt is not offered as lyrics."""
    if not text or len(text) < 20:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    # Lyrics are many short lines; prose is few long ones.
    avg = sum(len(l) for l in lines) / len(lines)
    return avg < 120


def embedded(path):
    """Every embedded lyric field we can find, best first."""
    hits = []
    try:
        import mutagen
        f = mutagen.File(path)
    except Exception:
        f = None

    if f is not None and getattr(f, "tags", None):
        tags = f.tags
        try:
            items = list(tags.items())
        except Exception:
            items = []
        for key, val in items:
            k = str(key).upper()
            if not (k.startswith("USLT") or k.startswith("SYLT")
                    or "LYRIC" in k or k == "\xa9LYR" or k == "UNSYNCEDLYRICS"):
                continue
            if hasattr(val, "text"):
                val = val.text
            if isinstance(val, (list, tuple)):
                val = "\n".join(str(v) for v in val)
            val = str(val).strip()
            if val:
                hits.append({"source": "embedded", "field": str(key),
                             "text": val})

    # WAV LIST/INFO: mutagen returns nothing for plain PCM, so read the
    # container directly. See ai_audio_forensics.read_riff_info.
    if path.lower().endswith(".wav"):
        try:
            from ai_audio_forensics import read_riff_info
            info = read_riff_info(path) or {}
            for key, val in info.items():
                if not isinstance(val, str):
                    continue
                if key.upper() in ("ICMT", "ILYR", "COMMENT", "LYRICS") \
                        and looks_like_lyrics(val):
                    hits.append({"source": "embedded",
                                 "field": f"RIFF {key}", "text": val.strip()})
        except Exception:
            pass
    return hits


def sidecars(path):
    """Ranked sidecar candidates. Never guesses silently -- each carries why."""
    folder = os.path.dirname(os.path.abspath(path)) or "."
    stem = os.path.splitext(os.path.basename(path))[0]
    stem_l = stem.lower()
    want = tokens(stem)

    search = []
    for d in (folder, os.path.join(folder, "lyrics")):
        if os.path.isdir(d):
            for n in os.listdir(d):
                if os.path.splitext(n)[1].lower() in SIDECAR_EXT:
                    search.append(os.path.join(d, n))

    out = []
    for p in search:
        try:
            if os.path.getsize(p) > MAX_SIDECAR_BYTES:
                continue
        except OSError:
            continue
        s = os.path.splitext(os.path.basename(p))[0]
        s_l = s.lower()
        got = tokens(s)
        if s_l == stem_l:
            score, why = 100, "exact filename match"
        elif s_l.startswith(stem_l) or stem_l.startswith(s_l):
            score, why = 80, "filename prefix match"
        elif want and got and got <= want:
            score = 60 + int(20 * len(got) / max(len(want), 1))
            why = f"name tokens {sorted(got)} all appear in the audio filename"
        elif want & got:
            shared = want & got
            score = 40 + int(20 * len(shared) / max(len(want | got), 1))
            why = f"shares {sorted(shared)} with the audio filename"
        else:
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()
        except OSError:
            continue
        ext = os.path.splitext(p)[1].lower()
        if ext == ".json":
            # Our own sidecar: take the lyrics field, ignore everything else.
            try:
                raw = (json.loads(raw) or {}).get("lyrics") or ""
            except Exception:
                continue
            if not raw.strip():
                continue
            why += " (studio sidecar)"
        text = clean_text(raw, ext)
        if not looks_like_lyrics(text):
            continue
        out.append({"source": "sidecar", "field": os.path.basename(p),
                    "path": p, "text": text, "score": score, "why": why})
    out.sort(key=lambda d: -d["score"])
    return out


def probe(path):
    if not os.path.exists(path):
        return {"ok": False, "error": f"no such file: {path}"}
    emb = embedded(path)
    side = sidecars(path)
    best = (emb + side)[0] if (emb or side) else None
    return {
        "ok": True,
        "file": os.path.abspath(path),
        "embedded": [{k: v for k, v in h.items() if k != "text"} for h in emb],
        "sidecars": [{k: v for k, v in h.items() if k != "text"}
                     for h in side],
        "best": best,
        "found": best is not None,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    r = probe(sys.argv[1])
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
