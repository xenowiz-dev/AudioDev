"""A human-readable name for a track, derived from what we already recorded.

Both generators name their output for the machine, not the listener: ACE-Step
emits a UUID and MiniMax a timestamp, so the library is a wall of hex. The
sidecar already holds the lyrics and the prompt, which is everything a person
would use to name the thing themselves -- this module just applies that.

Heuristic and pure stdlib on purpose. It runs in the studio venv, which has no
numpy and no torch, and it runs once per row on every library refresh; calling
a model to name a file would put a GPU load behind a folder listing and make
the same track render under a different name each time. Deterministic instead:
same record, same title, forever, so nobody has to store it.

Stable is NOT unique. The title is a function of the record, so the four takes
you rendered off one prompt all derive the same title and the same slug -- two
tracks in the folder right now are both "Indie Folk Guitar". A caller that
writes a file per title (art, playlists, exports) MUST disambiguate; use the
audio filename, which is unique by construction, as the key.

Nothing here writes. `derive()` is a view of the record, not a new field in it.
"""

import os
import re
import unicodedata
from datetime import datetime

# 56, not 48: a whole sung line is the best title a track can have, and the
# common shape of one ("Every Coin I Count Falls Through My Shaking Hands" is
# 49) landed just over a 48-char cap and got its last word amputated.
MAX_LEN = 56

# Structure tags, whole-line or leading. Models emit far more than [verse] and
# [chorus] -- [pre-chorus], [verse 2], (Bridge) -- so match the shape, not a
# list of names.
TAG_LINE = re.compile(r"^\s*[\[\(<][^\]\)>]*[\]\)>]\s*$")
TAG_LEAD = re.compile(r"^\s*[\[\(<][^\]\)>]*[\]\)>]\s*")
BARE_TAG = re.compile(
    r"^\s*(verse|chorus|pre[- ]?chorus|bridge|intro|outro|hook|refrain|solo|"
    r"instrumental|break|drop|interlude|coda)\b[\s:0-9x\-\.]*$", re.I)
CHORUS_TAG = re.compile(r"(chorus|hook|refrain)", re.I)
# [pre-chorus] contains "chorus" but is the run-up, not the payoff -- its lines
# are the ones that end mid-thought ("and I don't know").
PRE_TAG = re.compile(r"pre[-\s]?chorus", re.I)

# Vocalise, not words: a line made only of these carries no title.
FILLER = {
    "la", "na", "da", "ba", "ta", "pa", "sha", "doo", "dum", "bop", "boom",
    "oh", "ooh", "oooh", "ah", "aah", "aha", "ay", "aye", "eh", "uh", "huh",
    "hm", "hmm", "mm", "mmm", "mmmm", "yeah", "yea", "hey", "ho", "hoo",
    "whoa", "woah", "wo", "nah", "yo", "ooo", "woo", "hoooo", "ahh", "ohh",
}

# Function words: cheap proxy for "distinctive". A line that is mostly these
# is grammar, not a hook.
STOP = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "as", "is", "are", "was", "were", "be",
    "been", "am", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "there", "here", "not", "no", "do", "does",
    "did", "will", "would", "can", "could", "just", "all", "up", "out",
}

# Lowercased mid-title, sleeve style: short words only. The 4-letter cutoff is
# the AP rule rather than Chicago's "every preposition", because "Shouting into
# the Void" looks like a typo on a track listing and "Shouting Into" does not.
SMALL = {
    "a", "an", "the", "and", "or", "nor", "but", "of", "in", "on", "at", "to",
    "for", "by", "as", "off", "per", "via",
}

# Words a title should not end on -- strip them when a phrase is cut to length,
# so a trimmed prompt reads as a name and not as an interrupted sentence.
TRAIL = {"into", "onto", "over", "with", "from", "about", "than", "that",
         "when", "while", "upon", "through", "like", "and", "but", "who",
         "whom", "whose", "which", "what", "where", "why", "how", "if"}

# Conjunctions that join a line to the one before it. A lyric opening with one
# is the middle of a sentence.
LEADIN = {"and", "but", "so", "then", "yet", "or", "nor", "because", "cause",
          "cos", "coz", "til", "till", "though", "although"}

# What a title must not be left hanging on once it has been cut -- by word
# count in `_phrase` or by character count in `_fit`.
DANGLING = TRAIL | SMALL | STOP

GENRES = {
    "pop", "rock", "folk", "jazz", "blues", "soul", "funk", "disco", "house",
    "techno", "trance", "ambient", "drone", "metal", "punk", "indie",
    "country", "gospel", "reggae", "dub", "ska", "hiphop", "rap", "grime",
    "classical", "orchestral", "cinematic", "electronic", "edm", "synthwave",
    "vaporwave", "lofi", "shoegaze", "grunge", "americana", "bluegrass",
    "swing", "bossa", "samba", "salsa", "tango", "latin", "afrobeat", "garage",
    "dubstep", "breakbeat", "jungle", "chiptune", "industrial", "choral",
    "opera", "waltz", "march", "ballad", "anthem", "downtempo", "trip",
    "psychedelic", "prog", "emo", "hardcore", "surf", "motown", "wop",
    "gaze", "wave", "core", "step", "beat", "groove", "hymn", "lullaby",
    "requiem", "nocturne", "serenade", "chanson", "klezmer", "celtic",
    "flamenco", "blues-rock", "post-rock", "math-rock", "art-rock",
}

MOODS = {
    "upbeat", "warm", "intimate", "dreamy", "dark", "moody", "melancholy",
    "melancholic", "mellow", "driving", "gentle", "soft", "bright", "hazy",
    "lush", "sparse", "epic", "triumphant", "wistful", "nostalgic", "brooding",
    "frantic", "calm", "serene", "tense", "playful", "sultry", "gritty", "raw",
    "ethereal", "haunting", "haunted", "joyful", "sad", "happy", "energetic",
    "chill", "smooth", "aggressive", "tender", "anthemic", "atmospheric",
    "glacial", "breezy", "sunny", "stormy", "restless", "yearning", "fierce",
    "hypnotic", "shimmering", "glittering", "velvet", "neon", "midnight",
    "golden", "quiet", "loud", "slow", "fast", "heavy", "light", "sombre",
    "somber", "jaunty", "swaggering", "romantic", "bittersweet", "defiant",
}

INSTRUMENTS = {
    "guitar", "guitars", "piano", "drums", "drum", "bass", "synth", "synths",
    "strings", "violin", "cello", "viola", "horns", "trumpet", "sax",
    "saxophone", "flute", "organ", "rhodes", "wurlitzer", "harp", "banjo",
    "mandolin", "ukulele", "accordion", "harmonica", "choir", "pads", "keys",
    "percussion", "congas", "bongos", "marimba", "vibraphone", "xylophone",
    "clarinet", "oboe", "bassoon", "tuba", "trombone", "sitar", "koto",
    "kalimba", "theremin", "claps", "snare", "kick", "hats", "cymbals",
    "tambourine", "glockenspiel", "celesta", "dulcimer", "fiddle", "lute",
    "harpsichord", "moog", "mellotron", "arpeggio", "arp",
}

# Studio talk. True of the recording, never of the song.
PRODUCTION = {
    "production", "produced", "producer", "mix", "mixed", "mixing", "master",
    "mastered", "mastering", "recording", "recorded", "stereo", "mono",
    "compressed", "compression", "limiter", "eq", "bitrate", "khz", "hz",
    "bpm", "tempo", "quality", "fidelity", "loudness", "lufs", "db", "sample",
    "samples", "mp3", "wav", "flac", "codec", "render", "rendered", "stems",
    "kbps", "kbit", "bitdepth", "dither", "normalized", "peak",
}

# Left over when a segment is only a measurement.
UNITS = {"bpm", "hz", "khz", "kbps", "db", "s", "sec", "secs", "seconds",
         "min", "mins", "bars", "beats", "st", "cents"}

# Sentences in MiniMax's "Basic Attributes" that describe the grid, not the
# music. Everything left over is the genre.
TECHNICAL = re.compile(r"\b(bpm|key|scale|tempo|time signature|beats|meter|"
                       r"major|minor|hz|duration|length)\b", re.I)

# Reserved device stems -- a slug is only filename-safe if it dodges these.
RESERVED = ({"con", "prn", "aux", "nul"}
            | {f"com{i}" for i in range(1, 10)}
            | {f"lpt{i}" for i in range(1, 10)})

# strftime's %b is locale-dependent: under a French LC_TIME the same record
# renders "Août", which is neither ASCII nor the same title twice.
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Control characters reach us through the sidecar (JSON encodes them happily)
# and have no business in a name that may become a filename or a playlist row.
CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def derive(rec, fallback_name=""):
    """A display title for one sidecar record. Never empty.

    `rec` may be `{}` -- files predating the library, ACE-Step's second take,
    and CLI output all arrive with no record at all, which is why the caller
    can hand us the filename as a last resort.
    """
    if not isinstance(rec, dict):
        rec = {}
    # An explicit title -- typed by the user or accepted from the LLM -- always
    # wins. Everything below is inference; this is a decision.
    fixed = str(rec.get("title") or "").strip()
    if fixed:
        return _polish(fixed)
    # Each stage returns a finished title or None, so this is a real fall-
    # THROUGH rather than a fixed priority: lyrics that are nothing but
    # "[chorus] / la la la" still get a shot at the prompt. Short-circuiting
    # matters -- this runs once per row on every library refresh.
    return _polish(_from_lyrics(rec.get("lyrics"))
                   or _from_prompt(rec.get("prompt"))
                   or _from_name(fallback_name, rec)
                   or "Untitled")


# Separator punctuation that is meaningful BETWEEN words and meaningless at an
# edge. A genre field of "Electric Blues / Blues Rock." truncates to
# "Electric Blues /", and the orphaned slash reads as a rendering bug.
_EDGE = " \t/|&,;:-–—·.…"


def _polish(s):
    """Last pass over whatever any branch produced.

    Applied once here rather than in each extractor: every branch can leave a
    dangling separator, and one guard at the exit cannot be forgotten by the
    next one added.
    """
    s = " ".join(str(s or "").split()).strip(_EDGE)
    # Collapse a separator left stranded by an interior cut ("Blues /  Rock").
    s = re.sub(r"\s+([/|&])\s*$", "", s).strip(_EDGE)
    return s or "Untitled"


def slugify(title):
    """Filename-safe form of a title: lowercase, ASCII, hyphen-separated.

    Stable, but no more unique than the title it came from -- see the module
    docstring before using one as a filename.
    """
    s = str(title or "")
    # Fold the apostrophe out first ("don't" -> dont, not don-t), covering the
    # curly one the models actually emit.
    s = re.sub(r"['‘’ʼ´`]", "", s)
    # Dashes and middots separate words, but none of them has an ASCII
    # decomposition, so the fold below would DELETE rather than replace them
    # and weld "Wolves—Alone" into "wolvesalone". Spell them as the separator
    # they already are, first.
    s = re.sub(r"[‐-―⁃−·•‧・]", "-", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    if not s:
        return "untitled"
    # Windows resolves these stems to devices no matter the extension or the
    # case -- `con.png` is not a file, it is the console.
    return f"{s}-track" if s in RESERVED else s


# --- lyrics ----------------------------------------------------------------

def _from_lyrics(text):
    if not isinstance(text, str) or not text.strip():
        return None

    lines = []
    after_chorus = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if TAG_LINE.match(line) or BARE_TAG.match(line):
            after_chorus = bool(CHORUS_TAG.search(line)
                                and not PRE_TAG.search(line))
            continue
        line = TAG_LEAD.sub("", line).strip()   # "[verse] Morning light..."
        if not line:
            continue
        toks = _words(line)
        if not toks or all(t in FILLER for t in toks):
            continue
        # "run run run run" -- a real line, no information in it.
        if len(toks) >= 3 and len(set(toks)) <= max(1, len(toks) // 3):
            continue
        lines.append((line, toks, after_chorus))

    if not lines:
        return None

    # A line that comes back is the hook, wherever it sits.
    seen = {}
    for line, _, _ in lines:
        k = line.lower().rstrip(" .,!?")
        seen[k] = seen.get(k, 0) + 1

    best, best_score = None, float("-inf")
    for line, toks, in_chorus in lines:
        score = _lyric_score(line, toks, in_chorus,
                             seen[line.lower().rstrip(" .,!?")])
        if score > best_score:        # strict: first line wins a tie
            best, best_score = line, score
    return _finish(best)


def _lyric_score(line, toks, in_chorus, repeats):
    n = len(toks)
    if 3 <= n <= 6:
        score = 1.0
    elif n in (2, 7):
        score = 0.8
    elif n in (1, 8):
        score = 0.6
    elif n <= 10:
        score = 0.4
    else:
        score = 0.15
    if in_chorus:
        score += 0.25
    if repeats > 1:
        score += 0.35
    score += 0.2 * (sum(1 for t in toks if t not in STOP) / n)
    # A clause, not a title: it either runs on past its end or picks up from
    # the line before it. The trailing word is only ever a demotion, never a
    # trim -- "Tell Me Why" ends on a TRAIL word and is still a title, so this
    # picks a better line where one exists and quotes the lyric where none does.
    if line.rstrip().endswith((",", "-", "—")) or toks[-1] in (LEADIN | TRAIL):
        score -= 0.3
    if toks[0] in LEADIN:
        score -= 0.25
    return score


# --- prompt ----------------------------------------------------------------

def _from_prompt(text):
    if not isinstance(text, str) or not text.strip():
        return None
    secs = _sections(text)
    return (_from_sectioned(secs) if _is_sectioned(secs)
            else None) or _from_keywords(text)


def _sections(text):
    """`Label: value` lines, first occurrence wins."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z][A-Za-z &/'-]{2,40}?)\s*:\s*(.+)$", line)
        if m:
            out.setdefault(m.group(1).strip().lower(), m.group(2).strip())
    return out


def _is_sectioned(secs):
    return any(w in k for k in secs
               for w in ("attribute", "sonic", "vocal", "arrangement",
                         "primary", "secondary", "instrument"))


def _from_sectioned(secs):
    """MiniMax's caption format: the music is buried in labelled prose."""
    genre = mood = inst = None

    for k, v in secs.items():
        # First section wins, as in `_sections`: a caption that grew a second
        # "... Attributes:" line must not have it overwrite the real one.
        if "attribute" in k and genre is None:
            # "bpm is 96. key is C, and scale is major. Acoustic Pop."
            for s in v.split("."):
                s = s.strip(" ,;")
                if s and not TECHNICAL.search(s) and not _is_noise(s):
                    genre = _phrase(s, 4)
                    break
        elif ("sonic" in k or "style" in k or "timbre" in k) and not mood:
            for s in re.split(r"[,;.]", v):
                w = _words(s)
                if w and w[0] in MOODS:
                    mood = w[0]
                    break
        elif ("primary" in k or "secondary" in k or "instrument" in k) \
                and not inst:
            inst = next((t for t in _words(v) if t in INSTRUMENTS), None)

    return _compose(mood, genre, inst)


def _from_keywords(text):
    """ACE-Step's flat prompt: comma-separated tags, some of them studio talk."""
    segs = [s.strip(" .;") for s in re.split(r"[,;\n]+", text)]
    segs = [s for s in segs if s and not _is_noise(s)]
    if not segs:
        return None

    genre = next((s for s in segs if _genre_hit(s)), None)
    inst_seg = next((s for s in segs
                     if any(t in INSTRUMENTS for t in _words(s))), None)
    inst = next((t for t in _words(inst_seg or "") if t in INSTRUMENTS), None)
    mood = next((t for s in segs for t in _words(s) if t in MOODS), None)

    if genre:
        return _compose(mood, _phrase(genre, 5), inst)
    if inst_seg:
        # No genre named, so the instrument phrase carries the title: "Jangly
        # Electric Guitar" says more than the bare noun it was picked from.
        return _compose(mood, _phrase(inst_seg, 4), inst)

    # No vocabulary hit worth a name -- a prose prompt. Its opening words are
    # usually the idea, and the head of a phrase beats a bag of adjectives.
    head = _phrase(segs[0], 5)
    if len(head.split()) >= 2:
        return _finish(head)
    # Single-word tags all the way down (a bag of moods). Two is a title;
    # all of them would be the prompt again.
    return _finish(" ".join(_phrase(s, 2) for s in segs[:2]))


def _is_noise(seg):
    """Studio talk or a bare number. Neither belongs in a title.

    Digits alone are not disqualifying -- "1980s synthwave" and "8-bit" name
    the music. It is the segment with nothing BUT a number in it ("110 bpm",
    "44100 hz") that has to go.
    """
    toks = _words(seg)
    return (not toks or any(t in PRODUCTION for t in toks)
            or not any(t not in UNITS for t in toks))


def _genre_hit(seg):
    """Genres split across words more often than not.

    `_words` breaks "lo-fi hip hop" into lo/fi/hip/hop, none of which is a
    genre on its own -- so adjacent pairs are rejoined and looked up too,
    which is what turns "lofi" and "hiphop" back into hits.
    """
    toks = _words(seg)
    if any(t in GENRES for t in toks):
        return True
    return any(a + b in GENRES for a, b in zip(toks, toks[1:]))


def _compose(mood, genre, inst):
    """At most three ideas. Everything else in the prompt is deliberately lost.

    The failure mode this guards against is the keyword dump -- a prompt with
    nine tags becoming a nine-word "title" that is just the prompt again.
    """
    parts = [p for p in (mood, genre) if p]
    if not parts and inst:
        parts = [inst]
    if not parts:
        return None
    # Only reach for the instrument when the title is still too thin to be a
    # name; "Indie Folk" alone says less than "Indie Folk Guitar".
    so_far = " ".join(parts)
    if inst and len(so_far.split()) <= 2 and inst not in so_far.lower():
        parts.append(inst)

    seen, out = set(), []
    for w in " ".join(parts).split():
        # "upbeat" arrives twice when it is both the mood and the head of the
        # genre segment ("upbeat indie pop"); keep the first, drop the echo.
        k = w.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(w)
    return _finish(" ".join(out[:5]))


# --- filename --------------------------------------------------------------

UUIDISH = re.compile(r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-?[0-9a-fA-F]*$")
STAMPED = re.compile(r"^(\d{8})-(\d{6})(?:[_-](.+))?$")


def _from_name(name, rec):
    """Honest rather than pretty: this track told us nothing about itself."""
    stem = os.path.splitext(os.path.basename(str(name or "")))[0].strip()
    if not stem:
        stem = os.path.splitext(str(rec.get("audio") or ""))[0].strip()

    m = STAMPED.match(stem)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            tag = _titlecase(re.sub(r"[_-]+", " ", m.group(3) or "Take"))
            return _finish(f"{tag} {_stamp(dt)}")
        except ValueError:
            pass
    if UUIDISH.match(stem):
        # The full UUID is unreadable and unmemorable; its head is enough to
        # tell two rows apart, which is all the library needs.
        return f"Track {stem[:8].lower()}"
    # Not `return _finish(...)`: a stem of pure punctuation ("___.wav") titles
    # to nothing, and that must fall through to the date rather than end the
    # search and land the row on "Untitled".
    named = _finish(_titlecase(re.sub(r"[_\-\.]+", " ", stem))) if stem else None
    if named:
        return named

    created = rec.get("created")
    if isinstance(created, str):
        try:
            return _finish(f"Take {_stamp(datetime.fromisoformat(created))}")
        except ValueError:
            pass
    return None


def _stamp(dt):
    """`Aug 13 15:14` -- the same string on every machine. See MONTHS."""
    return (f"{MONTHS[dt.month - 1]} {dt.day:02d} "
            f"{dt.hour:02d}:{dt.minute:02d}")


# --- shared ----------------------------------------------------------------

def _words(s):
    return re.findall(r"[A-Za-z][A-Za-z']*", str(s).lower())


def _phrase(s, n):
    """First `n` words of a phrase, cut where a title can stand to end."""
    w = str(s).split()
    if w and w[0].lower() in ("a", "an"):
        w = w[1:]                  # an indefinite article buys nothing here
    w = w[:n]
    return " ".join(_undangle(w))


def _finish(s):
    s = _titlecase(_fit(str(s or "").strip(" \t\"'.,;:-")))
    return s or None


def _undangle(w):
    """Drop the words a title cannot be left ending on. Never returns empty."""
    while len(w) > 1 and w[-1].lower().strip(".,;:") in DANGLING:
        w.pop()
    return w


def _fit(s, n=MAX_LEN):
    """Hard cap at a word boundary, with no ellipsis -- a title is a name, and
    a name ending in "..." reads as a bug rather than as brevity."""
    s = " ".join(CTRL.sub(" ", s).split())
    if len(s) <= n:
        return s
    cut = s[:n]
    sp = cut.rfind(" ")
    if sp >= n // 2:
        cut = cut[:sp]
    # The cut lands wherever the 48th character falls, which is regularly mid-
    # phrase; left alone it yields "...Burning in The", and title-case then
    # capitalises the stray preposition and makes it look deliberate.
    return " ".join(_undangle(cut.rstrip(" ,;:-").split()))


def _titlecase(s):
    # A wholly upper-case line is shouting, not an acronym; recase it before
    # the per-word rules, which cannot tell "WE" from "DJ" on their own.
    if s == s.upper() and any(c.isalpha() for c in s):
        s = s.lower()
    words = s.split()
    out = []
    for i, w in enumerate(words):
        low = w.lower().strip(".,;:!?\"'")   # "On," must still count as small
        if 0 < i < len(words) - 1 and low in SMALL:
            out.append(w.lower())
        else:
            out.append("-".join(_cap(p) for p in w.split("-")))
    return " ".join(out)


def _cap(w):
    # str.title() would give "Don'T" and "Rock'N'Roll"; only the first letter
    # of a word is ours to touch.
    if w.isupper() and len(w) > 3:
        w = w.lower()                      # SHOUTED LYRIC -> Shouted Lyric
    for i, ch in enumerate(w):
        if ch.isdigit():
            return w        # "1980s", "808" -- the tail of a number is not a
        if ch.isalpha():    # word start, and "1980S" reads as a typo
            return w[:i] + ch.upper() + w[i + 1:]
    return w
