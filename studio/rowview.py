"""Suno-style LIST view of the library, rendered to one HTML string.

Presentation only. This module builds markup and stylesheet text; it wires
nothing, reads no files but a cover's existence, and never touches gradio. The
integrator drops the string into a `gr.HTML` and attaches ONE delegated click
listener -- which is why every clickable element here carries `data-idx` and
`data-act` and nothing here carries an `onclick`. Per-element handlers would
die the first time the list repaints (the container's innerHTML is replaced,
the container itself is not), and inventing the event mechanism here would
lock the integrator out of choosing it.

Why the gallery is not enough: `gr.Gallery` gives one image and one caption
per track, so the take number, the length, the model and the style all have to
be crammed into a two-line label, and there is nowhere to put a per-row button.
A row is the shape the data already has.

Runs in the STUDIO venv: pure stdlib plus `titles`. No numpy, no Pillow, and
not even `art` -- which is also why covers are *looked up* here and never
rendered. `art.ensure_cover` shells out to the media venv when the PNG is
missing, and a list that rendered art inline would turn one library refresh
into forty subprocesses. A missing cover draws a placeholder instead.

Costs, measured on this library: ~2.5 KB and ~55 us per row, most of it the
six inline icons. A repaint re-sends the whole list, so 400 tracks is a 1 MB
string over localhost -- fine here, and the reason `row_html` is a single join
rather than accumulation. If the folder ever gets big enough for that to
matter, page the entry list before rendering; do not start diffing HTML.

Everything that reaches the markup from a record goes through `html.escape`.
Prompts and lyrics are model output -- they hold quotes, angle brackets, em
dashes and newlines as a matter of course -- and one unescaped `<` would take
the rest of the list down with it.

Three things the integrator has to supply, none of which belong here:
  * `allowed_paths=[r"B:\\AudioDev\\Music\\studio"]` on `launch`, or every
    cover 403s and the list renders as a column of placeholders.
  * one delegated listener on the container, reading `data-act` and `data-idx`
    off `e.target.closest('[data-act]')`.
  * a keydown handler if playing from the keyboard should work: the thumbnail
    is `role="button" tabindex="0"` because it behaves like one, but only a
    real <button> gets Enter/Space translated into a click for free.
"""

import html
import os
import re
import urllib.parse

import titles

# --------------------------------------------------------------------- icons

# Inline SVG per button rather than one <symbol> sprite with <use href="#id">:
# gradio serves the app under a `<base href>`, and a fragment-only reference in
# <use> is resolved against the base URL, which silently blanks every icon on
# the browsers that follow the spec here. Inline paths cannot be broken that
# way. `currentColor` lets the CSS tint the active states.
ICONS = {
    "play": '<path d="M8 5v14l11-7z"/>',
    # One thumb, not two: a thumbs-down is this glyph point-mirrored, so the
    # dislike button rotates it 180deg in CSS and the pair cannot drift apart
    # in weight or size the way two separately-drawn icons would.
    "thumb": ('<path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57'
              '.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 '
              '8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05'
              'c.09-.23.14-.47.14-.73v-2z"/>'),
    "edit": ('<path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 '
             '7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 '
             '3.75 3.75 1.83-1.83z"/>'),
    "cover": ('<path d="M17.65 6.35A8 8 0 1 0 19.73 14h-2.08A6 6 0 1 1 12 6c'
              '1.66 0 3.14.69 4.22 1.78L13 11h7V4z"/>'),
    "more": ('<circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/>'
             '<circle cx="19" cy="12" r="2"/>'),
    "note": '<path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3z"/>',
}

# Which glyph each action draws with.
ICON_FOR = {"like": "thumb", "dislike": "thumb"}

# Order is the strip's reading order, left to right, and the tooltip text.
# `edit` sits between the ratings and the destructive-ish end of the strip so a
# mis-click on a dense row lands on something harmless.
ACTIONS = (
    ("like", "Like"),
    ("dislike", "Dislike"),
    ("edit", "Edit title and style"),
    ("cover", "Cover / reuse this prompt"),
    ("more", "More actions"),
)

# Style-line noise. A segment that is only a measurement describes the grid,
# not the music -- "110 bpm" and "44100 hz" are the two that show up most.
_TECHNICAL = re.compile(r"\b(bpm|khz|hz|kbps|lufs|db|key|scale|tempo|"
                        r"time signature|meter|duration)\b", re.I)
_NUMERIC = re.compile(r"^[\W\d]*$")
# MiniMax captions are `Label: prose` lines; the label is scaffolding.
_LABEL = re.compile(r"^[A-Za-z][A-Za-z &/'-]{2,40}:\s*")

STYLE_SEGMENTS = 3      # at most: more than this and the line is the prompt
STYLE_MAX = 96          # characters, hard cap before the ellipsis

# --------------------------------------------------------------------- covers

COVER_SUBDIR = "covers"


def file_url(path):
    """The URL gradio serves a local file from, given `allowed_paths`.

    Verified form: `/gradio_api/file=B%3A/AudioDev/Music/studio/covers/x.png`
    -- forward slashes, url-quoted. A `data:` URI would inline half a megabyte
    of PNG per row into the component value on every repaint.
    """
    return "/gradio_api/file=" + urllib.parse.quote(
        str(path).replace("\\", "/"))


def default_cover_url(audio_path):
    """`covers\\<stem>.png` beside the track, or None if it is not there yet.

    Mirrors `art.cover_path` for files that live in the library folder, without
    importing `art` -- keeping this module stdlib-only is what lets it load in
    the studio venv, where numpy does not exist. An integrator that also shows
    off-library derivatives (whose covers carry a hash suffix) should pass its
    own callable built on `art.cover_path`.
    """
    audio_path = str(audio_path or "")
    if not audio_path:
        return None
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    p = os.path.join(os.path.dirname(audio_path), COVER_SUBDIR, stem + ".png")
    # Existence check, never a render: covers are made in the media venv by a
    # background job, and a library refresh must not wait on one.
    return file_url(p) if os.path.exists(p) else None


def _cover_src(entry, cover_url):
    """The <img> src for one row, or None for the placeholder.

    A caller's callable is allowed to be wrong: a cover lookup that raises must
    cost that row its artwork, not take down the whole list.
    """
    try:
        u = (cover_url or default_cover_url)(entry.get("path"))
    except Exception:
        return None
    u = str(u or "").strip()
    return u or None


# ---------------------------------------------------------------- record bits

def _title(entry):
    """The row's display name.

    `rec['title']` is the user's override -- nothing writes it today, so the
    derived title is the live path and has to stay cheap; `titles.derive` is
    pure stdlib and deterministic for exactly that reason.
    """
    rec = entry.get("rec") or {}
    t = str(rec.get("title") or "").strip()
    return t or titles.derive(rec, entry.get("name") or "")


def _duration(rec):
    """`m:ss`, or None. Present only where a sidecar recorded the length."""
    s = rec.get("seconds")
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        return None
    s = int(round(s))
    if s <= 0:
        return None
    return f"{s // 60}:{s % 60:02d}"


def _rating(entry):
    """-1 dislike / 0 neutral / +1 like, from whatever the record holds.

    `rating` does not exist in any sidecar yet -- this is the tri-state Suno's
    thumbs imply, and the representation is still the integrator's to pick, so
    accept the plausible ones and keep the mapping in one place. The `favorite`
    star we DO store is a like, unless a rating explicitly contradicts it.

    Takes an entry, not a record, because of `shared_rec`: `group_takes` lends
    the sibling take the donor's record so its prompt can be shown, and a
    verdict must NOT ride along on that loan. Rating one take of a run is the
    whole reason there are two rows, and the rest of the app agrees -- it asks
    `pls.is_favorite(path)`, per file, never through the inherited record.
    """
    if entry.get("shared_rec"):
        return 0
    rec = entry.get("rec") or {}
    r = rec.get("rating")
    val = 0
    if isinstance(r, bool):
        val = 1 if r else 0
    elif isinstance(r, (int, float)):
        val = 1 if r > 0 else (-1 if r < 0 else 0)
    elif isinstance(r, str):
        s = r.strip().lower()
        if s in ("up", "like", "liked", "+1", "1"):
            val = 1
        elif s in ("down", "dislike", "disliked", "-1"):
            val = -1
    if val == 0 and rec.get("favorite"):
        val = 1
    return val


def _count(v, default=1):
    """A small positive count out of a record field, or `default`.

    `take`/`takes` arrive from `group_takes` as ints, but this module renders
    whatever the caller assembled, and anything that has been through a JSON
    round-trip in some other tool hands them back as strings. `"2" > 1` is a
    TypeError in py3, and it would be raised while building the list -- taking
    the WHOLE library down over one malformed row, which is exactly what
    `_cover_src` refuses to do for a bad cover.

    Coercing here is also what keeps "everything from a record is escaped"
    true: these two are the only record-side values interpolated into the
    markup as numbers rather than through `html.escape`, so a non-number has
    to be turned into one BEFORE it reaches an attribute, not merely compared
    safely. `or default` preserves the old falsy-means-one behaviour.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return default
    try:
        return int(float(v)) or default
    except (TypeError, ValueError, OverflowError):
        return default


def _style_line(entry):
    """The genre/style line: `rec['style']`, else a digest of the prompt.

    Both prompt dialects have to survive this. ACE-Step's is a flat list of
    tags, so the first few segments ARE the style. MiniMax's is a labelled
    caption whose first line is "Basic Attributes: bpm is 96. key is C." --
    printing that raw gives a row whose style reads as a tempo, so labels and
    measurements are dropped before anything is kept.

    Returns `(text, is_dim)`; dim is for the honest non-answers, which must not
    look like a style someone chose.
    """
    rec = entry.get("rec") or {}
    style = " ".join(str(rec.get("style") or "").split())
    if style:
        return _clip(style, STYLE_MAX), False

    segs = []
    for seg in _prompt_segments(rec.get("prompt")):
        segs.append(seg)
        if len(segs) >= STYLE_SEGMENTS:
            break
    if segs:
        return _clip(" · ".join(segs), STYLE_MAX), False

    if not (entry.get("recorded") or entry.get("shared_rec")):
        # Same wording the table view uses, so the two views agree about what
        # an unrecorded file is.
        return "no record — found on disk", True
    return "no prompt recorded", True


def _prompt_segments(prompt):
    """The style-worthy pieces of a prompt, best first.

    Three cuts, in this order, and the order is the whole trick:

    1. If ANY line is `Label: value`, only the values are considered. That is
       MiniMax's caption, whose bare heading lines ("Global Metadata",
       "Arrangement") are scaffolding and would otherwise be the first thing
       the row said about the music.
    2. Sentences, because MiniMax buries the genre behind the grid --
       "bpm is 96. key is C, and scale is major. Acoustic Pop." Only the last
       sentence there is the style.
    3. Commas, which is ACE-Step's entire dialect.

    Measurements are dropped at the *piece* level, never earlier: "110 bpm" is
    one tag among five in an ACE-Step prompt, and rejecting the sentence that
    contains it would throw away the four real ones with it.
    """
    lines = [ln.strip() for ln in str(prompt or "").splitlines() if ln.strip()]
    labelled = [ln[m.end():] for ln in lines for m in [_LABEL.match(ln)] if m]
    for chunk in (labelled or lines):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk):
            for piece in re.split(r"[,;]", sentence):
                piece = " ".join(piece.split()).strip(" .·-")
                if (not piece or _NUMERIC.match(piece)
                        or _TECHNICAL.search(piece)):
                    continue
                yield piece


def _clip(s, n):
    """Cut at a word boundary; an ellipsis is the only honest way to end."""
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n].rstrip()
    sp = cut.rfind(" ")
    if sp >= n // 2:
        cut = cut[:sp]
    return cut.rstrip(" ,;:·-") + "…"


# --------------------------------------------------------------------- markup

def _icon(name, cls="rv-i"):
    # viewBox 0 0 24 24 for every path above, so one wrapper fits all of them.
    return (f'<svg class="{cls}" viewBox="0 0 24 24" aria-hidden="true">'
            f'{ICONS[ICON_FOR.get(name, name)]}</svg>')


def _btn(act, label, idx, state=""):
    e = html.escape
    on = ' aria-pressed="true"' if "is-on" in state else ""
    return (f'<button type="button" class="rv-btn rv-btn--{act}{state}" '
            f'data-idx="{idx}" data-act="{act}" title="{e(label)}" '
            f'aria-label="{e(label)}"{on}>{_icon(act)}</button>')


def row_html(entries, playing_index=None, cover_url=None):
    """The whole list as one HTML string.

    `entries` are `library.entries()` dicts. `data-idx` is the position in THIS
    list, not in the library -- the caller filters and sorts before rendering,
    so the integrator must hand the same list back to its click handler (the
    app already keeps it in a `gr.State` for exactly this reason).

    `cover_url` is `path -> url or None`; see `default_cover_url`.
    """
    entries = list(entries or [])
    if not entries:
        return _empty()

    try:
        playing = int(playing_index)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is `int(float('inf'))` -- the same class of junk as the
        # None and the "" the other two already absorb. A player index that
        # went non-finite must grey the highlight, not blank the library.
        playing = -1

    e = html.escape
    # One list, one join. Repeated `s += row` is quadratic, and this list is
    # sized by the folder -- it grows every time anything is generated.
    parts = ['<div class="rv-list">']
    for i, entry in enumerate(entries):
        entry = entry or {}
        rec = entry.get("rec") or {}
        title = _title(entry)
        style, dim = _style_line(entry)
        dur = _duration(rec)
        rating = _rating(entry)
        takes = _count(entry.get("takes"))
        take = _count(entry.get("take"))
        src = _cover_src(entry, cover_url)

        row_cls = "rv-row rv-row--playing" if i == playing else "rv-row"
        art = (f'<img class="rv-art" src="{e(src)}" alt="" loading="lazy" '
               f'decoding="async">' if src
               else f'<div class="rv-art rv-art--none">{_icon("note")}</div>')
        badge = (f'<span class="rv-dur">{e(dur)}</span>' if dur else "")

        meta = [f'<span class="rv-title" title="{e(title)}">{e(title)}</span>']
        if takes > 1:
            # ACE-Step writes two clips per run; without this the pair looks
            # like one track duplicated.
            meta.append(f'<span class="rv-take" title="take {take} of {takes}'
                        f'">{take}/{takes}</span>')
        model = str(rec.get("model") or "").strip()
        if model:
            meta.append(f'<span class="rv-model">{e(model)}</span>')

        acts = [_btn("like", "Like", i, " is-on" if rating > 0 else ""),
                _btn("dislike", "Dislike", i, " is-on" if rating < 0 else "")]
        acts += [_btn(a, lab, i) for a, lab in ACTIONS[2:]]

        parts.append(
            f'<div class="{row_cls}" data-idx="{i}">'
            f'<div class="rv-thumb" data-idx="{i}" data-act="play" '
            f'role="button" tabindex="0" title="Play {e(title)}">'
            f'{art}<span class="rv-play">{_icon("play")}</span>{badge}</div>'
            f'<div class="rv-meta"><div class="rv-line">{"".join(meta)}</div>'
            f'<div class="rv-style{" rv-style--dim" if dim else ""}" '
            f'title="{e(style)}">{e(style)}</div></div>'
            f'<div class="rv-acts">{"".join(acts)}</div>'
            f'</div>')
    parts.append("</div>")
    return "".join(parts)


def _empty():
    # Inside `.rv-list` too -- every selector in CSS is scoped under it, so an
    # empty state outside the wrapper would render unstyled.
    return ('<div class="rv-list rv-list--empty">'
            f'<div class="rv-empty">{_icon("note", "rv-i rv-empty-i")}'
            '<div class="rv-empty-t">Nothing here yet</div>'
            '<div class="rv-empty-s">Generate a track, or clear the search '
            'and filters above.</div></div></div>')


# ------------------------------------------------------------------------ css

# Everything is scoped under `.rv-list` so this can be concatenated into the
# app's own stylesheet (or handed to gr.HTML's css_template) without leaking.
# Colours are the repo's: the cover PNGs are rendered on #141416 with amber and
# teal, and a row that framed them in anything else would make the art float.
# The bridge from the rendered rows back to Python.
#
# Two things here are load-bearing and were established by browser-testing this
# against a live gradio 6.24 server, not by reading docs:
#
#  1. The custom event name must appear QUOTED in this string. gr.HTML's
#     __getattr__ regex-scans js_on_load to decide whether `component.act(...)`
#     is a legal listener; without the literal 'act' below you get
#     AttributeError at wiring time.
#  2. The listener is DELEGATED onto `element`, never bound per button. A
#     Python-side value update replaces innerHTML but not `element`, so
#     per-button handlers would silently die on the first repaint while
#     delegation keeps working.
#
# There is deliberately no contenteditable here: an unsolicited repaint (a
# refresh, a finished generation) while a field had focus would discard whatever
# was half-typed. Editing lives in real gradio textboxes, which cannot be
# clobbered that way.
JS = """
element.addEventListener('click', (e) => {
  const t = e.target.closest('[data-act]');
  if (!t) return;
  e.preventDefault();
  trigger('act', { idx: +t.dataset.idx, act: t.dataset.act });
});
element.addEventListener('keydown', (e) => {
  const t = e.target.closest('[data-act]');
  if (!t || (e.key !== 'Enter' && e.key !== ' ')) return;
  e.preventDefault();
  trigger('act', { idx: +t.dataset.idx, act: t.dataset.act });
});
"""

CSS = """
.rv-list {
  --rv-bg: #141416;
  --rv-row: #1a1a1d;
  --rv-row-hover: #202024;
  --rv-line: #2a2a2f;
  --rv-text: #ececf1;
  --rv-dim: #8a8a94;
  --rv-amber: #f5c542;
  --rv-teal: #4fd1c5;
  background: var(--rv-bg);
  color: var(--rv-text);
  border-radius: 12px;
  padding: 4px;
  font-size: 13px;
  line-height: 1.35;
  /* The list is the scroll surface; the page must not grow with the folder. */
  max-height: 68vh;
  overflow-y: auto;
  overscroll-behavior: contain;
}

/* rows ------------------------------------------------------------------ */
.rv-list .rv-row {
  display: flex; align-items: center; gap: 10px;
  padding: 5px 8px 5px 6px;
  border-radius: 10px;
  border-left: 3px solid transparent;   /* reserved: the playing marker */
  transition: background .12s ease;
}
.rv-list .rv-row + .rv-row { box-shadow: 0 -1px 0 var(--rv-line); }
.rv-list .rv-row:hover { background: var(--rv-row-hover); }
.rv-list .rv-row--playing {
  background: rgba(245, 197, 66, .10);
  border-left-color: var(--rv-amber);
}
.rv-list .rv-row--playing:hover { background: rgba(245, 197, 66, .14); }
.rv-list .rv-row--playing .rv-title { color: var(--rv-amber); }

/* thumbnail ------------------------------------------------------------- */
.rv-list .rv-thumb {
  position: relative; flex: none; width: 60px; height: 60px;
  border-radius: 8px; overflow: hidden; cursor: pointer;
  background: #0f0f11;
}
.rv-list .rv-art {
  width: 100%; height: 100%; display: block;
  aspect-ratio: 1 / 1; object-fit: cover;
}
.rv-list .rv-art--none {
  display: flex; align-items: center; justify-content: center;
  color: #3d3d45;
  background: linear-gradient(140deg, #17171a, #202027);
}
.rv-list .rv-art--none .rv-i { width: 22px; height: 22px; }
.rv-list .rv-play {
  position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  color: #fff; background: rgba(0, 0, 0, .42);
  opacity: 0; transition: opacity .12s ease;
}
.rv-list .rv-play .rv-i {
  width: 26px; height: 26px; filter: drop-shadow(0 1px 4px #000);
}
.rv-list .rv-thumb:hover .rv-play,
.rv-list .rv-thumb:focus-visible .rv-play,
.rv-list .rv-row--playing .rv-play { opacity: 1; }
.rv-list .rv-row--playing .rv-play { color: var(--rv-amber); }
.rv-list .rv-dur {
  position: absolute; right: 3px; bottom: 3px;
  padding: 0 4px; border-radius: 3px;
  background: rgba(0, 0, 0, .72); color: #e8e8ee;
  font-size: 10.5px; font-variant-numeric: tabular-nums;
}

/* text ------------------------------------------------------------------ */
.rv-list .rv-meta { flex: 1; min-width: 0; }   /* min-width:0 or flex won't
                                                  let the title ellipsize */
.rv-list .rv-line { display: flex; align-items: baseline; gap: 6px;
                    min-width: 0; }
.rv-list .rv-title {
  font-weight: 600; font-size: 13.5px; letter-spacing: .1px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rv-list .rv-take, .rv-list .rv-model {
  flex: none; font-size: 10px; padding: 1px 5px; border-radius: 999px;
  border: 1px solid var(--rv-line); color: var(--rv-dim);
  text-transform: lowercase;
}
.rv-list .rv-take {
  color: var(--rv-teal); border-color: rgba(79, 209, 197, .35);
  font-variant-numeric: tabular-nums;
}
.rv-list .rv-style {
  margin-top: 2px; font-size: 11.5px; color: var(--rv-teal); opacity: .85;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rv-list .rv-style--dim { color: var(--rv-dim); font-style: italic;
                          opacity: .7; }

/* action strip ---------------------------------------------------------- */
.rv-list .rv-acts { flex: none; display: flex; align-items: center; gap: 2px; }
.rv-list .rv-btn {
  display: flex; align-items: center; justify-content: center;
  width: 30px; height: 30px; padding: 0;
  border: 0; border-radius: 50%;
  background: transparent; color: var(--rv-dim);
  cursor: pointer; -webkit-appearance: none; appearance: none;
}
.rv-list .rv-btn .rv-i { width: 16px; height: 16px; }
.rv-list .rv-btn:hover { background: rgba(255, 255, 255, .08);
                         color: var(--rv-text); }
.rv-list .rv-btn:focus-visible { outline: 2px solid var(--rv-amber);
                                 outline-offset: -2px; }
.rv-list .rv-btn--like.is-on { color: var(--rv-amber);
                               background: rgba(245, 197, 66, .14); }
.rv-list .rv-btn--dislike .rv-i { transform: rotate(180deg); }  /* see ICONS */
.rv-list .rv-btn--dislike.is-on { color: #e2706b;
                                  background: rgba(226, 112, 107, .14); }

/* icons are tinted by the button, never by their own fill ---------------- */
.rv-list .rv-i { fill: currentColor; display: block; }

/* empty state ----------------------------------------------------------- */
.rv-list--empty { display: flex; align-items: center; justify-content: center;
                  min-height: 180px; }
.rv-list .rv-empty { text-align: center; color: var(--rv-dim); padding: 24px; }
.rv-list .rv-empty-i { width: 34px; height: 34px; margin: 0 auto 10px;
                       color: #33333b; }
.rv-list .rv-empty-t { color: var(--rv-text); font-weight: 600;
                       font-size: 14px; }
.rv-list .rv-empty-s { margin-top: 4px; font-size: 12px; }

/* narrow phones: the app is reachable over the LAN, so this list is read on
   one. Drop the model chip before dropping anything clickable. */
@media (max-width: 560px) {
  .rv-list .rv-model { display: none; }
  .rv-list .rv-thumb { width: 52px; height: 52px; }
  .rv-list .rv-btn { width: 28px; height: 28px; }
  .rv-list .rv-acts { gap: 0; }
}
"""
