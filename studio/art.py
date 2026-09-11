"""Cover art rendered from the track's own spectrogram -- no second model.

Every image-generating alternative here means another checkpoint, another venv,
and a cover that has nothing to do with the audio beyond the prompt text. This
draws the track itself: time wraps once around the circle, log-frequency runs
from the centre outward, and the ring outside the disc is the RMS envelope --
a circular waveform. Two tracks look alike only if they *sound* alike.

Runs in the media venv (numpy / soundfile / Pillow). The studio venv has none
of those, so this module imports them lazily and `ensure_cover` falls back to
re-running this file as a subprocess under the media interpreter. That means
`import art` is safe from either side: `cover_path` and the cheap hit in
`ensure_cover` are pure stdlib.

Covers live in a SUBDIRECTORY of the output folder. `library.entries()` globs
the top level only, so a PNG next to the tracks would be invisible to it -- but
a WAV named like a cover would not, and the day someone loosens that glob the
covers must not turn into phantom library rows.

Deterministic by construction: the same audio produces the same bytes. The only
stochastic element, the anti-banding grain, is seeded from the rendered field.

Usage (media venv):
    python art.py --input track.flac [--out cover.png] [--force]
Prints the resulting path as the last stdout line.
"""

import hashlib
import os
import subprocess
import sys
import time
import uuid

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

OUTDIR = os.path.join(ROOT, "Music", "studio")
COVERDIR = os.path.join(OUTDIR, "covers")
MEDIA_PY = os.path.join(ROOT, "watermark", ".venv", "Scripts", "python.exe")
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SIZE = 512          # square, the shape every player expects
SS = 2              # supersample factor; LANCZOS down = free anti-aliasing
N_COLS = 720        # angular resolution (time)
N_RAD = 400         # radial resolution (frequency bands)
N_FFT = 2048
MAX_SECONDS = 480.0     # analysis cap; bounds both the read and the FFT count

F_LO, F_HI = 45.0, 15000.0   # radial band, log-spaced

# The viewer's -95 dB floor exists to expose noise floors. Art wants contrast,
# not evidence, so the ramp is packed into the top 46 dB and the rest is night.
DB_RANGE = 46.0
WHITEN = 0.72       # per-band contrast vs. absolute level; see _field()
GAMMA = 0.82        # < 1: lifts the mid-tones, or the disc reads as empty

# Below ~15 columns of smoothing every drum hit fires off as its own radial
# spoke and the cover comes out looking like a starburst chart.
SMOOTH_T, SMOOTH_R = 25, 9      # columns (12 deg of arc), radial bands

# An annulus, not a disc: the log-frequency bands get a wide radial run, and
# the dark centre gives the thing a record label to sit around.
R_IN, R_OUT = 0.285, 0.885
R_RING = 0.905          # circular-waveform rim outside the bloom
RING_MIN = 0.009        # keeps the rim unbroken through the quiet passages
RING_MAX = 0.030        # ...and this much thicker at full level
HALO = 0.16             # outward bleed; keeps the corners off pure black
CORE = 0.40             # inward bleed, as a fraction of R_IN

TEAL = (0x4f, 0xd1, 0xc5)
AMBER = (0xf5, 0xc5, 0x42)

# Magma-like, but it lands on the repo amber instead of matplotlib's, and the
# shadows are pulled cool so the warm highlights have something to sit against.
# Stop 0.00 is the repo background -- silence has to melt into the page, not
# stop at a visible disc edge.
RAMP = ((0.00, "#141416"), (0.09, "#15201f"), (0.20, "#1e1b33"),
        (0.34, "#432a63"), (0.50, "#8e2f6e"), (0.66, "#d2494e"),
        (0.80, "#ef8b34"), (0.92, "#f5c542"), (1.00, "#fdf6dc"))

# Bound lazily by _stack(). Module-level `import numpy` would make this file
# unimportable from the studio venv, which is exactly where ensure_cover lives.
np = sf = Image = None


# ---------------------------------------------------------------- public API

def cover_path(audio_path):
    """Where this track's cover belongs. Pure function -- creates nothing."""
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    parent = os.path.dirname(os.path.abspath(audio_path))
    if _same_dir(parent, OUTDIR):
        return os.path.join(COVERDIR, stem + ".png")
    # Off-library input (an upscaled derivative, a test fixture): keep the
    # folder flat but stop `upscaled\x.wav` from overwriting `x.flac`'s cover.
    tag = hashlib.sha1(os.path.normcase(parent).encode()).hexdigest()[:8]
    return os.path.join(COVERDIR, f"{stem}-{tag}.png")


def ensure_cover(audio_path, force=False):
    """Path to this track's cover, rendering it if needed. None on failure.

    Cheap on the hit path -- one `exists` -- because the library calls this per
    row. Never raises: a missing cover is a cosmetic problem, and it must not
    take down a table refresh.
    """
    try:
        out = cover_path(audio_path)
        if not force and os.path.exists(out):
            return out
        if not os.path.exists(audio_path):
            return None
        if _have_stack():
            return render(audio_path, out)
        return _subprocess_render(audio_path, out, force)
    except Exception:
        return None


def render(audio_path, out_png=None, size=SIZE):
    """Render in-process. Needs the media stack. Returns the cover path."""
    _stack()
    out_png = out_png or cover_path(audio_path)
    mono, sr = _read(audio_path)
    field = _field(_bands(mono, sr), mono)
    rgb = _compose(field, _envelope(mono), size)
    _save(rgb, out_png, size)
    return out_png


# ------------------------------------------------------------- venv plumbing

def _stack():
    """Import the media stack once, into module globals."""
    global np, sf, Image
    if np is None:
        import numpy
        import soundfile
        from PIL import Image as _Image
        np, sf, Image = numpy, soundfile, _Image


def _have_stack():
    try:
        _stack()
        return True
    except ImportError:
        return False


def _subprocess_render(audio_path, out_png, force):
    if not os.path.exists(MEDIA_PY):
        return None
    cmd = [MEDIA_PY, os.path.abspath(__file__), "--input", audio_path,
           "--out", out_png]
    if force:
        cmd.append("--force")
    # The child's stdout defaults to the ANSI codepage (cp1252 here), and we
    # decode it as utf-8. A track named with an em dash then came back with a
    # U+FFFD in place of it, and one named with a character cp1252 cannot hold
    # at all (a musical note, CJK) killed the child with UnicodeEncodeError on
    # its final print -- after the PNG was safely written. Pin both ends.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300, creationflags=NOWIN,
                       env=env)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise RuntimeError((r.stderr or r.stdout or "")[-1500:])
    # We chose out_png and just confirmed it exists, so return it rather than
    # the echo of it. specview parses the last line because it needs the JSON
    # the child computed; here the round trip can only lose information.
    return out_png


def _same_dir(a, b):
    return os.path.normcase(os.path.normpath(a)) == \
           os.path.normcase(os.path.normpath(b))


# ------------------------------------------------------------------ analysis

def _read(path):
    """Mono float32 and its sample rate, capped at MAX_SECONDS.

    One `SoundFile` in a `with`, rather than `sf.info()` followed by
    `sf.read()`. `sf.info` opens the file to read its header and leaves closing
    to garbage collection, and inside a long-lived server that is a leaked
    handle: cover art is rendered for every new track, so every new track stayed
    open. The symptom was Move to trash failing with "Could not move that file"
    on anything recently generated -- Windows will not rename an open file --
    and it cleared the moment the server was stopped, which is what identified
    it. One open also halves the header reads.
    """
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        want = int(min(f.frames, MAX_SECONDS * sr))
        x = f.read(frames=max(want, 0), dtype="float32", always_2d=True)
    m = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if m.size < N_FFT:
        m = np.pad(m, (0, N_FFT - m.size))    # a 40 ms file still gets a frame
    return np.ascontiguousarray(m, dtype=np.float32), sr


def _stft(m, n_cols):
    """Magnitude spectrogram averaged down to exactly `n_cols` columns.

    A six-minute track has ~15k analysis frames to spend on 720 columns;
    point-sampling every 21st would throw away most of the mix and alias what
    is left. Each column is the mean of `sub` evenly-spaced frames instead,
    and `sub` is capped so the FFT count stays bounded however long the input.
    """
    avail = 1 + max(0, m.size - N_FFT) // (N_FFT // 4)
    sub = int(min(max(avail // n_cols, 1), 16))
    total = n_cols * sub
    starts = np.linspace(0, max(0, m.size - N_FFT), total).astype(np.int64)
    win = np.hanning(N_FFT).astype(np.float32)
    off = np.arange(N_FFT, dtype=np.int64)

    out = np.empty((n_cols, N_FFT // 2 + 1), np.float32)
    chunk = max(sub, (512 // sub) * sub)   # multiple of `sub`: whole columns
    for i in range(0, total, chunk):
        st = starts[i:i + chunk]
        F = np.abs(np.fft.rfft(m[st[:, None] + off] * win, axis=1))
        k = len(st) // sub
        out[i // sub: i // sub + k] = F.reshape(k, sub, -1).mean(axis=1)
    return out.T                            # (bins, cols)


def _bands(m, sr):
    """Log-frequency band energies in dB, shape (N_RAD, N_COLS).

    Bands are integrated between fractional bin edges rather than sampled at
    them: up at 12 kHz one band spans dozens of FFT bins, and picking one of
    them makes the outer rim sparkle with noise instead of showing texture.
    """
    S = _stft(m, N_COLS)
    # float64 for the running sum: the bass bands are narrower than one FFT bin
    # and are read as the difference of two much larger prefix sums, which in
    # float32 cancels down to noise -- and to negatives that NaN the log.
    power = S.astype(np.float64) ** 2
    nbins = power.shape[0]

    edges = np.geomspace(F_LO, F_HI, N_RAD + 1) / (sr / 2.0) * (nbins - 1)
    edges = np.clip(edges, 0.0, nbins - 1.0)
    c = np.zeros((nbins + 1, power.shape[1]), np.float64)
    np.cumsum(power, axis=0, out=c[1:])

    lo, hi = _lerp_rows(c, edges[:-1]), _lerp_rows(c, edges[1:])
    width = np.maximum(edges[1:] - edges[:-1], 1e-6)[:, None]
    band = np.maximum(hi - lo, 0.0) / width
    return (10.0 * np.log10(band + 1e-12)).astype(np.float32)


def _lerp_rows(a, idx):
    """Linear interpolation along axis 0 at fractional row indices."""
    i0 = np.clip(np.floor(idx).astype(np.int64), 0, a.shape[0] - 2)
    f = (idx - i0)[:, None]
    return a[i0] * (1.0 - f) + a[i0 + 1] * f


def _field(db, mono):
    """dB bands -> 0..1 brightness, smoothed. Silence stays black."""
    if float(np.abs(mono).max()) < 1e-6:
        return np.zeros_like(db)

    # Pure absolute level buries everything above ~4 kHz, because that is what
    # music does; pure per-band contrast flattens a track into uniform mush.
    # The blend keeps the loudness shape while letting the rim show detail.
    ref = np.percentile(db, 60.0, axis=1, keepdims=True)
    mix = (1.0 - WHITEN) * (db - db.max()) + WHITEN * (db - ref)

    # Levels off the histogram, not off the peak. Anchoring the top of the ramp
    # to db.max() hands the entire dynamic range to one bright pixel and every
    # cover comes out near-black; percentiles put a known fraction of the disc
    # in the light no matter how the track was mixed.
    lo, hi = (float(v) for v in np.percentile(mix, [22.0, 99.6]))
    v = (mix - lo) / max(hi - lo, DB_RANGE * 0.25)
    v = np.clip(v, 0.0, 1.0).astype(np.float32)
    v = _box(v, SMOOTH_T, axis=1, wrap=True)   # time wraps: the circle loops
    v = _box(v, SMOOTH_R, axis=0, wrap=False)
    return np.clip(v, 0.0, 1.0) ** GAMMA


def _envelope(m):
    """Per-column RMS, 0..1 -- the circular waveform."""
    n = (m.size // N_COLS) * N_COLS
    if n < N_COLS:
        return np.zeros(N_COLS, np.float32)
    e = np.sqrt((m[:n].reshape(N_COLS, -1) ** 2).mean(axis=1))
    hi = float(np.percentile(e, 98.0))
    if hi < 1e-9:
        return np.zeros(N_COLS, np.float32)
    # Compressed and smoothed harder than the field: an unsmoothed envelope
    # turns the rim into a gear wheel instead of a breathing line.
    e = np.clip(e / hi, 0.0, 1.0) ** 0.55
    return _box(e[None, :], 17, axis=1, wrap=True)[0]


def _box(a, k, axis, wrap):
    """Centred moving average of width k by prefix sum. Deterministic, O(n)."""
    if k < 2:
        return a
    a = np.moveaxis(a, axis, -1)
    n = a.shape[-1]
    k, half = min(k, n), min(k, n) // 2
    if wrap:
        p = np.concatenate([a[..., n - half:], a, a[..., :k - half]], axis=-1)
    else:
        p = np.concatenate([np.repeat(a[..., :1], half, -1), a,
                            np.repeat(a[..., -1:], k - half, -1)], axis=-1)
    c = np.zeros(p.shape[:-1] + (p.shape[-1] + 1,), np.float32)
    np.cumsum(p, axis=-1, out=c[..., 1:])
    return np.moveaxis((c[..., k:k + n] - c[..., :n]) / k, -1, axis)


# ------------------------------------------------------------------- drawing

def _lut():
    """256-entry RGB ramp, float 0..255."""
    stops = np.array([s for s, _ in RAMP], np.float32)
    cols = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)]
                     for _, h in RAMP], np.float32)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    return np.stack([np.interp(x, stops, cols[:, i]) for i in range(3)],
                    axis=1)


def _compose(field, env, size):
    """Warp the field into a disc and lay the graphic elements over it."""
    w = size * SS
    ax = (np.arange(w, dtype=np.float32) + 0.5) / w * 2.0 - 1.0
    x, y = ax[None, :], ax[:, None]
    r = np.sqrt(x * x + y * y)
    # atan2(x, -y): 0 at twelve o'clock, increasing clockwise, so the track
    # reads the way a clock does rather than the way a maths textbook does.
    t = (np.arctan2(x, -y) / (2.0 * np.pi)) % 1.0

    u = np.clip((r - R_IN) / (R_OUT - R_IN), 0.0, 1.0)
    v = _sample(field, u, t)

    # The bloom bleeds off both edges rather than stopping at them -- a hard
    # boundary is what makes a polar plot look like a plot. Inward it decays
    # into the dark core, outward into the corners.
    v *= np.where(r < R_IN, np.exp(-(R_IN - r) / (CORE * R_IN)), 1.0)
    v *= np.where(r > R_OUT, np.exp(-(r - R_OUT) / HALO), 1.0)
    v *= 1.0 - 0.30 * np.clip((r - 0.62) / 0.8, 0.0, 1.0) ** 2   # vignette

    rgb = _lut()[np.clip(v * 255.0, 0, 255).astype(np.uint8)]

    px = 2.0 / w
    e = np.interp((t * N_COLS) % N_COLS, np.arange(N_COLS + 1),
                  np.append(env, env[0])).astype(np.float32)
    band = _band(r, R_RING, R_RING + RING_MIN + RING_MAX * e, px)
    tint = (np.array(TEAL, np.float32) * (1.0 - e[..., None])
            + np.array(AMBER, np.float32) * e[..., None])
    # One rim element, not two: a separate hairline circle sitting a few pixels
    # inside this one just read as a chart border drawn slightly off-centre.
    # The ring is continuous everywhere (RING_MIN) and states the loudness
    # twice over -- thickness, and teal warming to amber.
    rgb += band[..., None] * tint * 0.45

    # Seeded from the field, not from the clock or the path: a copy of the same
    # audio must render byte-identical. Grain only exists to break the banding
    # a 256-stop ramp shows across the large, nearly flat halo.
    seed = int(hashlib.sha1(np.ascontiguousarray(field).tobytes())
               .hexdigest()[:8], 16)
    rgb += np.random.default_rng(seed).normal(0.0, 1.6, rgb.shape[:2] + (1,))

    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _sample(field, u, t):
    """Bilinear lookup into (rad, col), wrapping in time."""
    fu = u * (field.shape[0] - 1)
    i0 = np.clip(fu.astype(np.int32), 0, field.shape[0] - 2)
    du = (fu - i0).astype(np.float32)

    fc = t * field.shape[1]
    j0 = fc.astype(np.int32) % field.shape[1]
    j1 = (j0 + 1) % field.shape[1]
    dc = (fc - np.floor(fc)).astype(np.float32)

    top = field[i0, j0] * (1 - dc) + field[i0, j1] * dc
    bot = field[i0 + 1, j0] * (1 - dc) + field[i0 + 1, j1] * dc
    return top * (1 - du) + bot * du


def _band(r, lo, hi, px):
    """Soft-edged annulus mask; the ~1 px ramp is the anti-aliasing."""
    return (np.clip((r - lo) / px, 0.0, 1.0) *
            np.clip((hi - r) / px, 0.0, 1.0)).astype(np.float32)


def _save(rgb, out_png, size=SIZE):
    img = Image.fromarray(rgb, "RGB")
    # `size`, not SIZE: _compose drew at size*SS, and comparing against the
    # module constant made every non-default size silently come out 512.
    if img.width != size:
        img = img.resize((size, size), Image.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    # Written aside and renamed: gradio may already be serving this path, and a
    # half-written PNG is a broken image in the UI rather than a stale one.
    # uuid, not just pid: gradio renders rows from threads in ONE process, so
    # two calls for the same track shared a tmp name and lost the race to
    # os.replace with PermissionError -- a row with no cover at all.
    tmp = f"{out_png}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    img.save(tmp, format="PNG", optimize=True)

    # Windows refuses the rename while another writer is replacing the same
    # destination, or while something still has it open -- so the unique tmp
    # above is only half the fix. Renders are deterministic, so whoever wins is
    # writing byte-identical output: retry briefly, and if a cover is sitting
    # there afterwards, that is the right answer and the row gets its image.
    for attempt in range(5):
        try:
            os.replace(tmp, out_png)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.02 * (attempt + 1))
    os.remove(tmp)      # never leave .tmp litter in the covers folder
    # A rival's own rename briefly unlinks the destination, so one miss here is
    # not proof the cover is absent -- look twice before failing the row.
    if not os.path.exists(out_png):
        time.sleep(0.05)
        if not os.path.exists(out_png):
            raise last


def main():
    import argparse
    ap = argparse.ArgumentParser(description="render cover art from audio")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = a.out or cover_path(a.input)
    # ensure_cover's semantics, inlined. Calling ensure_cover here would let a
    # media venv that lost a dependency spawn itself forever.
    if a.force or not os.path.exists(out):
        out = render(a.input, out)
    # Printing the path IS the CLI contract, so it must not be the step that
    # fails: a cp1252 console cannot encode half the punctuation these track
    # names carry. (ensure_cover also forces utf-8 from the parent side.)
    rc = getattr(sys.stdout, "reconfigure", None)
    if rc and (getattr(sys.stdout, "encoding", "") or "").lower() \
            not in ("utf-8", "utf8"):
        rc(encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
