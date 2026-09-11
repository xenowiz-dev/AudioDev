"""AudioDev Studio -- one local UI over the installed music generators.

Not a ComfyUI backend, deliberately. ACE-Step and MiniMax Music 3 need
incompatible Python environments, so nothing can import both; each runs as a
warm subprocess in its own venv (see supervisor.py). ComfyUI would add a third
environment that can import neither, plus a competing claim on the same 10 GB.

VRAM discipline, which is the whole reason this is shaped the way it is:
  * exactly one model is loaded at a time -- switching kills the other process,
    which returns its memory unconditionally
  * both workers park their weights on the CPU between jobs (~0.15 GB idle)
  * Unload frees everything without stopping the UI

Run:  studio.ps1        (or .venv\\Scripts\\python.exe app.py)
"""

import os
import queue
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supervisor import Supervisor, BACKENDS, WorkerDied, gpu_memory
import postprocess as pp
import lyrics as ly
import library as lib
import titles
import playlists as pls
import art
import rowview
import llm_title

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

# Pure-arithmetic VRAM budget for MiniMax's autoregressive stage. Lives with
# the model, but imports cleanly here because the maths needs no torch.
sys.path.insert(0, os.path.join(ROOT, "minimax"))
import ar_cache

OUTDIR = os.path.join(ROOT, "Music", "studio")
UPDIR = os.path.join(OUTDIR, "upscaled")
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SUP = Supervisor()

# Generate and Upscale are separate gradio events, and queue()'s concurrency
# limit applies per event -- so without a shared id the two could run at once
# and put two jobs on the same 10 GB card. Same id = one slot between them.
GPU_SLOT = "gpu"

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

LYRICS = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""

# Amber = the cliff-line accent every figure in this repo already uses; zinc
# keeps the neutrals close to the spectrograms' #141416 so the PNGs don't float
# on a mismatched background.
THEME = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="zinc",
    font=["ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
    font_mono=["ui-monospace", "Cascadia Code", "Consolas", "monospace"],
)

# Force dark: the spectrograms are dark PNGs and look wrong on light chrome.
#
# This is `head=`, NOT `launch(js=...)`. In gradio 6.24 the js= string is
# shipped in the page config but NEVER EXECUTED -- there is no code path that
# runs app-level config js, only per-event `js=` on a dependency. (blocks.py
# still documents it as auto-executing; that docstring is wrong.) Verified: a
# marker function set via js= never ran across two full loads. So the previous
# redirect here was dead code, and with a light system preference the app was
# rendering LIGHT behind dark spectrograms.
#
# Both halves are needed. The class makes THIS load dark; replaceState puts
# ?__theme=dark in the URL so the NEXT load is dark from first paint, without
# navigating (window.location.replace would be a full reload and would wipe
# every field). The observer re-adds the class when gradio's own matchMedia
# handler strips it on a system light/dark flip.
HEAD_DARK = """
<script>
(function () {
  try {
    var u = new URL(window.location.href);
    if (u.searchParams.get('__theme') !== 'dark') {
      u.searchParams.set('__theme', 'dark');
      history.replaceState(null, '', u.href);
    }
  } catch (e) {}
  var b = document.body;
  var dark = function () {
    if (b && !b.classList.contains('dark')) b.classList.add('dark');
  };
  dark();
  if (b) new MutationObserver(dark).observe(
      b, {attributes: true, attributeFilter: ['class']});

  // Only ever one thing playing. Results all land in the bottom bar, but the
  // two FILE PICKERS (cover source, restore input) are gr.Audio too and
  // gradio gives them their own controls -- so this is enforced in the DOM
  // rather than by wiring, which also covers any player added later.
  // Capture phase: 'play' does not bubble.
  // The dock is fixed, so nothing in normal flow knows how tall it is.
  // Publish its height as --dock-h and keep the panes clear of it.
  var sizeDock = function () {
    var d = document.querySelector('.playerdock');
    if (!d) return;
    var h = Math.ceil(d.getBoundingClientRect().height);
    if (h > 0) {
      document.documentElement.style.setProperty('--dock-h', h + 'px');
      document.body.style.paddingBottom = (h + 12) + 'px';
    }
  };
  var poke = function () { requestAnimationFrame(sizeDock); };
  window.addEventListener('resize', poke);
  new MutationObserver(poke).observe(document.documentElement,
      {childList: true, subtree: true});
  poke();

  document.addEventListener('play', function (e) {
    var me = e.target;
    if (!(me instanceof HTMLMediaElement)) return;
    document.querySelectorAll('audio, video').forEach(function (other) {
      if (other !== me && !other.paused) other.pause();
    });
  }, true);
})();
</script>
"""

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* header */
.app-header { display: flex; align-items: center; gap: 14px;
              padding: 4px 2px 0; }
.app-header .glyph { font-size: 26px; line-height: 1; letter-spacing: 1px;
                     color: var(--primary-500, #f5c542);
                     text-shadow: 0 0 18px rgba(245, 197, 66, .35); }
.app-header h1 { margin: 0; font-size: 21px; font-weight: 700;
                 letter-spacing: .2px; }
.app-header .sub { margin: 2px 0 0; opacity: .55; font-size: 12.5px; }

/* status pill (the vram readout) */
.vram-pill p { display: inline-block; padding: 5px 14px;
               border-radius: 999px; font-size: 12.5px; margin: 0;
               background: rgba(127, 127, 127, .10);
               border: 1px solid rgba(127, 127, 127, .22); }

/* logs */
.wrap-log textarea { font-family: var(--font-mono);
                     font-size: 12px; line-height: 1.45; }

/* the spectrogram is a picking surface, so say so with the cursor */
.specview img { cursor: crosshair; }
.specview { background: #141416; border-radius: 10px; overflow: hidden; }

/* phone access */
.phone-qr img { max-width: 240px; margin: 0 auto; border-radius: 12px; }

/* duration fit warning */
.fit-note p { font-size: 12px; margin: 2px 0 0; opacity: .8; }
.fit-note p:has(> :first-child) { opacity: 1; }

/* library */
.lib-grid .grid-wrap { padding: 2px; }
.lib-grid button, .lib-grid .thumbnail-item { border-radius: 10px;
    overflow: hidden; cursor: pointer; }
.lib-grid img { aspect-ratio: 1 / 1; object-fit: cover; }
/* Captions are two lines (title, then take/length/model) and must not be
   clipped to one, or every take in a pair looks identical. */
.lib-grid .caption-label, .lib-grid figcaption {
    white-space: pre-line !important; font-size: 11.5px; line-height: 1.3;
    text-align: left; padding: 6px 8px; opacity: .92; }
/* ---- two panes + a player locked to the viewport ------------------------
   The dock is position:fixed, so it stays put no matter which pane scrolls.
   Each pane owns its own scrollbar and is sized against the viewport, which
   is what stops the whole page scrolling as one long column. */
.gradio-container { max-width: 100% !important; padding: 8px 12px 0 !important; }
.mainrow { align-items: stretch !important; gap: 12px; }
/* flex-wrap:nowrap is LOAD-BEARING. gradio's .column sets flex-wrap:wrap,
   which is harmless while a column is auto-height -- but the moment you give
   it a fixed height, items that do not fit wrap into a NEW COLUMN to the
   right. Measured: the gallery laid out at x=1321 in a 662px pane, and the
   pane's scrollWidth was 1949. */
/* `.gradio-container .column.pane`, not `.pane`: gradio's own scoped rule is
   `.column.svelte-xxxx` (specificity 0,2,0) and is also !important, so a
   single-class selector loses the tie and the wrap silently stays on. */
.gradio-container .column.pane {
        height: calc(100vh - var(--dock-h, 132px) - 96px);
        flex-wrap: nowrap !important;
        overflow-y: auto; overflow-x: hidden; padding-right: 6px; }
/* The other half of nowrap: children default to `flex: 0 1 auto`, so once
   they can no longer wrap they SHRINK to fit the fixed height instead -- which
   squashed the gallery to nothing. Pin their basis so the pane scrolls. */
.gradio-container .column.pane > * { flex-shrink: 0 !important; }
.pane-right { border-left: 1px solid rgba(127,127,127,.16); padding-left: 12px; }
/* a quieter scrollbar than the OS default, on dark */
.pane::-webkit-scrollbar { width: 10px; }
.pane::-webkit-scrollbar-thumb { background: rgba(127,127,127,.28);
                                 border-radius: 6px; }
.pane::-webkit-scrollbar-track { background: transparent; }

.playerdock { position: fixed; left: 0; right: 0; bottom: 0; z-index: 60;
              background: var(--body-background-fill);
              border-top: 1px solid rgba(127,127,127,.22);
              padding: 6px 14px 8px; gap: 0 !important;
              box-shadow: 0 -8px 24px rgba(0,0,0,.35); }
.playerbar { align-items: center; gap: 10px; flex-wrap: nowrap !important; }
.playerbar .np p { margin: 0; font-size: 13.5px; font-weight: 600;
                   line-height: 1.25; }
.np-sub { font-size: 11.5px; font-weight: 400; opacity: .6; }
.np-art img { width: 52px; height: 52px; border-radius: 8px;
              object-fit: cover; }
.np-art { flex: 0 0 52px; }
/* an empty gr.Image still renders its upload placeholder -- hide the slot
   entirely until a track is actually loaded */
.np-art:not(:has(img)) { display: none !important; }
.playerbar .tbtn { min-width: 42px !important; padding: 4px 8px !important;
                   font-size: 15px; }
.player { min-width: 0; }
.player .wrap, .player audio { width: 100%; }

@media (max-width: 900px) {
  /* stack the panes and let the page scroll normally; a fixed dock on a
     phone would eat too much of a short viewport if it also held art */
  .mainrow { flex-direction: column; }
  .pane { height: auto; overflow: visible; }
  .pane-right { border-left: none; padding-left: 0;
                border-top: 1px solid rgba(127,127,127,.16); }
  .np-art { display: none; }
  .gradio-container { padding-bottom: 150px !important; }
}

@media (max-width: 768px) {
  .lib-grid { height: auto !important; }
}

/* small screens: stack is automatic, but the fixed-height spectrogram would
   otherwise letterbox a ~165px image inside 560px of container */
@media (max-width: 768px) {
  .gradio-container { padding: 6px !important; }
  .app-header .sub { display: none; }
  .specview, .specview > div { height: auto !important;
                               min-height: 0 !important; }
  .specview img { width: 100%; height: auto !important; }
  button { min-height: 44px; }   /* touch targets */
}
"""


def vram_text():
    used, total = gpu_memory()
    if used is None:
        return "GPU: unavailable"
    free = total - used
    loaded = SUP.current()
    who = f" · loaded: {BACKENDS[loaded]['label']}" if loaded else " · nothing loaded"
    warn = ""
    if loaded is None and free < 8.0:
        warn = ("  ⚠ under 8 GB free — MiniMax will spill to system memory "
                "and run several times slower")
    return f"GPU: {used:.1f} / {total:.1f} GB used · {free:.1f} GB free{who}{warn}"


def stamp(model):
    return f"{datetime.now():%Y%m%d-%H%M%S}_{model}.wav"


def duration_hint(model, duration):
    """Will this length actually fit in the VRAM that is free right now?

    MiniMax's AR stage grows its KV cache and per-frame hidden states at about
    7.3 MiB per second of music on top of a 7.49 GB floor, so 'does it fit' is a
    question about DURATION, not about the model. Spilling does not fail --
    Windows silently pages to system memory and the run goes ~3x slower and
    keeps degrading -- so it is worth saying before a long wait, not after.
    """
    if model != "minimax":
        return ""
    used, total = gpu_memory()
    if used is None:
        return ""
    free = total - used
    need, fits, head = ar_cache.budget(float(duration or 0), free)
    if fits:
        return (f"needs ~{need:.1f} GB, {free:.1f} GB free — "
                f"fits with {head:.1f} GB spare")
    return (f"⚠ needs ~{need:.1f} GB but only {free:.1f} GB is free "
            f"({-head:.1f} GB short) — this will spill to system memory and "
            f"run several times slower. Close browsers/ComfyUI, or shorten it.")


def _bar(text, frac, width=28):
    """A progress bar drawn in markdown.

    Not `gr.Progress`: this composes with a streaming generator handler without
    depending on another gradio-6 API surface, and it survives in the log.
    """
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return f"**{text}**  \n`{'█' * n}{'░' * (width - n)}` {frac*100:.0f}%"


def generate(model, prompt, lyrics, duration, steps, seed, instrumental,
             src_audio, cover_strength, noise_strength, post_kind,
             is_preview=False):
    # `is_preview` is a Python-only keyword: gradio calls this with 11
    # positional inputs and never sees it, so the endpoint arity is unchanged.
    """Streams log lines to the UI while the worker runs."""
    if not (prompt or "").strip():
        raise gr.Error("A style description is required.")
    if model == "minimax" and not (lyrics or "").strip():
        raise gr.Error("MiniMax Music 3 requires lyrics — it has no "
                       "instrumental mode. Use ACE-Step for instrumentals.")

    params = dict(
        prompt=prompt.strip(),
        lyrics="" if instrumental else (lyrics or "").strip(),
        duration=float(duration), steps=int(steps), seed=int(seed),
        instrumental=bool(instrumental),
        out_name=stamp(model),
    )
    if model == "acestep" and src_audio:
        params.update(src_audio=src_audio,
                      cover_strength=float(cover_strength),
                      noise_strength=float(noise_strength))

    q, box = queue.Queue(), {}

    def work():
        try:
            box["res"] = SUP.generate(
                model, params, on_progress=lambda m: q.put(m))
        except Exception as e:                       # WorkerDied and friends
            box["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    t0 = time.time()
    lines, live, live_stage = [], None, None

    def body():
        return "\n".join(lines + ([live] if live else []))

    while True:
        msg = q.get()
        if msg is None:
            break
        text = msg.get("msg", "")
        ts = f"[{time.time() - t0:5.0f}s] "
        frac, stage = msg.get("frac"), msg.get("stage")
        if frac is None:
            if live:
                lines.append(live)
                live, live_stage = None, None
            lines.append(ts + text)
            yield body(), gr.update(), vram_text(), f"**{text}**"
        else:
            # A percentage line replaces the previous one instead of adding a
            # hundred near-identical rows -- but each stage keeps its last.
            if live and stage != live_stage:
                lines.append(live)
            live, live_stage = ts + text, stage
            yield body(), gr.update(), vram_text(), _bar(text, frac)

    if "err" in box:
        if live:
            lines.append(live)
        lines.append(f"\nFAILED: {box['err']}")
        yield body(), gr.update(), vram_text(), "**failed**"
        raise gr.Error(str(box["err"])[:400])
    if live:
        lines.append(live)
        live = None

    res = box["res"]
    rt = res["elapsed"] / max(res["seconds"], 1e-6)
    lines.append(
        f"\n{os.path.basename(res['path'])}\n"
        f"{res['seconds']:.1f}s @ {res['sampling_rate']} Hz · "
        f"generated in {res['elapsed']:.0f}s ({rt:.1f}x realtime)\n"
        f"peak VRAM {res.get('vram_peak', 0):.2f} GB · "
        f"idle now {res.get('vram_idle', 0):.2f} GB")
    final = res["path"]
    # Sidecar written for the GENERATED file, before any post step: the
    # upscaled derivative lives in upscaled\ and is reachable from `post_kind`.
    # A side effect, not an output -- adding one here would change the arity of
    # every yield in this function and of two endpoints.
    lib.write(res["path"], model=model, prompt=prompt,
              lyrics=None if instrumental else (lyrics or None),
              instrumental=bool(instrumental), duration=float(duration),
              steps=int(steps), seed=int(seed), post_kind=post_kind,
              preview=bool(is_preview),
              cover_src=src_audio if (model == "acestep" and src_audio) else None,
              cover_strength=cover_strength if src_audio else None,
              noise_strength=noise_strength if src_audio else None,
              seconds=res.get("seconds"), elapsed=res.get("elapsed"),
              sampling_rate=res.get("sampling_rate"),
              vram_peak=res.get("vram_peak"))
    # Cover art now, so a new track appears in the library already looking like
    # a track. ~1 s against a generation measured in minutes, and it renders in
    # the media venv, so it cannot disturb the model worker. Best-effort: a
    # missing cover degrades to a tile with no image, never a failed generation.
    try:
        art.ensure_cover(res["path"])
    except Exception:
        pass

    done = (f"**done** — {res['seconds']:.0f}s of audio in "
            f"{res['elapsed']:.0f}s ({rt:.1f}x realtime)")
    yield "\n".join(lines), final, vram_text(), done

    if post_kind and post_kind != "none":
        # Report the bandwidth verdict but proceed regardless: the user asked
        # for this step. Neither generator leaves a codec cliff, so this will
        # usually say there is nothing to restore -- that is worth seeing, not
        # worth overriding.
        lines.append("\n--- post: " + pp.UPSCALERS[post_kind] + " ---")
        lines.append(pp.bandwidth(final))
        yield "\n".join(lines), final, vram_text(), \
            f"**post-processing** — {pp.UPSCALERS[post_kind]}"
        # On failure `final` keeps pointing at the generated file, so a bad
        # post step degrades to "you still have your generation".
        for text, path in stream_upscale(post_kind, final, lines):
            final = path or final
            yield text, final, vram_text(), f"**post-processing**"
        yield "\n".join(lines), final, vram_text(), done


def stream_upscale(kind, src, lines, auto_lowpass=True):
    """Run pp.upscale on a thread, yielding (log_text, out_path_or_None).

    A generator, because pp.upscale blocks for the whole run: appending to a
    list from its progress callback without yielding would freeze the log for
    12 s on Apollo and potentially half an hour for AudioSR on a full track, in
    an app where everything else streams. Only the final yield carries a path;
    on failure it yields the error text and no path, leaving the caller's
    existing file untouched.
    """
    q, box = queue.Queue(), {}

    def work():
        try:
            box["out"] = pp.upscale(kind, src, UPDIR,
                                    on_progress=lambda m: q.put(m),
                                    auto_lowpass=bool(auto_lowpass))
        except Exception as e:
            box["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append("  " + msg)
        yield "\n".join(lines[-400:]), None

    if "err" in box:
        lines.append(f"\nFAILED: {type(box['err']).__name__}: {box['err']}")
        yield "\n".join(lines), None
        return
    lines += ["", "=== after ===", pp.bandwidth(box["out"])]
    yield "\n".join(lines), box["out"]


def _spec_path(tag):
    os.makedirs(UPDIR, exist_ok=True)
    return os.path.join(UPDIR, f"_spec_{tag}.png")


def region_specs(t0, t1, flo, fhi):
    """UI numbers -> the t0:t1:flo:fhi strings both tools take.

    A blank/0 high edge means "up to Nyquist", which differs per file, so it is
    passed as '*' and resolved by the tool rather than guessed here.
    """
    hi = "*" if not fhi or float(fhi) <= 0 else f"{float(fhi):.0f}"
    end = "*" if not t1 or float(t1) <= 0 else f"{float(t1):.3f}"
    return [f"{float(t0):.3f}:{end}:{float(flo):.0f}:{hi}"]


def _render(src, t0, t1, flo, fhi, marker=None):
    """Render the interactive view. Returns (png, geometry) or (None, None)."""
    if not src:
        return None, None
    reg = region_specs(t0, t1, flo, fhi)[0]
    # A distinct filename per render: gradio caches by path, so reusing one
    # name makes the browser show a stale image after a click.
    _render.n = getattr(_render, "n", 0) + 1
    out = _spec_path(f"view{_render.n % 6}")
    g = pp.specview(src, out, region=reg, marker=marker)
    return g["png"], g


def inspect(src, t0, t1, flo, fhi):
    """Bandwidth report + interactive spectrogram with the selection drawn."""
    if not src:
        return "", None, None, "Load a file to begin."
    txt = pp.bandwidth(src)
    try:
        png, g = _render(src, t0, t1, flo, fhi)
    except Exception as e:
        return txt + f"\n\n(spectrogram failed: {e})", None, None, "render failed"
    note = ""
    if g and g.get("truncated"):
        note = (f"  ·  showing the first {g['t1']:.0f}s of "
                f"{g['full_duration']:.0f}s")
    return txt, png, g, "Click a corner of the region you want to fill." + note


def on_click(src, geom, anchor, t0, t1, flo, fhi, evt: gr.SelectData):
    """Two-click corner selection straight on the spectrogram.

    First click drops an anchor, second completes the box. The click is always
    re-rendered with a crosshair at the position this code *believes* was
    clicked, so a coordinate-space mismatch shows up immediately instead of
    quietly producing wrong regions.
    """
    if not src or not geom:
        return anchor, t0, t1, flo, fhi, gr.update(), "Load a file first."

    x, y = (evt.index or (None, None))[:2]
    if x is None:
        return anchor, t0, t1, flo, fhi, gr.update(), "No click position."

    # Outside the plot rectangle: ignore rather than clamp -- a clamped margin
    # click silently produces a degenerate region.
    if not (geom["x0"] <= x <= geom["x1"] and geom["y0"] <= y <= geom["y1"]):
        return (anchor, t0, t1, flo, fhi, gr.update(),
                "Click inside the spectrogram (not the margins).")

    t = geom["t0"] + (x - geom["x0"]) / (geom["x1"] - geom["x0"]) * \
        (geom["t1"] - geom["t0"])
    f = geom["f0"] + (geom["y1"] - y) / (geom["y1"] - geom["y0"]) * \
        (geom["f1"] - geom["f0"])
    marker = f"{t:.4f}:{f:.1f}"

    if anchor is None:
        png, g = _render(src, t0, t1, flo, fhi, marker=marker)
        return ((t, f), t0, t1, flo, fhi, png,
                f"Anchor at {t:.2f}s / {f/1000:.2f} kHz — "
                f"click the opposite corner.")

    at, af = anchor
    nt0, nt1 = sorted((at, t))
    nflo, nfhi = sorted((af, f))
    png, g = _render(src, nt0, nt1, nflo, nfhi, marker=marker)
    return (None, round(nt0, 3), round(nt1, 3), round(nflo), round(nfhi), png,
            f"Region set: {nt0:.2f}–{nt1:.2f}s, "
            f"{nflo/1000:.2f}–{nfhi/1000:.2f} kHz. Click again to start a new one.")


def reset_region(src, geom):
    """Whole file, cliff upward — the band_splice default."""
    if not src:
        return None, 0, 0, 16000, 0, gr.update(), "Load a file first."
    flo = (geom or {}).get("cliff") or 16000
    png, _ = _render(src, 0, 0, flo, 0)
    return (None, 0, 0, round(flo), 0, png,
            f"Reset to whole file above {flo/1000:.2f} kHz.")


def detect_cliff(src, t0, t1, flo, fhi):
    """Set the low edge to the measured cliff AND re-render in one handler.

    Not `.click(...).then(inspect, ...)`: a chained `.then` is invoked with no
    inputs when the first event is triggered over the API, and gradio validates
    arity before calling, so the chain raised
    "didn't receive enough input values (needed: 5, got: 0)" on every API call.
    Doing both here keeps it to one event with one input list.
    """
    if not src:
        raise gr.Error("Pick a file first.")
    hz, drop = pp.cliff_of(src)
    if hz is None:
        raise gr.Error("Could not measure a cliff for that file.")
    if drop is not None and drop < 8:
        gr.Info(f"Cliff {hz:.0f} Hz but only {drop:.1f} dB — that is not a real "
                f"codec cliff, so there may be nothing to restore.")
    return (hz,) + inspect(src, t0, t1, hz, fhi)


def run_upscale(src, kind, auto_lowpass, do_fill, t0, t1, flo, fhi):
    """Upscale, then optionally splice the result into the original.

    Splicing matters: an upscaler rewrites the whole file, and measured on this
    material that costs ~13% relative error below the crossover. region_fill
    adds only the masked correction, leaving everything else bit-identical.
    """
    if not src:
        raise gr.Error("Pick a file to upscale.")

    lines = [f"--- {pp.UPSCALERS[kind]} ---", pp.bandwidth(src), ""]
    yield "\n".join(lines), gr.update(), vram_text(), gr.update()

    out = None
    for text, path in stream_upscale(kind, src, lines, auto_lowpass):
        out = path or out
        yield text, out or gr.update(), vram_text(), gr.update()
    if out is None:
        raise gr.Error("Upscale failed — see the log.")

    regions = region_specs(t0, t1, flo, fhi)
    panels = [("1. INPUT", src), (f"2. {kind.upper()} (whole file)", out)]

    if do_fill:
        hybrid = os.path.join(
            UPDIR, os.path.splitext(os.path.basename(out))[0] + "_hybrid.wav")
        lines += ["", "--- splice into selected region ---"]
        yield "\n".join(lines), out, vram_text(), gr.update()
        try:
            pp.region_fill(src, out, hybrid, regions,
                           on_progress=lambda m: lines.append("  " + m))
            out = hybrid
            panels.append(("3. HYBRID (spliced)", hybrid))
        except Exception as e:
            lines.append(f"  splice FAILED: {type(e).__name__}: {e}")

    try:
        png = pp.spectrogram(
            panels, _spec_path("out"), regions=regions,
            title="Before / after",
            subtitle="Box = the selected region. On the hybrid, everything "
                     "outside it is bit-identical to the input.")
    except Exception as e:
        png = None
        lines.append(f"  spectrogram failed: {e}")
    yield "\n".join(lines), out, vram_text(), png


PREVIEW_SECONDS = 15.0
ALL_TRACKS = "All tracks"

# Remembered generation settings, in the BROWSER's localStorage (gr.BrowserState).
# Per-device by design: the phone and the desktop each keep their own last
# session. Bump the version when the dict shape changes -- an old blob under the
# same key would restore fields that no longer mean the same thing.
SETTINGS_KEY = "audiodev.studio.generate.v1"
SETTINGS_SECRET = "audiodev-studio-local-v1"   # obfuscation, not a security boundary
SETTINGS_FIELDS = ["model", "prompt", "lyrics", "duration", "steps", "seed",
                   "instrumental", "post_kind"]


def preview(model, prompt, lyrics, duration, steps, seed, instrumental,
            src_audio, cover_strength, noise_strength, post_kind):
    """Short render of the opening, at the same seed.

    This is the honest substitute for a live preview. Neither backend can
    stream: MiniMax's autoregressive stage emits no audio at all until it has
    produced every frame, and that stage is most of the wall time, so there is
    literally nothing to play mid-generation. Rendering a short clip at the
    same seed gets you the voice, style and arrangement in a fraction of the
    time -- the opening will be close to the full render's, though not
    identical, since the denoising is chunked differently for a shorter piece.
    """
    yield from generate(model, prompt, lyrics,
                        min(float(duration), PREVIEW_SECONDS), steps, seed,
                        instrumental, src_audio, cover_strength,
                        noise_strength, "none", is_preview=True)


def cover_source(upload, typed_path):
    """A real on-disk path beats the browser's copy.

    `gr.Audio(type="filepath")` hands over a copy in %TEMP%\\gradio\\<hash>\\,
    and the browser never sends the original directory -- so sidecar lyrics
    next to the file are invisible for an upload. A typed path restores that.
    """
    p = (typed_path or "").strip().strip('"')
    if p and os.path.exists(p):
        return p, False
    return upload, bool(upload)


def find_lyrics(upload, typed_path):
    """Probe on selection. Returns (status_markdown, found_text_state)."""
    src, uploaded_only = cover_source(upload, typed_path)
    if not src:
        return "Pick a source track to search for its lyrics.", None
    res = ly.probe(src)
    text = res.get("best", {}).get("text") if res.get("found") else None
    return ly.describe(res, uploaded_only), text


def apply_lyrics(found, current):
    """Never silently clobber typed lyrics."""
    if not found:
        raise gr.Error("Nothing found to apply — try Transcribe.")
    if (current or "").strip() and (current or "").strip() != found.strip():
        gr.Info("Replaced the lyrics box with the lyrics that were found.")
    return found


def transcribe_lyrics(upload, typed_path, separate, model, seconds):
    src, _ = cover_source(upload, typed_path)
    if not src:
        raise gr.Error("Pick a source track first.")

    lines = ["transcribing " + os.path.basename(src)]
    yield "\n".join(lines), gr.update(), vram_text()

    q, box = queue.Queue(), {}

    def work():
        try:
            box["res"] = ly.transcribe(src, separate=separate, model=model,
                                       seconds=float(seconds or 0),
                                       on_progress=lambda m: q.put(m))
        except Exception as e:
            box["err"] = e
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        msg = q.get()
        if msg is None:
            break
        lines.append("  " + msg)
        yield "\n".join(lines[-200:]), gr.update(), vram_text()

    if "err" in box:
        lines.append(f"\nFAILED: {box['err']}")
        yield "\n".join(lines), gr.update(), vram_text()
        raise gr.Error(str(box["err"])[:400])

    r = box["res"]
    lines.append(f"\n{r['segments']} segments, language {r['language']} "
                 f"({r['language_probability']:.2f}), {r['elapsed']:.0f}s"
                 f"{' · vocals isolated' if r['separated'] else ' · full mix'}")
    lines.append("Structure tags are NOT invented — add [verse] / [chorus] "
                 "yourself if you want ACE-Step to follow them.")
    yield "\n".join(lines), r["text"], vram_text()


def _label(e):
    """Caption under a cover: title, then the facts that distinguish takes."""
    t = titles.derive(e["rec"], e["name"])
    bits = []
    if e.get("takes", 1) > 1:
        bits.append(f"take {e['take']}/{e['takes']}")
    r = e["rec"]
    if r.get("seconds"):
        bits.append(f"{r['seconds']:.0f}s")
    if r.get("model"):
        bits.append(r["model"])
    if pls.is_favorite(e["path"]):
        t = "★ " + t
    return f"{t}\n{' · '.join(bits)}" if bits else t


def _filtered(query, only_fav, playlist):
    """The entry list the gallery is showing, after search/filter.

    Returned to a State alongside the gallery so a click indexes the exact
    list that produced the tiles -- recomputing inside the select handler
    would map through a different ordering.
    """
    es = lib.entries()
    if playlist and playlist != ALL_TRACKS:
        want = pls.tracks(playlist)
        order = {n: i for i, n in enumerate(want)}
        es = [e for e in es if e["name"] in order]
        es.sort(key=lambda e: order[e["name"]])   # playlist order, not date
    if only_fav:
        es = [e for e in es if pls.is_favorite(e["path"])]
    q = (query or "").strip().lower()
    if q:
        def hay(e):
            r = e["rec"]
            return " ".join(str(x) for x in (
                titles.derive(r, e["name"]), e["name"], r.get("prompt", ""),
                r.get("lyrics", ""), r.get("model", ""))).lower()
        es = [e for e in es if q in hay(e)]
    return es


def _gallery(es):
    """(cover, caption) tiles. A missing cover falls back to no image rather
    than blocking the refresh -- rendering 40 covers is a background job."""
    out = []
    for e in es:
        c = art.cover_path(e["path"])
        out.append((c if os.path.exists(c) else None, _label(e)))
    return [(c, cap) for c, cap in out if c] or []


_cover_busy = threading.Lock()


def _backfill_covers(entries):
    """Render any missing cover art off-thread.

    Generation covers only the file the worker hands back, but ACE-Step writes
    TWO takes per run and keeps one -- so the sibling never got art, and
    neither did anything produced outside the UI. Doing it here makes the
    library self-healing instead of needing a patch at every path that can
    create audio. Each render is a ~0.6 s subprocess in the media venv, so it
    must not block the refresh; the covers appear on the next one.
    """
    missing = [e["path"] for e in entries
               if not os.path.exists(art.cover_path(e["path"]))]
    if not missing or _cover_busy.locked():
        return

    def work():
        with _cover_busy:
            for p in missing[:60]:
                try:
                    art.ensure_cover(p)
                except Exception:
                    pass                      # a bad file must not stop the rest

    threading.Thread(target=work, daemon=True).start()


def refresh_library(query, only_fav, playlist, view="grid", row=None):
    es = _filtered(query, only_fav, playlist)
    _backfill_covers(es)
    n = sum(1 for e in es if e["recorded"] or e.get("shared_rec"))
    miss = sum(1 for e in es if not os.path.exists(art.cover_path(e["path"])))
    note = f"  ·  {miss} awaiting cover art" if miss else ""
    is_list = view == "list"
    return (_gallery(es), es,
            f"{len(es)} tracks · {n} with a prompt{note}",
            gr.update(choices=[ALL_TRACKS] + pls.names()),
            gr.update(value=rowview.row_html(es, playing_index=row),
                      visible=is_list),
            gr.update(visible=not is_list))


def on_row_action(es, query, only_fav, playlist, view, evt: gr.EventData):
    """One delegated click handler for every button in the row list.

    `gr.HTML`'s js_on_load bridge calls trigger('act', {row, act}), and
    EventData surfaces those straight as evt.row / evt.act.
    """
    idx = getattr(evt, "idx", None)
    act = getattr(evt, "act", "")
    blank = ("Select a track.", "", "", "", "")
    if idx is None or not es or idx >= len(es):
        _, _, st, _, html, _ = refresh_library(query, only_fav, playlist, view)
        return (html, es, gr.update(), None, idx,
                "That row is out of date — refreshed.", *blank)

    e = es[idx]
    r = e["rec"]
    note = None

    if act == "like":
        on = (r.get("rating") or 0) != 1
        lib.update(e["path"], rating=1 if on else None)
        pls.set_favorite(e["path"], on)         # ★ and 👍 are the same gesture
        note = ("Liked" if on else "Like removed") + f" — {e['name']}"
    elif act == "dislike":
        on = (r.get("rating") or 0) != -1
        lib.update(e["path"], rating=-1 if on else None)
        if on:
            pls.set_favorite(e["path"], False)
        note = ("Disliked" if on else "Dislike removed") + f" — {e['name']}"
    elif act == "trash":
        pls.trash(e["path"])
        note = f"Moved **{e['name']}** to trash\\ (recoverable)."

    es2 = _filtered(query, only_fav, playlist)
    # The list can shrink (trash, ★-filter), so re-find the track by name
    # rather than trusting the old index to still point at it.
    idx2 = next((i for i, x in enumerate(es2) if x["path"] == e["path"]), None)
    html = rowview.row_html(es2, playing_index=idx2)
    if idx2 is None:
        return (html, es2, gr.update(), None, None,
                note or "Track left the current view.", *blank)

    cur = es2[idx2]["rec"]
    detail = f"### {titles.derive(cur, e['name'])}\n\n" + lib.detail(es2[idx2])
    return (html, es2, gr.update(), e["path"], idx2,
            note or f"▶ {titles.derive(cur, e['name'])}",
            detail, cur.get("prompt", ""), cur.get("lyrics") or "",
            cur.get("title", ""), cur.get("style", ""))


def cover_track(es, row):
    """Load a library track into Generate as an ACE-Step cover source.

    Sets `src_path` (the real on-disk path) as well as the audio player,
    deliberately: an uploaded copy lands in a gradio temp folder with no
    neighbours, so the lyrics probe can find nothing. A real path lets it read
    the track's OWN json sidecar and offer the lyrics it was generated with --
    which for a cover is usually exactly what you want to keep.

    ACE-Step is the only backend with a cover task, so this switches the model.
    `on_model_change` is bound to `.input` (human clicks only), so setting the
    radio here does not fire it -- which is why the ACE panel's visibility and
    the model note are set explicitly, same as `reuse()`.
    """
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[row]
    r = e["rec"]
    name = titles.derive(r, e["name"])
    status, found = find_lyrics(None, e["path"])
    return (
        "acestep",                                        # model
        r.get("prompt", ""),                              # prompt (edit me)
        r.get("lyrics") or "",                            # lyrics (keep words)
        gr.update(value=bool(r.get("instrumental")), visible=True),
        e["path"],                                        # src_audio player
        e["path"],                                        # src_path (real)
        1.0,                                              # cover_strength
        0.75,                                             # noise_strength
        gr.update(visible=True, open=True),               # ace_box
        f"*{BACKENDS['acestep']['note']}*",               # model_note
        status, found,                                    # lyrics probe result
        gr.Tabs(selected="tab_generate"),
        f"Covering **{name}** — source loaded in Generate. "
        f"Edit the style, then Generate.",
    )


_COVER_NOOP = (gr.update(),) * 14


def on_row_cover(es, evt: gr.EventData):
    """The row list's ↻ button. A SECOND listener on the same `act` event.

    Kept separate from `on_row_action` rather than folded into it: cover writes
    to fourteen Generate-tab components, and merging those into the row
    handler's output list would make every like/dislike click declare outputs
    it never touches. Non-cover actions return no-ops here and are handled
    entirely by the other listener.
    """
    if getattr(evt, "act", "") != "cover":
        return _COVER_NOOP
    idx = getattr(evt, "idx", None)
    if not es or idx is None or idx >= len(es):
        return _COVER_NOOP
    return cover_track(es, idx)


def save_details(es, row, title, style):
    """Commit the detail-pane edits. Empty clears the override."""
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[row]
    lib.update(e["path"], title=(title or "").strip() or None,
               style=(style or "").strip() or None)
    e["rec"]["title"] = (title or "").strip()
    e["rec"]["style"] = (style or "").strip()
    return f"Saved details for **{e['name']}**"


def retitle(es, row, use_llm):
    """Suggest a title. The LLM is opt-in; the heuristic is instant."""
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[row]
    r = e["rec"]
    if use_llm:
        got = llm_title.suggest(lyrics=r.get("lyrics"),
                                prompt=r.get("prompt"), n=1)
        # [] means the model is absent, slow, or the input had no substance --
        # every one of those falls back rather than failing the action.
        t = got[0] if got else titles.derive({**r, "title": None}, e["name"])
        how = "LLM" if got else "heuristic (LLM declined)"
    else:
        t = titles.derive({**r, "title": None}, e["name"])
        how = "heuristic"
    lib.update(e["path"], title=t)
    e["rec"]["title"] = t
    return t, f"Titled **{t}** ({how})"


def pick_track(es, evt: gr.SelectData):
    """Gallery select -> player + detail. evt.index is a plain int here."""
    blank = (None, "Select a track.", "", "", None, "☆ Favourite", "", "",
             "*nothing playing*", None)
    if not es:
        return blank
    i = evt.index
    if isinstance(i, (list, tuple)):
        i = i[0]
    if i is None or i < 0 or i >= len(es):
        return blank
    e = es[i]
    r = e["rec"]
    head = f"### {titles.derive(r, e['name'])}\n\n" + lib.detail(e)
    if e.get("shared_rec"):
        head += ("\n\n*Prompt inherited from take "
                 f"{1 if e.get('take') == 2 else 2} of the same run — "
                 f"ACE-Step writes two takes and only one carries the record.*")
    _, _, _, np_md, np_art = _at(es, i)
    return (e["path"], head, r.get("prompt", ""), r.get("lyrics") or "", i,
            ("★ Favourited" if pls.is_favorite(e["path"]) else "☆ Favourite"),
            r.get("title", ""), r.get("style", ""), np_md, np_art)


def _at(es, i):
    """(path, row, status, now-playing markdown, cover) for one queue index."""
    e = es[i]
    t = titles.derive(e["rec"], e["name"])
    sub = []
    if e["rec"].get("model"):
        sub.append(e["rec"]["model"])
    if e.get("takes", 1) > 1:
        sub.append(f"take {e['take']}/{e['takes']}")
    tail = ("  \n<span class='np-sub'>" + " · ".join(sub) + "</span>") if sub else ""
    cover = art.cover_path(e["path"])
    return (e["path"], i, f"▶ {t}",
            f"**{t}**  ·  {i + 1}/{len(es)}{tail}",
            cover if os.path.exists(cover) else None)


def play_step(es, row, delta, shuffle=False, repeat=False):
    """Move through the queue. Bound to gr.Audio's `stop`, which fires when
    playback reaches the end of the media -- that is what makes this a real
    playlist rather than a list of files you click one at a time."""
    blank = (None, None, "Nothing queued.", "*nothing playing*", None)
    if not es:
        return blank
    if row is None:
        return _at(es, 0)
    if shuffle and delta > 0 and len(es) > 1:
        import random
        choices = [i for i in range(len(es)) if i != row]
        return _at(es, random.choice(choices))
    nxt = row + delta
    if nxt >= len(es):
        if not repeat:
            return None, row, "End of playlist.", "*end of playlist*", None
        nxt = 0
    if nxt < 0:
        nxt = len(es) - 1 if repeat else 0
    return _at(es, nxt)


def play_next(es, row, shuffle=False, repeat=False):
    return play_step(es, row, +1, shuffle, repeat)


def play_prev(es, row, shuffle=False, repeat=False):
    return play_step(es, row, -1, False, repeat)


def toggle_shuffle(on):
    on = not on
    return on, gr.update(variant="primary" if on else "secondary")


def toggle_repeat(on):
    on = not on
    return on, gr.update(variant="primary" if on else "secondary")


def toggle_favorite(es, row):
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[row]
    on = not pls.is_favorite(e["path"])
    pls.set_favorite(e["path"], on)
    return "★ Favourited" if on else "☆ Favourite"


def add_to_playlist(es, row, name):
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    name = (name or "").strip()
    if not name or name == ALL_TRACKS:
        raise gr.Error("Type a playlist name, or pick an existing one.")
    pls.create(name)
    pls.add(name, es[row]["path"])
    return (gr.update(choices=[ALL_TRACKS] + pls.names()),
            f"Added to **{name}** ({len(pls.tracks(name))} tracks)")


def trash_track(es, row, query, only_fav, playlist, view):
    """Move to trash\\ -- reversible, never os.remove."""
    if not es or row is None or row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[row]
    dest = pls.trash(e["path"])
    if not dest:
        raise gr.Error("Could not move that file to the trash folder.")
    g, new_es, status, choices, html, gal = refresh_library(
        query, only_fav, playlist, view)
    return (g, new_es, f"Moved **{e['name']}** to trash\\ (recoverable). "
            + status, choices, html, None, "Select a track.", "", "", None)


def reuse(es, sel_row):
    """Load a track's prompt/lyrics/params back into the Generate tab.

    Supplies everything `on_model_change` would have: that cascade is wired to
    `.input` (human clicks only), so a programmatic model change here does NOT
    fire it -- which is the point. If it did, it would run afterwards and
    overwrite the very prompt being restored with the stock default.
    """
    if not es or sel_row is None or sel_row >= len(es):
        raise gr.Error("Pick a track first.")
    e = es[sel_row]
    r = e["rec"]
    if not e["recorded"]:
        raise gr.Error("That track has no saved prompt — it predates the "
                       "library, or is an alternate take.")
    m = r.get("model", "minimax")
    is_ace = m == "acestep"
    return (
        m,                                                    # model
        r.get("prompt", ""),                                  # prompt
        r.get("lyrics") or "",                                # lyrics
        float(r.get("duration", 30)),                         # duration
        int(r.get("steps", 30)),                              # steps
        int(r.get("seed", 7)),                                # seed
        # value AND visibility in one update -- a component may only appear
        # once in an outputs list.
        gr.update(value=bool(r.get("instrumental", False)), visible=is_ace),
        gr.update(visible=is_ace),                            # ace_box
        f"*{BACKENDS.get(m, {}).get('note', '')}*",           # model_note
        gr.Tabs(selected="tab_generate"),
        f"Loaded **{e['name']}** into Generate ✓",
    )


def unload():
    SUP.unload()
    return "unloaded — GPU released", vram_text()


def on_model_change(model):
    is_ace = model == "acestep"
    return (
        gr.update(visible=is_ace),                       # ACE-only accordion
        gr.update(value=ACESTEP_PROMPT if is_ace else MINIMAX_PROMPT,
                  lines=4 if is_ace else 10,
                  info=("Plain keyword-style prompt." if is_ace else
                        "MiniMax was trained on sectioned captions — keep the "
                        "Global Metadata / Vocal Details / Arrangement "
                        "headings; short prompts lose arrangement control.")),
        gr.update(value=30 if not is_ace else 8),        # steps
        # A separate line, not `info=` on the radio itself: updating a
        # component from inside its own .change handler risks a feedback loop.
        f"*{BACKENDS[model]['note']}*",
        # MiniMax rejects empty lyrics outright, so the checkbox would only
        # ever produce an error there.
        gr.update(visible=is_ace, value=False),          # instrumental
        gr.update(info="Required — MiniMax has no instrumental mode."
                  if not is_ace else
                  "Leave blank, or tick Instrumental above."),
    )


# analytics_enabled=False is not only telemetry hygiene. gradio 6.24 builds its
# analytics payload by asking "is this a custom theme?", comparing THEME against
# every built-in one. THEME passes font= as bare strings and gradio's Font.__eq__
# does `other.name` on them, so that probe dies with AttributeError *after* the
# server thread has printed its URL: launch() raises, the process exits, and 7861
# never comes up. Turning analytics off skips the probe entirely. Wrapping the
# names in gr.themes.Font instead would make gradio quote them ('ui-sans-serif'),
# which is invalid CSS for those keywords and would silently change the typeface.
with gr.Blocks(title="AudioDev Studio",             # css/theme -> launch() in gradio 6
               analytics_enabled=False) as demo:
    gr.HTML(
        '<div class="app-header">'
        '<div class="glyph">▂▅▃▇▄▆▂</div>'
        '<div><h1>AudioDev Studio</h1>'
        '<p class="sub">MiniMax Music 3 · ACE-Step 1.5 · restore &amp; '
        'region-fill — one model resident at a time, so nothing squats on '
        'the GPU</p></div></div>')

    # Still a Markdown: four handlers output plain strings to this component,
    # so only its clothes change. `every=` keeps it honest while a phone is
    # watching and nothing else is firing events.
    vram = gr.Markdown(value=vram_text, every=8, elem_classes="vram-pill")
    settings = gr.BrowserState({}, storage_key=SETTINGS_KEY,
                               secret=SETTINGS_SECRET)

    with gr.Row(elem_classes="mainrow", equal_height=False):
        with gr.Column(scale=5, elem_classes="pane pane-left", min_width=420):
            with gr.Tabs() as tabs:
                with gr.Tab("Generate", id="tab_generate"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            model = gr.Radio(
                                [(BACKENDS[k]["label"], k) for k in BACKENDS],
                                value="minimax", label="Model")
                            model_note = gr.Markdown(f"*{BACKENDS['minimax']['note']}*")
                            prompt = gr.Textbox(
                                MINIMAX_PROMPT, label="Style description", lines=10,
                                info="MiniMax was trained on sectioned captions — keep "
                                     "the Global Metadata / Vocal Details / Arrangement "
                                     "headings; short prompts lose arrangement control.")
                            lyrics = gr.Textbox(
                                LYRICS, label="Lyrics", lines=6,
                                info="Structure tags like [verse] / [chorus] each on "
                                     "their own line. Leave blank to sing freely.")
                            instrumental = gr.Checkbox(
                                False, label="Instrumental (no vocals)", visible=False)

                            with gr.Accordion("ACE-Step: cover an existing track",
                                              # open=True is load-bearing, not taste:
                                              # a COLLAPSED gr.Accordion inside a
                                              # gr.Tab makes gradio 6.24 re-apply every
                                              # sibling's value from the client config
                                              # snapshot when the tab is re-shown, so
                                              # typed lyrics/prompt/seed revert to the
                                              # Python defaults on every tab round
                                              # trip. Bisected two-sided; open=True is
                                              # the whole fix.
                                              open=True, visible=False) as ace_box:
                                gr.Markdown(
                                    "Upload a track to regenerate it in the style "
                                    "above. **Noise strength is the parameter that "
                                    "makes it a cover** — the library default of 0.0 "
                                    "means *pure noise* and produces an unrelated song.")
                                src_audio = gr.Audio(label="Source track",
                                                     type="filepath")
                                src_path = gr.Textbox(
                                    "", label="…or its path on disk",
                                    placeholder=os.path.join(ROOT, "Music", "song.wav"),
                                    info="An upload is a copy in a temp folder, so "
                                         "lyric files sitting next to the original "
                                         "cannot be found. A real path can be.")
                                lyr_status = gr.Markdown(
                                    "Pick a source track to search for its lyrics.")
                                lyr_found = gr.State(None)
                                with gr.Row():
                                    lyr_apply = gr.Button("Apply found lyrics",
                                                          size="sm")
                                    lyr_go = gr.Button("Transcribe", size="sm")
                                with gr.Row():
                                    lyr_sep = gr.Checkbox(
                                        True, label="Isolate vocals first",
                                        info="Demucs before Whisper.")
                                    lyr_model = gr.Dropdown(
                                        ly.WHISPER_MODELS, value="large-v3",
                                        label="Whisper model")
                                    lyr_secs = gr.Number(
                                        0, label="Seconds", info="0 = whole track")
                                cover_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05,
                                                           label="Cover strength")
                                noise_strength = gr.Slider(
                                    0.0, 1.0, 0.75, step=0.05, label="Noise strength",
                                    info="Higher = closer to the source. 0.4–0.8.")

                            with gr.Row():
                                duration = gr.Slider(10, 300, 30, step=5,
                                                     label="Duration (s)")
                                steps = gr.Slider(4, 60, 30, step=1, label="Steps")
                                seed = gr.Number(7, label="Seed", precision=0)
                            fit_note = gr.Markdown("", elem_classes="fit-note")

                            # open=True: see the ace_box note -- a collapsed accordion
                            # in a tab resets its siblings' values on tab switch.
                            with gr.Accordion("Post-process: bandwidth extension",
                                              open=True):
                                gr.Markdown(
                                    "Runs after generation. **Measured: neither "
                                    "generator leaves a codec cliff** (MiniMax 0.6 dB, "
                                    "ACE-Step 3.9 dB), and upresing audio with no cliff "
                                    "makes it worse — so this is normally for lossy "
                                    "*source* material, on the Restore tab. The "
                                    "bandwidth verdict is logged either way.\n\n"
                                    f"*{pp.FINGERPRINT_NOTE}*")
                                post_kind = gr.Radio(
                                    [(v, k) for k, v in pp.UPSCALERS.items()],
                                    value="none", label="Upscaler")

                            with gr.Row():
                                go = gr.Button("Generate", variant="primary", scale=3)
                                prev = gr.Button("Preview 15s", scale=2)
                                drop = gr.Button("Unload", scale=1)

                        with gr.Column(scale=2):
                            gen_status = gr.Markdown("idle")
                            log = gr.Textbox(label="Log", lines=20,
                                             elem_classes="wrap-log")

                with gr.Tab("Restore / Upscale"):
                    # The spectrogram is the instrument here, not an illustration:
                    # full width, at the top, and the thing you click on.
                    up_spec = gr.Image(label=None, type="filepath", interactive=False,
                                       show_label=False, height=560,
                                       elem_classes="specview")
                    up_status = gr.Markdown("Load a file to begin.")
                    up_geom = gr.State(None)      # geometry of the CURRENT render
                    up_anchor = gr.State(None)    # first corner of a pending selection

                    with gr.Row():
                        with gr.Column(scale=2):
                            up_src = gr.Audio(label="File", type="filepath")
                            up_kind = gr.Radio(
                                [(v, k) for k, v in pp.UPSCALERS.items() if k != "none"],
                                value="apollo", label="Upscaler")
                            up_lowpass = gr.Checkbox(
                                True, label="AudioSR: auto-lowpass first",
                                info="AudioSR was trained on lowpass-filtered audio "
                                     "only; fed a raw lossy file it hallucinates from "
                                     "codec artifacts. Leave on for anything compressed.")
                        with gr.Column(scale=2):
                            gr.Markdown(
                                "**Click two corners on the spectrogram** to set the "
                                "region, or type below. An upscaler rewrites the whole "
                                "file (~13% relative error below the crossover); "
                                "splicing adds only the masked correction, so "
                                "everything outside the box stays **bit-identical**.\n\n"
                                "*Filled from the upscaler's rendition of that region — "
                                "'apply the restoration only here', not inpainting.*")
                            with gr.Row():
                                up_t0 = gr.Number(0, label="Start (s)")
                                up_t1 = gr.Number(0, label="End (s)",
                                                  info="0 = end of file")
                            with gr.Row():
                                up_flo = gr.Number(16000, label="Low edge (Hz)")
                                up_fhi = gr.Number(0, label="High edge (Hz)",
                                                   info="0 = Nyquist")
                            with gr.Row():
                                up_detect = gr.Button("Low edge = detected cliff",
                                                      size="sm")
                                up_reset = gr.Button("Reset region", size="sm")
                            up_fill = gr.Checkbox(
                                True, label="Splice into the region "
                                            "(leave the rest untouched)")
                        with gr.Column(scale=1):
                            up_go = gr.Button("Upscale", variant="primary")

                    up_log = gr.Textbox(label="Bandwidth / log", lines=12,
                                        elem_classes="wrap-log")
                    # Results render into their OWN panel. If they overwrote up_spec,
                    # its geometry (up_geom) would describe a different figure and the
                    # next click would map through a stale rect.
                    up_compare = gr.Image(label="Before / after", type="filepath")

        with gr.Column(scale=6, elem_classes="pane pane-right", min_width=380):
            lib_entries = gr.State([])
            lib_row = gr.State(None)

            with gr.Row():
                lib_search = gr.Textbox(
                    "", placeholder="Search titles, prompts, lyrics…",
                    show_label=False, scale=4, container=False)
                lib_playlist = gr.Dropdown(
                    [ALL_TRACKS], value=ALL_TRACKS, show_label=False,
                    scale=2, container=False, filterable=True)
                lib_onlyfav = gr.Checkbox(False, label="★ only",
                                          scale=1, container=False)
                lib_view = gr.Radio(["grid", "list"], value="grid",
                                    show_label=False, scale=1,
                                    container=False)
                lib_refresh = gr.Button("↻", scale=0, size="sm")

            lib_status = gr.Markdown("—")

            # An EXPLICIT height, sized to the viewport. Letting the gallery
            # grow to its content instead looks tidier in theory but is not
            # worth it: without a height its container collapses to 0 and the
            # tiles paint out of flow (measured galleryH:0 with tiles still
            # visible). Its own scroller is the behaviour gradio supports.
            lib_gallery = gr.Gallery(
                show_label=False, columns=4, object_fit="cover",
                height="calc(100vh - var(--dock-h, 132px) - 300px)",
                allow_preview=False, elem_classes="lib-grid")
            # The custom event names 'act' and 'edit' MUST appear quoted inside
            # js_on_load: gr.HTML.__getattr__ regex-scans that string to decide
            # whether lib_rows.act / .edit are legal listeners at all.
            lib_rows = gr.HTML(value="", visible=False,
                               css_template=rowview.CSS,
                               js_on_load=rowview.JS,
                               elem_id="rowlist")

            with gr.Row():
                with gr.Column(scale=3):
                    lib_detail = gr.Markdown("Select a track.")
                    lib_prompt = gr.Textbox(label="Prompt", lines=7,
                                            interactive=False)
                    lib_lyrics = gr.Textbox(label="Lyrics", lines=7,
                                            interactive=False)
                with gr.Column(scale=2):
                    lib_title = gr.Textbox(
                        "", label="Title", placeholder="(derived — type to "
                        "override)", container=True)
                    lib_style = gr.Textbox(
                        "", label="Style / tags",
                        placeholder="doom folk, post rock")
                    with gr.Row():
                        lib_save = gr.Button("Save details", size="sm")
                        lib_retitle = gr.Button("✨ Retitle", size="sm")
                    lib_uselllm = gr.Checkbox(
                        True, label="Use the local LLM for ✨",
                        info=f"{llm_title.MODEL_ID} on CPU, ~9 s. Unticked "
                             f"uses the instant heuristic.")
                    lib_reuse = gr.Button("Reuse prompt in Generate",
                                          variant="primary")
                    lib_cover = gr.Button("↻ Cover this track")
                    lib_fav = gr.Button("☆ Favourite")
                    with gr.Row():
                        lib_plname = gr.Textbox(
                            "", placeholder="playlist name", show_label=False,
                            container=False, scale=3)
                        lib_add = gr.Button("+ Add", scale=1)
                    lib_trash = gr.Button("Move to trash", variant="stop",
                                          size="sm")
                    gr.Markdown(
                        "Each track keeps a **JSON sidecar** holding its "
                        "prompt, lyrics and settings, so the record travels "
                        "with the file. *Not written into the audio's tags — "
                        "that is how the Suno track announced itself.* "
                        "Cover art is rendered from each track's own "
                        "spectrogram. Trash is a folder move, never a delete.")

    _url_file = os.path.join(ASSETS, "phone_url.txt")
    _qr_file = os.path.join(ASSETS, "phone_qr.png")
    with gr.Accordion("📱 Open on your phone", open=False):
        if os.path.exists(_url_file):
            with open(_url_file) as fh:
                _phone_url = fh.read().strip()
            gr.Markdown(
                f"Scan, or open **{_phone_url}** — works anywhere both "
                f"devices are on your tailnet, not just at home. Use your "
                f"browser's *Add to Home Screen* to install it as an app.\n\n"
                f"*Tailnet-only: `tailscale serve` proxies to this machine's "
                f"localhost, so nothing is exposed to the internet. Anyone on "
                f"the tailnet can drive the GPU.*")
            if os.path.exists(_qr_file):
                gr.Image(_qr_file, interactive=False, show_label=False,
                         container=False, elem_classes="phone-qr")
        else:
            gr.Markdown(
                "Tailscale wasn't detected when assets were generated. "
                "Run `tailscale serve --bg 7861` once, then "
                "`watermark\\.venv\\Scripts\\python.exe make_assets.py` "
                "in the studio folder.")

    # ---- the one player -----------------------------------------------------
    # OUTSIDE gr.Tabs on purpose: it must survive tab switches, and every
    # result -- a generation, an upscale, a library track -- lands here instead
    # of in its own per-tab player. Three separate output players is how you end
    # up with two songs going at once.
    with gr.Column(elem_classes="playerdock"):
        with gr.Row(elem_classes="playerbar"):
            now_art = gr.Image(None, show_label=False, container=False,
                               interactive=False, height=52, width=52,
                               elem_classes="np-art", scale=0)
            now_playing = gr.Markdown("*nothing playing*", elem_classes="np")
            player_shuffle = gr.Button("🔀", scale=0, size="sm",
                                       elem_classes="tbtn")
            player_prev = gr.Button("⏮", scale=0, size="sm",
                                    elem_classes="tbtn")
            player_next = gr.Button("⏭", scale=0, size="sm",
                                    elem_classes="tbtn")
            player_repeat = gr.Button("🔁", scale=0, size="sm",
                                      elem_classes="tbtn")
        player = gr.Audio(label=None, show_label=False, type="filepath",
                          autoplay=True, interactive=False, container=False,
                          elem_classes="player", elem_id="mainplayer")
    # Transport state lives server-side so it survives a tab switch; the
    # buttons only toggle it.
    play_shuffle = gr.State(False)
    play_repeat = gr.State(False)

    # .input, NOT .change: .change fires on programmatic updates too, so
    # Reuse setting the model would queue this cascade and it would run
    # afterwards, overwriting the very prompt being restored with the stock
    # default. A human clicking the radio still gets the defaults.
    model.input(on_model_change, model,
                [ace_box, prompt, steps, model_note, instrumental, lyrics])

    _filters = [lib_search, lib_onlyfav, lib_playlist, lib_view]
    _lib_out = [lib_gallery, lib_entries, lib_status, lib_playlist,
                lib_rows, lib_gallery]
    _detail_out = [player, lib_detail, lib_prompt, lib_lyrics, lib_row,
                   lib_fav, lib_title, lib_style, now_playing, now_art]

    # Library is a permanent pane now, not a tab, so there is no select
    # event to hang the first refresh on.
    demo.load(refresh_library, _filters, _lib_out)
    lib_refresh.click(refresh_library, _filters, _lib_out)
    # .input on the search box: .change would also fire when refresh_library
    # rewrites the dropdown, re-entering the refresh it just finished.
    lib_search.input(refresh_library, _filters, _lib_out)
    lib_onlyfav.input(refresh_library, _filters, _lib_out)
    lib_playlist.input(refresh_library, _filters, _lib_out)
    lib_view.input(refresh_library, _filters, _lib_out)

    lib_rows.act(on_row_action,
                 [lib_entries] + _filters,
                 [lib_rows, lib_entries, lib_gallery, player, lib_row,
                  lib_status, lib_detail, lib_prompt, lib_lyrics,
                  lib_title, lib_style])
    lib_save.click(save_details, [lib_entries, lib_row, lib_title, lib_style],
                   lib_status)
    lib_retitle.click(retitle, [lib_entries, lib_row, lib_uselllm],
                      [lib_title, lib_status])

    lib_gallery.select(pick_track, lib_entries, _detail_out)
    # gr.Audio's `stop` fires at end of playback (verified in the 6.24 bundle:
    # wavesurfer `finish` / native `ended` -> dispatch('stop')). That is what
    # makes this an auto-advancing queue instead of a file list.
    _trans_in = [lib_entries, lib_row, play_shuffle, play_repeat]
    _trans_out = [player, lib_row, lib_status, now_playing, now_art]
    player.stop(play_next, _trans_in, _trans_out)
    player_next.click(play_next, _trans_in, _trans_out)
    player_prev.click(play_prev, _trans_in, _trans_out)
    player_shuffle.click(toggle_shuffle, play_shuffle,
                         [play_shuffle, player_shuffle])
    player_repeat.click(toggle_repeat, play_repeat,
                        [play_repeat, player_repeat])
    _cover_out = [model, prompt, lyrics, instrumental, src_audio, src_path,
                  cover_strength, noise_strength, ace_box, model_note,
                  lyr_status, lyr_found, tabs, lib_status]
    lib_cover.click(cover_track, [lib_entries, lib_row], _cover_out)
    lib_rows.act(on_row_cover, lib_entries, _cover_out)

    lib_fav.click(toggle_favorite, [lib_entries, lib_row], lib_fav)
    lib_add.click(add_to_playlist, [lib_entries, lib_row, lib_plname],
                  [lib_playlist, lib_status])
    lib_trash.click(trash_track, [lib_entries, lib_row] + _filters,
                    [lib_gallery, lib_entries, lib_status, lib_playlist,
                     lib_rows, player, lib_detail, lib_prompt,
                     lib_lyrics, lib_row])
    lib_reuse.click(reuse, [lib_entries, lib_row],
                    [model, prompt, lyrics, duration, steps, seed,
                     instrumental, ace_box, model_note, tabs, lib_status])

    # Fit warning: recomputed whenever the length or the model changes, and
    # once on load so it is right before the first click. queue=False keeps it
    # off the GPU queue -- it is arithmetic plus one nvidia-smi call.
    gr.on(triggers=[duration.change, model.change, demo.load],
          fn=duration_hint, inputs=[model, duration], outputs=fit_note,
          show_progress="hidden", queue=False)
    _gen_in = [model, prompt, lyrics, duration, steps, seed, instrumental,
               src_audio, cover_strength, noise_strength, post_kind]
    _gen_out = [log, player, vram, gen_status]
    go.click(generate, _gen_in, _gen_out, concurrency_id=GPU_SLOT)
    # Preview = the same handler with a short duration and no post step. There
    # is no mid-generation preview to offer: the AR stage produces no audio at
    # all until it finishes, and it is most of the wall time.
    prev.click(preview, _gen_in, _gen_out, concurrency_id=GPU_SLOT)
    drop.click(unload, None, [log, vram])
    # Detection is instant and CPU-only -- it runs automatically on selection
    # and stays outside the GPU slot. Applying is a separate, explicit click so
    # a probe can never overwrite lyrics the user typed.
    src_audio.change(find_lyrics, [src_audio, src_path],
                     [lyr_status, lyr_found])
    src_path.input(find_lyrics, [src_audio, src_path],
                   [lyr_status, lyr_found])
    lyr_apply.click(apply_lyrics, [lyr_found, lyrics], lyrics)
    lyr_go.click(transcribe_lyrics,
                 [src_audio, src_path, lyr_sep, lyr_model, lyr_secs],
                 [log, lyrics, vram], concurrency_id=GPU_SLOT)
    # Verdict + annotated spectrogram before any GPU time. CPU-only, so these
    # deliberately stay OUTSIDE the gpu slot and can run during a generation.
    _sel = [up_src, up_t0, up_t1, up_flo, up_fhi]
    _view = [up_log, up_spec, up_geom, up_status]
    up_src.change(inspect, _sel, _view)
    # .input, NOT .change: .change also fires on programmatic updates, so a
    # click that writes all four boxes would kick off four redundant renders
    # on top of the one the click already did.
    for box in (up_t0, up_t1, up_flo, up_fhi):
        box.input(inspect, _sel, _view)
    up_spec.select(on_click,
                   [up_src, up_geom, up_anchor, up_t0, up_t1, up_flo, up_fhi],
                   [up_anchor, up_t0, up_t1, up_flo, up_fhi, up_spec,
                    up_status])
    up_detect.click(detect_cliff, _sel, [up_flo] + _view)
    up_reset.click(reset_region, [up_src, up_geom],
                   [up_anchor, up_t0, up_t1, up_flo, up_fhi, up_spec,
                    up_status])
    up_go.click(run_upscale,
                [up_src, up_kind, up_lowpass, up_fill, up_t0, up_t1,
                 up_flo, up_fhi],
                [up_log, player, vram, up_compare], concurrency_id=GPU_SLOT)
    # --- remember the last generation settings -------------------------------
    # Save on every edit. Deliberately NOT wired to settings.change: that would
    # be save -> change -> save forever. queue=False keeps these off the GPU
    # queue so typing can never sit behind a generation.
    _persist = [model, prompt, lyrics, duration, steps, seed, instrumental,
                post_kind]

    gr.on(triggers=[c.change for c in _persist],
          fn=lambda *v: dict(zip(SETTINGS_FIELDS, v)),
          inputs=_persist, outputs=settings,
          show_progress="hidden", queue=False)

    def restore_settings(saved, cur_model, cur_prompt, cur_lyrics, cur_dur,
                         cur_steps, cur_seed, cur_instr, cur_post):
        """Repopulate the Generate tab from the last session, if there was one.

        Falls back per FIELD, not all-or-nothing: a blob written by an older
        version can be missing keys, and defaulting the whole dict would throw
        away the fields it does have.
        """
        cur = [cur_model, cur_prompt, cur_lyrics, cur_dur, cur_steps,
               cur_seed, cur_instr, cur_post]
        if not isinstance(saved, dict) or not saved:
            return cur + [gr.update()]
        out = [saved.get(k, c) for k, c in zip(SETTINGS_FIELDS, cur)]
        is_ace = out[0] == "acestep"
        # on_model_change is bound to .input (human clicks only), so restoring
        # the model here does NOT fire it -- which is exactly what we want, but
        # it means the ACE-only panel's visibility has to be set here too.
        return out + [gr.update(visible=is_ace)]

    demo.load(restore_settings, [settings] + _persist,
              _persist + [ace_box], show_progress="hidden", queue=False)
    demo.load(lambda: vram_text(), None, vram)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    try:
        # Local only. This UI drives a GPU and writes files; it is not
        # something to expose beyond this machine.
        # api_open=False on queue(), not show_api= on launch(), in gradio 6.
        # server_name stays 127.0.0.1 even with phone access: `tailscale
        # serve` proxies the tailnet HTTPS hostname to localhost, so the app
        # itself is never exposed beyond this machine.
        demo.queue(default_concurrency_limit=1, api_open=False).launch(
            server_name="127.0.0.1", server_port=7861, inbrowser=True,
            css=CSS, theme=THEME, head=HEAD_DARK,
            pwa=True, favicon_path=os.path.join(ASSETS, "icon.png"),
            # Without this, gradio refuses to serve any output file, because it
            # only allows paths under the *launch directory* -- so whether the
            # audio player worked silently depended on where studio.ps1 was run
            # from. OUTDIR covers upscaled/ as a subdirectory; ASSETS carries
            # the QR image.
            allowed_paths=[OUTDIR, ASSETS],
            show_error=True)
    finally:
        SUP.unload()
