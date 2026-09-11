# AudioDev Studio

One local web UI over the installed music generators.

```powershell
B:\AudioDev\studio\studio.ps1        # PC:    http://127.0.0.1:7861
                                     # phone: https://bonelab.tailb09e67.ts.net
```

## Remembered settings, and two gradio bugs worth knowing

### A collapsed accordion resets its whole tab

**`gr.Accordion(open=False)` inside a `gr.Tab` makes gradio 6.24 re-apply every
sibling component's value from the client-side config snapshot when that tab is
hidden and re-shown.** Typed lyrics, prompt, seed and duration all silently
revert to the Python defaults on a tab round trip.

The mechanism: gradio only refreshes its client config snapshot for components
that took part in a fired event. Anything with no `.change` binding — which was
all of the Generate inputs — gets the original default written back. Proven
two-sided in one app: a textbox *with* a `.change` dependency survived while an
identical one without it was lost.

Bisected to a two-line fix, both accordions in the Generate tab set `open=True`:

```
REAL app, unmodified                          LOST
REAL app, minus BOTH accordions               SURVIVED
minimal app + gr.Accordion(open=True)         SURVIVED
minimal app + gr.Accordion(open=False)        LOST      <- flips here
REAL app, accordions forced open=True         SURVIVED
```

Contents and position of the accordion are irrelevant; only `open=False`
matters, and it is decided at config time — expanding one by hand does not help.
`key=` / `preserved_by_key` do **not** fix it. The accordion outside the tabs
("📱 Open on your phone") is unaffected and stays collapsed.

Verified on the fixed app with trusted CDP input and a real mouse click on the
tab: `midTab: "Library"`, `navsAfterLoad: 0`, `verdict: SURVIVED`.

### `launch(js=...)` never runs

The forced-dark redirect this app carried was **dead code**. In gradio 6.24 the
`js=` string is shipped in the page config but there is no code path that
executes app-level config js — only per-event `js=` on a dependency. (`blocks.py`
still documents it as auto-executing; that docstring is wrong.) A marker
function set via `js=` never fired across two full loads.

So the app was **rendering light on a light-preference device**, behind dark
spectrograms. Dark is now forced via `head=`, which does run, and which sets the
class for *this* load and `history.replaceState`s `?__theme=dark` for the next —
`replaceState`, not `location.replace`, because the latter is a full reload and
would wipe every field. A `MutationObserver` re-adds the class when gradio's own
`matchMedia` handler strips it on a system theme flip.

Measured under an emulated **light** system preference: `bodyDark: true`,
`appBg: rgb(15,15,17)`. First-ever visit flashes light for ~60–80 ms before the
head script lands; every later load is dark from first paint.

### Settings persistence

`gr.BrowserState` stores the last-used generation settings — model, prompt,
lyrics, duration, steps, seed, instrumental, post-process — in the browser's
**localStorage** under `audiodev.studio.generate.v1`. Per device, so the phone
and the desktop each keep their own last session, and they survive closing the
browser entirely (verified across a full reload).

Two traps: never wire anything that writes back into the state to its own
`.change` (save → change → save forever), and restore **per field** rather than
all-or-nothing, so a blob written by an older version that is missing keys does
not blank out the fields it does have. Saves run with `queue=False` so typing
never sits behind a generation.

## Phone access (Tailscale)

`tailscale serve --bg 7861` was configured once and **persists across
reboots** — just run `studio.ps1` and the tailnet URL works from any signed-in
device, anywhere, not only at home. The UI's *Open on your phone* accordion
shows the URL and a QR.

Design decisions worth knowing:

- **The app stays bound to `127.0.0.1`.** serve proxies tailnet HTTPS to
  localhost, so nothing listens on any external interface and Windows Firewall
  never enters the picture. Do not "fix" this to `0.0.0.0`.
- **Tailnet-only, never `tailscale funnel`** — funnel is the public internet.
  The boundary is the tailnet: any device on it can drive the GPU. That is
  your own devices, but it is worth saying once.
- **Streaming survives the proxy — verified, not assumed.** Reverse proxies
  classically buffer SSE, which would have turned live progress into
  "everything arrives at 100%" precisely on the phone. Measured through the
  ts.net URL: 6 updates spread over 18.4 s, arriving incrementally.
- `pwa=True`, so the browser's *Add to Home Screen* installs it as an app with
  the studio icon.
- To tear it down: `tailscale serve --https=443 off`. If the machine or
  tailnet name ever changes, re-run `make_assets.py` (watermark venv) to
  regenerate the QR.

On a phone: the layout stacks, the spectrogram scales to full width (a media
query drops its fixed 560 px height, which would otherwise letterbox), and the
two-click region picking works as taps — though fingers are ±1 s at that
scale, so the numeric fields remain the precision tool. The click-position
crosshair check applies doubly on a phone: first tap, confirm the pink cross
lands under your finger.

| backend | output | speed | peak VRAM | can cover a track |
|---|---|---|---|---|
| MiniMax Music 3 | 44.1 kHz stereo, sung lyrics | ~7x realtime | 7.8 GB | no |
| ACE-Step 1.5 | 48 kHz | **~1x realtime** | 4.6 GB | yes |

---

## Why not a ComfyUI backend

It was the obvious idea and it is the wrong one here:

- **ComfyUI has no MiniMax Music 3 node.** The day-0 ComfyUI support widely
  written up is for **MiniMax H3**, a different model (omni-modal video with
  native audio). diffusers ships both, as separate pipelines.
- **ComfyUI is one Python environment.** ACE-Step and MiniMax need incompatible
  ones — ACE-Step's diffusers fights the pinned dev commit MiniMax requires — so
  a node could not import both. It would end up shelling out to subprocesses
  anyway, just buried where you cannot see them.
- **ComfyUI holds VRAM.** On a 10 GB card where MiniMax alone needs 7.5 GB, the
  goal is to *not* share.

## Architecture

```
app.py  (gradio, own venv, no torch)
  └── supervisor.py          at most ONE worker alive
        ├── workers/minimax_worker.py    -> B:\AudioDev\minimax\.venv
        └── workers/acestep_worker.py    -> B:\AudioDev\acestep\.venv
```

Each worker is a warm subprocess speaking **JSON, one object per line**, over
stdin/stdout. The UI venv has no torch at all and reads VRAM from `nvidia-smi`.

### VRAM discipline

This is the point of the design, not a side effect:

1. **One model at a time.** Switching backends kills the other process, which
   returns its memory unconditionally — no dependence on a library releasing it.
2. **Workers park between jobs.** MiniMax pushes every component back to CPU via
   its offload hooks; ACE-Step runs with `offload_dit_to_cpu=True`. Measured
   idle: **0.11 GB and 0.17 GB**, against 4.6 GB for a naively warm ACE-Step.
3. **Unload** frees everything without stopping the UI.

So the models stay loaded for fast repeat generations *without* squatting on the
GPU while you do something else with it.

## Gotchas that shaped the code

**Pipes must be UTF-8 and stdout must be hijacked.** Windows defaults subprocess
text pipes to cp1252, and real lyrics are full of typographic quotes and em
dashes. Separately, workers import code that prints to stdout — one stray line
corrupts the JSON stream — so each worker moves the protocol to a private handle
and points `sys.stdout` at stderr.

**stderr needs a draining thread.** tqdm and transformers write a lot there. A
full Windows pipe buffer blocks the *worker*, which is indistinguishable from a
hung model.

**`offload_dit_to_cpu=True` for ACE-Step.** `gen.py` and `cover.py` hardcode
`False`, which is ACE-Step's own 12–16 GB setting; its `gpu_config.py` defaults
to `True` for a 10 GB tier. Costs ~5 s per clip, saves 4.5 GB of idle residency.

**gradio 6 moved a lot.** `theme`/`css` are `launch()` arguments now, `show_api`
became `queue(api_open=False)`, and `show_copy_button`/`show_download_button`
were replaced by a unified `buttons=` parameter.

**MiniMax has no instrumental mode.** Its text-encoder block declares `lyrics`
required and rejects an empty string, so the Instrumental checkbox is
ACE-Step-only and the UI refuses a MiniMax request with blank lyrics rather than
passing a guessed `[instrumental]` token the checkpoint may not know.

**Under 8 GB free, MiniMax does not fail — it gets slow.** Windows spills CUDA
allocations to shared system memory rather than raising OOM. Measured 7x
realtime with 8.9 GB free versus **28x realtime with 7.0 GB free**. The VRAM
line at the top of the UI warns when free memory is short, because a silent 4x
slowdown is easy to misread as the model being slow.

## Library

A cover-art grid of everything generated, with a play queue, search, playlists,
favourites and reversible delete — the Suno/Udio shape, locally.

| | |
|---|---|
| **Titles** | derived from the lyrics' best line, else the prompt's genre/mood, else the date — never a UUID (`titles.py`) |
| **Cover art** | rendered from each track's **own spectrogram** as a polar disc (`art.py`) |
| **Play queue** | autoplay + auto-advance; `Next ⏭` to skip |
| **Search** | across titles, prompts, lyrics, model |
| **Playlists** | named collections, plus a ★ favourites filter |
| **Takes** | the two clips per run grouped as *take 1 / take 2* |
| **Trash** | a folder move, never `os.remove` |
| **Reuse** | loads model, prompt, lyrics, duration, steps and seed back into Generate |
| **Cover** | loads the track as an ACE-Step cover source, with its lyrics |

### Layout: two panes, one locked player

Creation tools on the left (Generate / Restore as tabs), the library on the
right, each with **its own scrollbar**, and the player **fixed to the viewport
bottom** so it never scrolls away. The Library is no longer a tab — it is a
permanent pane, so its first refresh moved from `lib_tab.select` to
`demo.load`.

The dock is `position: fixed`, and a small script publishes its measured height
as `--dock-h` so the panes can size themselves against it (`height: calc(100vh
- var(--dock-h) - 96px)`) rather than a hardcoded guess that breaks the moment
the dock grows a waveform.

**Three CSS traps, all specific to gradio, all found by measuring the DOM
rather than by looking:**

1. **`flex-wrap: wrap` on `.column`.** Harmless while a column is auto-height;
   the moment you give it a fixed height, children that do not fit wrap into a
   *new column to the right*. Measured: the gallery laid out at `x=1321` in a
   662px pane, with the pane's `scrollWidth` at 1949. Fix: `flex-wrap: nowrap`.
2. **The selector has to out-specify gradio's own.** `.pane { flex-wrap:
   nowrap !important }` did **nothing** — gradio's scoped `.column.svelte-xxxx`
   is also `!important` and has higher specificity, so the single-class
   selector loses the tie. `.gradio-container .column.pane` wins.
3. **`nowrap` then makes children shrink instead.** Default `flex: 0 1 auto`
   means once they cannot wrap they compress to fit the fixed height — which
   squashed the gallery to nothing. Fix: `flex-shrink: 0` on the pane's
   children so the pane scrolls instead.

The gallery keeps an **explicit height** rather than growing to its content.
Letting it grow reads better in theory, but its container collapses to zero and
the tiles paint out of flow (measured `galleryH: 0` with tiles still visible) —
its own scroller is the behaviour gradio actually supports.

Verified in a real headless browser: dock `position: fixed` flush to the
viewport bottom, both panes independently scrollable, and scrolling the library
moves neither the left pane, the page, nor the dock.

### One player, at the bottom

There used to be three output players — Generate's result, the Library's, and
Restore's — plus two file pickers that gradio also renders with controls. Five
sets of transport buttons on one page, one of them autoplaying, which is how you
end up with two songs going at once.

Now there is a single `gr.Audio` **outside `gr.Tabs`**, pinned to the bottom
(`position: sticky`, static on phones). Everything lands there: generations,
previews, upscale results, library tracks, and the auto-advancing queue. Being
outside the tabs is what lets playback continue while you move between them.

Single playback is *also* enforced in the DOM, not just by wiring:

```js
document.addEventListener('play', e => { pause every other audio/video }, true)
```

Capture phase, because `play` does not bubble. The two remaining `gr.Audio`
components are **file pickers** (cover source, restore input) — they need upload
UI and gradio gives them their own controls regardless, so a wiring-only fix
would not have covered them. This also covers any player added later.

A now-playing line above the bar shows the title and queue position
(`▶ Make the Machine Go Uh-Oh · 2/78`).

### Cover from the library

The **↻** button on a row, and **↻ Cover this track** in the detail panel, both
send a track straight into Generate set up as a cover: model switched to
ACE-Step (the only backend with a cover task), the track wired in as the source,
its prompt and lyrics carried over as a starting point, and the strengths set to
the known-good 1.0 / 0.75. Edit the style, press Generate.

It sets the **real on-disk path**, not just the audio player. That matters: an
uploaded copy lands in a gradio temp folder with no neighbours, so the lyrics
probe finds nothing — while a real path lets it read the track's own JSON
sidecar and offer the exact lyrics it was generated with, which for a cover is
usually what you want to keep.

Wired as a *second* listener on the row list's `act` event rather than folded
into the main row handler: cover writes to fourteen Generate-tab components, and
merging those in would make every like/dislike click declare outputs it never
touches. Non-cover actions no-op through it.

The **…** overflow button still emits its event but has no menu behind it yet.

### Grid or list

A **grid/list toggle** sits with the filters. List view is the Suno row layout:
cover thumbnail with a play overlay and duration badge, title, style line, take
badge, and a strip of circular actions (like / dislike / edit / cover / more).

That view is one `gr.HTML` component, not a stack of gradio widgets, because
nothing else in gradio 6 can render a variable-length list of rows each with
their own buttons. `gr.HTML` turns out to be a full custom-component system —
`css_template`, `js_on_load`, and a `trigger(event, payload)` bridge. Two
things about it are load-bearing and were established by driving a real
headless browser against a live server, not by reading docs:

- **The custom event name must appear quoted inside `js_on_load`.**
  `gr.HTML.__getattr__` regex-scans that string to decide whether
  `component.act(...)` is even a legal listener; without the literal `'act'`
  you get `AttributeError` at wiring time.
- **The click listener is delegated onto `element`, never bound per button.**
  A Python-side value update replaces `innerHTML` but not `element`, so
  per-button handlers die silently on the first repaint.

Measured round-trip for a row action: **62–125 ms**.

**Editing lives in real gradio textboxes, not `contenteditable` in the HTML.**
That was a deliberate reversal: inline editing works, but an *unsolicited*
repaint — a refresh, a finished generation — while a field has focus discards
whatever was half-typed. Textboxes below the list cannot be clobbered that way.
Title and style are stored as `title` / `style` overrides in the sidecar via
`library.update()`, which read-modify-writes so an edit never drops the prompt
the track was generated with, and writes through a temp file so a torn sidecar
reads as "no record" rather than corrupt. Clearing a field removes the override
and falls back to the derived title.

Like/dislike write `rating` (+1/−1) into the sidecar; a like also sets ★, since
they are the same gesture.

### ✨ Retitle — a tiny local LLM

`llm_title.py` runs **Qwen2.5-1.5B-Instruct on CPU** (the GPU belongs to music
generation — verified zero CUDA allocation) in the MiniMax venv, which already
has transformers. ~9 s cold, and it declines rather than inventing:

| input | title |
|---|---|
| the Paycheck lyrics | *Paycheck to Paycheck* |
| "Morning light filtering through the pine…" | *Morning Light Filtering Through Pine* |
| style only: "bpm 92, E minor, Electric Blues" | *Night Train Through the Fog* |
| style only: "upbeat indie pop, jangly guitar" | *Jangle Jamboree* |
| `[verse]\n[chorus]` | *(declines — falls through to the prompt)* |

That last row was a defect I fixed after the workflow: the model confidently
produced "Fire in the Hole" from nothing but structure tags. Input is now
screened for actual substance — tags and filler syllables stripped — and a
hollow lyric falls through to the style instead. Untick the LLM box and ✨ uses
the instant heuristic titler instead.

Everything degrades: no model, no network, or an unusable answer all fall back
to `titles.derive()`, which cannot fail. Offline with no cached model fails in
**0.1 s** rather than hanging.

### Why the covers are spectrograms

Every track gets real art with no second model, no download and no API — and
because it is derived from the audio, it actually depicts the track. It also
worked **retroactively**: all 36 pre-existing files got covers in 23 s.
Backfill or re-render any time with

```powershell
B:\AudioDev\watermark\.venv\Scripts\python.exe B:\AudioDev\studio\make_covers.py [--force]
```

New generations render their own cover inline (~1 s against a multi-minute job).

### The auto-advancing queue is real, not a Next button

`gr.Audio` has **no `end` event** in gradio 6.24 — but it does have `stop`,
whose docstring is *"triggered when the user reaches the end of the media"*.
Verified in the shipped frontend bundle: wavesurfer's `finish` and the native
`<audio>` `ended` handler both dispatch `stop`. So `lib_audio.stop(play_next,…)`
gives genuine continuous playback. Without that, a playlist would have been a
list of files you click one at a time.

### Takes, recovered from timestamps

ACE-Step writes **two clips per run** and the worker keeps one, so the sibling
would sit in the library as an orphan. Measured across 18 runs here, the pair is
always written 0–1 s apart while separate generations are minutes apart — so
`library.group_takes()` pairs them by mtime proximity (same extension, ≤2.5 s)
and lets the take without a sidecar inherit its run's prompt. This works
**retroactively on files that predate the library entirely**. It stays
conservative because a wrong pairing would attach one track's prompt to
unrelated audio; the GPU slot serialises generations, so two real runs cannot
land that close. An inherited record is marked `shared_rec` and still reports
"no sidecar", so Reuse refuses it rather than pretending.

### Storage

Everything generated keeps a **JSON sidecar** next to it, and clicking a track
shows the prompt and lyrics it was made from.

Each track keeps a **JSON sidecar** beside it (`<track>.json`) holding prompt,
lyrics, model, seed, steps, duration, cover parameters, timing and peak VRAM.
Per-file rather than one central index, because the record then travels with
the audio if it is moved or copied and cannot drift out of sync with an index;
the library view is just a folder scan, so it is idempotent and self-healing
and files that appear later show up on their own.

**Not written into the audio's tags, deliberately.** This toolkit exists partly
because a Suno track carried "made with suno" in its RIFF `LIST/INFO` chunk.
Baking prompts into the file would replay that on our own output. A sidecar is
provenance you keep, not provenance you ship.

The sidecar doubles as a lyrics store: `..\lyrics_probe.py` reads `.json`
sidecars, so using one of our own generations as a cover source finds the
lyrics it was made with automatically.

Two things the table is honest about rather than hiding:

- Tracks with no sidecar show **"(no record — found on disk)"**. Those predate
  the library, came from CLI/test runs, or are the **alternate take ACE-Step
  emits alongside every generation** — it produces two variants per run and the
  worker keeps one.
- CLI runs (`cli.py`) do not write records; only the UI does, since that is
  where the parameters are known.

### One wiring subtlety

Reuse would otherwise clobber its own work. `model.change` fires on
*programmatic* updates too, so setting the model from a record would queue the
`on_model_change` cascade, which runs afterwards and overwrites the restored
prompt with the stock default. The model radio is therefore wired to
**`.input`** (human clicks only) and the Reuse handler supplies the visibility
updates the cascade used to. Same trap as the numeric boxes on the Restore tab.

Recording is a **side effect**, not an output: adding it to `generate()`'s
returns would change the arity of every yield and of two endpoints. The
regression asserts the endpoints are unchanged.

## Why long MiniMax songs get slower, and what fits

A MiniMax run that starts at a sane speed and then degrades is not the model
warming up — it is running out of VRAM part-way through.

**The autoregressive stage needs two models co-resident**: the 4-bit LM
(6.29 GB) and the RVQ depth decoder (1.20 GB) = **7.49 GB before a single frame
exists**. On top of that, two things grow every frame:

| | per token | per second of music |
|---|---|---|
| KV cache (2 K/V x 8 kv-heads x 128 dim x bf16 x 36 layers x **CFG batch 2**) | 288 KiB | 7.03 MiB |
| per-frame hidden states (8 codebooks x 4096, bf16) | 64 KiB | 1.56 MiB |

So roughly **8.6 MiB per second of audio**, on a card that had ~0.1–1.4 GB spare
to begin with. Windows does not fail when you exceed VRAM — it silently pages to
system memory over PCIe, which is far slower for weight reads. Measured on this
machine: **8.9 GB free → 7x realtime; 7.0 GB free → 28x realtime.** A real run
of a 30 s song took 689 s (23x) because the card only had ~7.6 GB free.

Worse, transformers' default `DynamicCache.update()` is
`self.keys = torch.cat([self.keys, key_states])` — it **reallocates and copies
the entire cache every frame, across all 36 layers**. That is O(N²) copying, a
transient second copy at every step, and steady allocator fragmentation.

### What is fixed

- **`--dry-run` and the Generate tab now report the budget by duration**, so a
  length that cannot fit says so *before* the wait rather than after.
- **`..\minimax\ar_cache.py`** replaces the cache. `Qwen3Model.forward` only
  builds a `DynamicCache` when none is passed, so wrapping the LM and supplying
  one on the prompt pass is enough to own it for the whole song.
  - `static` — `StaticCache`, preallocated, written in place. No per-frame
    copy, no transient double, no fragmentation.
  - `quantized` — `QuantizedCache` (quanto, 4-bit, recent tokens kept full
    precision). ~4x less KV memory while keeping the **whole** history.
  - `auto` (default) — picks `static` when bf16 fits, `quantized` when it does
    not, and warns when neither will.

Measured on the LM directly, 40 steps: dynamic 2.60 s, static 2.41 s,
quantized 3.13 s. Quantizing costs ~20% per step and buys 4x the length.

### What actually fits

| song | bf16 KV | 4-bit KV |
|---|---|---|
| 30 s | 8.4 GB | 8.1 GB |
| 60 s | 8.7 GB | 8.2 GB |
| 2 min | 9.2 GB | 8.3 GB |
| 3 min | 9.7 GB | 8.5 GB |
| **4 min** | **10.2 GB — exceeds the card** | **8.7 GB** |
| 5 min | 10.7 GB | 8.9 GB |

**A 4-minute song is impossible at bf16 on a 10 GB card** regardless of tuning.
With the 4-bit cache it needs 8.7 GB — which means it also needs the desktop to
give some back. Idle Windows + browsers here holds ~2.4 GB, leaving 7.6 GB;
closing browsers gets to roughly 9.2 GB, and then 4- and 5-minute songs fit.

### On sliding the cache instead

Evicting old frames to bound memory is sound in principle — RoPE rotates keys by
their absolute position at write time, so dropping old keys leaves the retained
ones mathematically intact, and 4 minutes stays inside this checkpoint's 10240
trained positions. The trap is that `Qwen3Model` derives each new token's
position from `past_key_values.get_seq_length()`: shrink the cache and the next
token gets a *lower* position than keys already in it, so relative distances go
negative and attention corrupts silently. Doing it properly means driving
`cache_position` from a monotonic counter that ignores evictions.

It is also the lossy option — it throws away the model's memory of what it
already sang, which is the long-range coherence this checkpoint is sold on.
Quantizing keeps the entire history at a quarter of the size, so it is the
better answer to the same problem and is what `auto` reaches for.

## Progress, and why there is no live preview

The Generate tab shows a live stage + percentage + ETA and a bar, driven by
**torch forward hooks**, not tqdm parsing. That distinction matters: tqdm
redraws in place with `\r` and emits no newline until it closes, while the
supervisor and log readers all iterate `for line in pipe`, which only yields on
`\n` — a tqdm bar would surface exactly once, at 100%. `register_forward_hook`
is the supported `nn.Module` API and lives in `_forward_hooks`, so it does not
collide with accelerate's `_hf_hook` offload wrappers or with `park()`.

Counts are derived from the pipeline source, not guessed:

| stage | hook | denominator |
|---|---|---|
| autoregressive | `language_model.model` | `max_frames + 2` |
| flow matching | `transformer` | `chunks x steps x guider.num_conditions` |

`encoders.py` calls `language_model.model(...)` once for the prompt and once per
frame; `_generate_depth_codes` only touches `language_model.model.embed_tokens`,
a *submodule*, so it does not trip the hook. `denoise.py` calls the transformer
once per guider branch per step over 200-frame windows. The branch count can
vary by step range, so the denoise figure is an estimate and the meter clamps to
100%. Ticks are throttled to ~0.5 s / 2% — 25 AR frames per second would
otherwise be 25 JSON messages per second.

**ACE-Step keeps coarse stage messages only**, deliberately: its inference
exposes no tqdm or callback to hook, and a whole generation takes ~10 s, so
instrumenting its internals could not repay the cost.

### Preview is a short render, not a stream

There is **no mid-generation preview to offer, architecturally**. MiniMax's
autoregressive stage accumulates `frame_hiddens` and only stacks them *after*
the loop finishes — no audio exists until it completes, and that stage is most
of the wall time. (The model card agrees: "Only non-streaming generation is
currently supported.") A partial decoder would therefore show nothing for most
of the wait.

**Preview 15s** is the honest substitute: the same handler, same seed, short
duration, post-step skipped. It gets you the voice, style and arrangement in a
fraction of the time. The opening will be close to the full render's but not
identical — a shorter piece is chunked differently by the denoiser.

## Lyrics for a cover

The ACE-Step cover panel probes the source track automatically on selection and
reports what it found; **Apply found lyrics** writes them into the Lyrics box.
Detection never overwrites the box on its own — you may have typed something.

Two places are searched, by `..\lyrics_probe.py` (no models, instant):

- **Embedded** — ID3 `USLT`/`SYLT`/`TXXX:LYRICS`, Vorbis
  `LYRICS`/`UNSYNCEDLYRICS`, MP4 `©lyr`, **and RIFF `LIST/INFO`** for plain WAV.
  That last one is read by importing `read_riff_info` from
  `..\watermark\ai_audio_forensics.py` rather than trusting mutagen, which does
  not surface those chunks for plain PCM — the same gap that once hid a literal
  "made with suno" comment 164 bytes into a file.
- **Sidecars** — `.lrc` / `.txt` / `.srt` / `.vtt` beside the audio or in a
  `lyrics\` subfolder, ranked: exact stem > filename prefix > token overlap.
  The match is always named and justified in the status line, because a folder
  can easily hold `excerpt_lyrics.txt` next to the file you actually want. On
  the real fixture it picks `paycheck_lyrics.txt` for
  `Paycheck to Paycheck.wav` and rejects the excerpt.

Timing is stripped but **structure tags are not**: a greedy `\[.*?\]` would eat
`[verse]` and `[chorus]`, which are exactly what ACE-Step conditions on.

### The upload trap

`gr.Audio(type="filepath")` gives the server a copy in `%TEMP%\gradio\<hash>\`,
and the browser never sends the original directory. **Sidecar detection can
therefore never work on an upload** — there are no neighbouring files to search.
That is why the panel has a *"…or its path on disk"* box, which takes precedence
over the upload; with only an upload, the status line says so rather than
implying no lyrics exist.

### Transcription

If nothing is found, **Transcribe** runs `..\transcribe.py` in its own venv
(`B:\AudioDev\lyrics\.venv`, Python 3.10, torch 2.7.1+cu128 installed *before*
demucs so it cannot drag in CPU torch): Demucs `htdemucs` isolates the vocal,
then faster-whisper transcribes it. The two never share the GPU — Demucs is
freed before Whisper loads, so peak VRAM is the larger, not the sum.

Two settings that were measured, not assumed, on `Paycheck to Paycheck.wav`
(which has `paycheck_lyrics.txt` as ground truth):

- **`vad_filter` is OFF.** With it on, output fragmented badly and words were
  corrupted — "My pain looks around me / With holy" where the truth is "My pet
  looks up at me with holy hungry eyes" — and language confidence fell to 0.65.
  Turning it off restored 0.95. It was the single biggest quality factor here,
  and it initially made vocal separation look harmful when it was not.
- **`condition_on_previous_text=False`.** Lyrics repeat by design; with
  conditioning on, Whisper latches onto a chorus and loops it.

Honest result of the separation A/B, VAD off: on this track Demucs and the full
mix produced **equivalent content**. Separation additionally caught the
non-lexical "Mmm, mmm, mmm" intro and sung contractions ("Livin'", "shakin'")
that the mix version smoothed away, at roughly double the runtime. It is on by
default as the more robust choice for dense mixes, but this one clean, vocal-
forward track did not demonstrate a large win — uncheck it if you want speed.

Output is plain lines, paragraph-broken on long silences. It deliberately does
**not** invent `[verse]` / `[chorus]` tags: ACE-Step conditions on those, and
wrong structure is worse than none. Add them by hand.

## Bandwidth extension (post step)

Both a post step on the Generate tab and a standalone **Upscale** section for
arbitrary files. `postprocess.py` shells out to the existing wrappers
(`apollo.ps1`, `audiosr.ps1`, `flashsr_long.py`) rather than reimplementing
them — they carry the ffmpeg staging, chunking and folder-contract handling.

| tool | ground-truth error | speed on a 20 s clip | notes |
|---|---|---|---|
| AudioSR | **4.37 dB** (best) | slowest | keeps 48 kHz; auto-lowpass for lossy input |
| Apollo | 7.00 dB | **12 s** | trained on codec artifacts; resamples 48 k → 44.1 k |
| FlashSR | 11.34 dB (worst) | 93 s | keeps 48 kHz |

**Read this before using it on generated audio.** Super-resolution restores a
hard codec cliff. Measured on this machine:

| file | cliff | drop | worth upresing? |
|---|---|---|---|
| MiniMax output | 8.3 kHz | 0.6 dB | **no** — no cliff |
| ACE-Step output | 12.6 kHz | 3.9 dB | **no** — no cliff |
| 96 kbps MP3 | 15.4 kHz | 48.1 dB | yes |

So the post step on fresh output is usually a no-op at best and a degradation at
worst. The genuinely useful path is the Upscale section: **restore a lossy
source track before covering it**. The UI runs `check_bandwidth.py` and shows
the verdict before spending GPU time, and logs it during the post step — but
proceeds either way, since you asked for it.

Verified end to end on a 96 kbps file: Apollo took the 15.4 kHz / 48 dB cliff to
22.0 kHz / 9.8 dB; FlashSR to 18.3 kHz / 9.7 dB; AudioSR left no detectable
cliff at all.

### How much high end each one invents

The cliff number alone is misleading. Because that test file was made by
compressing a track we still have, the pre-compression original is available as
ground truth — energy above 15 kHz as a share of the total:

| | >15 kHz |
|---|---|
| 96 kbps input | 0.001% |
| **original, pre-MP3** | **0.019%** |
| Apollo | 0.015% |
| FlashSR | 0.030% |
| AudioSR | 0.761% |

Apollo lands nearest the truth; AudioSR adds roughly **40x more high end than
was ever there**, with auto-lowpass already on. Caveat before generalising:
this source was MiniMax output, which is unusually HF-poor to begin with
(0.15% above 12 kHz), so AudioSR restoring ordinary-music levels necessarily
overshoots here. It does not overturn the repo's ground-truth ranking on real
recordings — it does mean **check the result rather than assuming the most
accurate tool is the right one for a given source.**

**One thing to weigh given the forensics work in `..\watermark\`:** AudioSR and
FlashSR stamp a 100 Hz neural-vocoder comb (hop 480 @ 48 kHz, confirmed in their
source). Upscaling *adds* a detectable AI fingerprint that was not there before.

### Picking the region on the spectrogram

The **Restore / Upscale** tab is built around a full-width spectrogram you click
on. **Two clicks set a region** — first drops an anchor, second completes the
box; the numeric fields stay in sync as the precise source of truth, so you can
click roughly and then type an exact edge. *Low edge = detected cliff* and
*Reset region* cover the common cases, and the comparison figure renders into
its own panel below.

The click mapping is exact, and that took a purpose-built renderer.
`spectrogram.py` saves with `bbox_inches="tight"`, which crops the canvas to the
drawn content — fine for a report, useless for picking, because the final pixel
size *and* the axes position are then both unknown. `specview.py` instead:

- fixes `figsize x dpi` so the canvas is exactly 1300 x 560 px
- places the axes with an explicit `add_axes([L, B, W, H])`
- uses **none** of `tight_layout()`, `constrained_layout`, or
  `bbox_inches="tight"` — any one of them silently moves the axes
- emits the plot rectangle as JSON, computed from the *same* constants that
  positioned the axes, so the two cannot drift

Result: plot rect `x[78,1248] y[28,504]`, and a (t,f) -> pixel -> (t,f)
round-trip that is exact to floating point. A click at 5.00 s lands at 4.995 s —
under a third of a pixel.

Two invariants the wiring depends on:

- The geometry lives in a `gr.State` set by the **same event return** that sets
  the image, never a module global — otherwise a click during a re-render maps
  through a stale rectangle. For the same reason results render into a separate
  panel rather than replacing the interactive view.
- The numeric boxes use `.input`, not `.change`. `.change` also fires on
  programmatic updates, so a click that writes all four boxes would kick off
  four redundant re-renders on top of its own.

Clicks outside the plot rectangle are rejected with a status message rather than
clamped — a clamped margin click silently produces a degenerate region.

**If a click ever lands somewhere you did not click**, that is the one untested
assumption: gradio is taken to report *natural image pixels*. Every click
re-renders with a crosshair at the position the app believes you clicked, so a
mismatch is visible immediately instead of silently producing wrong regions.

`..\region_fill.py` is the tool underneath. `band_splice.py` already did the
frequency-domain splice, but whole-file and frequency-only; this generalises it
to rectangles in the time-frequency plane:

```powershell
# default region = detected cliff -> Nyquist, whole file (same as band_splice)
B:\AudioDev\watermark\.venv\Scripts\python.exe B:\AudioDev\region_fill.py `
  --orig in.wav --restored sr.wav -o hybrid.wav

# explicit rectangles, repeatable:  t0:t1:flo:fhi   ('' or * = open end)
... --region 5:10:16000:* --region 12:15:12000:18000
```

It splices a **correction**, not a replacement:

```
hybrid = orig + ISTFT( mask * (STFT(restored) - STFT(orig)) )
```

ISTFT is linear, so wherever the mask is zero the correction is exactly 0.0 and
`orig + 0.0` is bit-identical — no STFT round-trip error is introduced outside
the regions at all. The tool measures that rather than asserting it. Verified on
the 96 kbps fixture, filling only 5–10 s above 16 kHz:

| | |
|---|---|
| samples outside the region | 653,682 (74.1%) |
| max abs difference there | **0.000e+00** |
| drift below 16 kHz inside the region | 0.0966% (−60.3 dB) |
| …for comparison, whole-file replacement | **12.97%** (−17.7 dB) |

That last pair is the argument for the feature: splicing does ~134x less damage
to the band you can actually hear.

Handled because it bites on the very first real pair: **the upscalers change
sample rate** (AudioSR/FlashSR emit 48 kHz, Apollo forces 44.1). The restored
file is resampled into the original's rate, since the correction has to live in
the original's domain or bit-exactness is impossible, and the output is written
with the original's subtype so untouched samples round-trip through the file
format unchanged.

**Not inpainting.** Regions are filled from the upscaler's rendition of that
region — "apply the restoration only here". A true dropout, actual silence in
the source, has nothing to splice from, and none of the installed tools invent
content from nothing.

## Testing without the UI

`cli.py` drives the same supervisor headless, which keeps protocol bugs
separable from UI bugs:

```powershell
python cli.py --model minimax --duration 15
python cli.py --model acestep --duration 15
python cli.py --model acestep --src "B:\AudioDev\Music\track.wav"   # cover
python cli.py --both                                               # switch test

# upscalers, headless
python postprocess.py "Music\studio\lossy_test.wav" --kind apollo
```

`Music\studio\lossy_test.wav` is a kept fixture — a 96 kbps re-encode with a real
15.4 kHz cliff, which is what makes it a meaningful upscaler test. Generated
output has no cliff and tests nothing.

Outputs land in `B:\AudioDev\Music\studio\`.

## Scope

v1 exposes what the installed scripts already do. No model downloading, preset
library, or job history.
