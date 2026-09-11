# Requirements the migration must satisfy

User-reported, 2026-08-14, while the FastAPI/mobile rebuild was in flight.
These are **acceptance criteria**, not nice-to-haves — the migration is not done
until each is demonstrably true on an **iPhone-sized viewport**.

## 1. Row actions must actually do their thing

**Bug, root cause confirmed in the gradio app:** `on_row_action` in
`studio/app.py` branches on `like`, `dislike` and `trash` only. `edit` and
`more` are emitted by the row list but hit the fall-through, which selects the
row and loads the player — so **both buttons just play the song**.

The new UI must implement, per row:

| action | behaviour |
|---|---|
| `play` / thumbnail | load into the player and play |
| `like` / `dislike` | toggle `rating` ±1 in the sidecar; like also sets ★ |
| `edit` | open an editor for **title** and **style/tags** (inline row editor or a sheet). Must NOT start playback. |
| `cover` | load as an ACE-Step cover source (already works — keep) |
| `more` | open a real menu: Edit, Cover, Reuse prompt, Add to playlist, Download, Show in folder, Move to trash |

Every action must `stopPropagation` so it never falls through to the row's own
play handler. That fall-through is the entire bug.

## 2. Played / unplayed state must be visible

There is currently no way to tell a new track from one already heard.

- Mark a track **played** once it has actually been listened to (fire at
  ≥50% elapsed or on `ended`, not on load — merely selecting it is not playing
  it).
- Persist server-side in the track's JSON sidecar via `library.update()`:
  `played: true`, `plays: <int>`, `last_played: <iso>`. Server-side, not
  localStorage, so the phone and the desktop agree.
- A track with no `played` field is **new**. Show an unmistakable but quiet
  marker — a small accent dot at the row's leading edge (Suno uses exactly
  this) — and expose an "unplayed only" filter alongside "★ only".

## 3. The player dock is too tall

Mini bar target on mobile: **≤ 64 px** of content plus the safe-area inset.
Art 44 px, single-line title with the style/queue line beneath, transport to
the right. The waveform/scrubber belongs in the **expanded** view, not the
mini bar. Tapping the mini bar expands to a full-screen now-playing sheet;
swipe-down or a chevron collapses it.

## 4. Default to list view

List is the default on every viewport. Grid is opt-in and the choice persists
in localStorage.

## 5. iOS — this is an iPhone, so these are not optional

- **`100vh` is wrong on iOS Safari.** It counts the collapsed URL bar, so a
  full-height layout is taller than the visible area and the bottom is cut off.
  Use `100dvh` with a `100vh` fallback, or `-webkit-fill-available`.
- **`viewport-fit=cover` is required in the viewport meta**, otherwise
  `env(safe-area-inset-*)` returns 0 and the dock sits under the home
  indicator. Pad the dock with `env(safe-area-inset-bottom)`.
- **Autoplay needs a user gesture.** Reuse ONE `<audio>` element for the whole
  session and only change its `src` — recreating the element, or calling
  `play()` outside a gesture chain, gets blocked and the queue silently stops
  advancing. Auto-advance works because the first tap unlocked that element.
- `touch-action: manipulation` on controls to kill double-tap zoom and the
  300 ms tap delay.
- `position: fixed` misbehaves while the on-screen keyboard is open — the
  Create form must not trap the dock over an input.
- For home-screen install: `apple-mobile-web-app-capable` and
  `apple-mobile-web-app-status-bar-style`; standalone mode removes the URL bar,
  which changes the viewport height again — hence `dvh`.

## 6. Quality settings behind an Advanced panel

Everything we did to squeeze MiniMax onto a 10 GB card is a **quality
trade**, and on a bigger card it should simply be switched off. Today those
choices are hardcoded (`quant="4bit"`, `QUANT_RVQ` env var, `kv_cache="auto"`,
`--reserve 1GB`, ACE-Step's `offload_dit_to_cpu=True`). Surface them.

Computed from the measured budget model (`ar_cache.budget`), GB needed for a
**4-minute** song, against realistic free VRAM (total minus ~1.5 GB desktop):

| preset | LLM | RVQ | KV cache | needs | 10GB | 12GB | 16GB | 24GB |
|---|---|---|---|---:|:--:|:--:|:--:|:--:|
| Smallest | 4-bit | 4-bit | 4-bit | 7.87 | ✅ | ✅ | ✅ | ✅ |
| Balanced | 4-bit | 8-bit | 4-bit | 8.17 | ✅ | ✅ | ✅ | ✅ |
| **Default (10 GB)** | 4-bit | bf16 | 4-bit | 8.74 | ✅ | ✅ | ✅ | ✅ |
| Full cache (12 GB+) | 4-bit | bf16 | **bf16** | 10.18 | ❌ | ✅ | ✅ | ✅ |
| High (16 GB+) | **8-bit** | bf16 | bf16 | 12.37 | ❌ | ❌ | ✅ | ✅ |
| Maximum (24 GB+) | **bf16** | bf16 | bf16 | 19.88 | ❌ | ❌ | ❌ | ✅ |

Component footprints: LLM 4-bit **6.29** / 8-bit **8.47** / bf16 **15.99** GB;
RVQ 4-bit **0.33** / 8-bit **0.64** / bf16 **1.20** GB.

Requirements:

- Collapsed **Advanced** section, closed by default. Nothing here should be in
  a casual user's way.
- **Auto-select the preset from detected total VRAM** on first run, then
  remember the choice. A 24 GB card should not silently inherit a 10 GB
  compromise — that is the whole point of this item.
- Choosing **Custom** exposes the individual controls: LLM precision, RVQ
  precision, KV cache mode (auto / static / quantized / dynamic), offload
  reserve, and ACE-Step's DiT offload.
- The **fit warning must follow the chosen preset**, not the hardcoded
  defaults — `ar_cache.budget()` already takes `base_gb` and `kv_bits`, so pass
  them through. A wrong warning is worse than none.
- Say what each trade costs in plain language: the KV cache is a *storage*
  change (full history, coarser); RVQ precision affects **acoustic detail**;
  LLM precision affects **musicality and prompt adherence**.
- The biggest quality lever is LLM precision, and it needs **24 GB** — so on
  10 GB the honest ceiling is "Default", and that should be stated, not hidden.

## 7. Cover mode: hidden until on, obvious when on

- The cover controls (source track, cover strength, noise strength) are
  **hidden by default**. They appear only in cover mode.
- Cover mode turns on either by an explicit toggle in Create, or automatically
  when arriving from the library's **↻ Cover** action.
- While it is on, it must be **unmistakable**:
  - a persistent banner at the top of the Create form: the source track's cover
    art, "Covering **<title>**", and an ✕ to leave cover mode
  - the primary button reads **Create cover**, not Generate
  - the model is pinned to ACE-Step (the only backend with a cover task) and
    the reason is shown, rather than the radio silently moving
- Leaving cover mode clears the source and restores the normal Create form.

## 8. Recovery: nothing may be unrecoverable

There is one GPU and one lane, so a job that wedges blocks everything and there
is no second machine to fall back on. Every state must have a way out **from
the UI**, without restarting the server.

- **Every job is cancellable**, at every stage and in every kind — queued or
  running, generate, upscale or transcribe. A queued job leaves the lane; a
  running one has its process killed (`taskkill /T`, because the upscalers run
  under `powershell.exe` and killing only the shell orphans the python that
  holds the VRAM).
- A killed job is recorded as **cancelled**, never as an error.
- Tapping the GPU chip opens the **GPU sheet**: current VRAM, what is resident,
  every live job with its own Cancel, and:
  - **Unload model** when the lane is clear
  - **Force reset GPU** — cancels the queue, kills whatever is running, frees
    the memory. Armed by a second tap, never a browser `confirm()`.
- `POST /api/gpu/unload {"force": true}` is that endpoint. It must not call
  `Supervisor.unload()` while a job holds the lock — that is the deadlock it
  exists to break.
- The forced response tells the truth: when a kill is in flight it reports
  `killing: <job_id>`, **not** `unloaded: true` with a stale VRAM figure. The
  client polls until the memory actually comes back.
- After a job fails, its progress card offers **Free the GPU**.
- A dropped SSE connection is **not** a failure. Locking a phone must never
  report a running job as errored; the stream reconnects, and a snapshot poll
  covers the gap while it is down.

## 9. Library → Restore, and the two new Restore stages

**Send to Restore.** The song's ⋮ menu carries it, next to Reuse and Cover. It
fills the source path *and* inspects immediately — landing on an empty pane and
having to press Inspect is the clunky version.

**Degrade first.** `postprocess.py` already records that neither generator here
produces a codec cliff, and that upresing material with no cliff makes it worse.
This is the other half of that finding: make a cliff, then let the upscaler
rebuild it, so the band is *synthesised* rather than merely sharpened.

- The band cut is a **resample round trip**, not a lowpass. ffmpeg's `lowpass`
  is a biquad capped at 2 poles; chained to 4th order it still measured only
  7 dB down an octave above the corner, and `check_bandwidth` found no cliff at
  all. Resampling to 2·fc and back puts soxr's anti-alias filter at Nyquist —
  a real brick wall, and for AudioSR literally the training condition.
- `auto` matches the damage to the upscaler: Apollo was trained on codec
  artifacts (→ MP3 round trip), AudioSR and FlashSR on lowpassed audio (→ cut).
- The codec round trip is **two ffmpeg passes**. Encoding and decoding inside
  one filter graph keeps float samples throughout and produces no artifacts,
  which is the entire point of the mode.
- The degraded file is emitted as its own artifact and is playable — you must
  be able to hear what was done to the input.
- **The splice still runs against the pristine original**, never the degraded
  copy. Everything outside the region stays bit-identical to the source.

**Analogue pass** (`naturalize.py`, watermark venv). Runs last, on the finished
file: 24 Hz highpass → oversampled asymmetric saturation → pink noise floor →
LUFS match. Requirements that are not negotiable:

- Saturation is **4× oversampled**. Measured: a 15 kHz tone at 44.1 kHz folds
  its 3rd harmonic back to 900 Hz at −23.7 dB without oversampling and
  −207.5 dB with it. That fold-back is an audible inharmonic whistle.
- The curve is **asymmetric** (a small bias), because that is what generates
  even harmonics — the warm ones. A symmetric curve gives odd harmonics only.
- A **second highpass after** the saturation. Asymmetry rectifies, so it makes
  DC out of real material: measured −0.032 without it, +0.0005 with it.
- The result card offers **A/B against the un-naturalized file**. "More
  natural" is a claim only ears settle, so the UI must make the comparison
  one tap.

Both stages validate at POST time (400, like `region_spec`), both are
cancellable (§8), and both default to off.

## 10. The writer: LLM lyrics and style prompts

A local LLM fills either text box. **Two targets × three modes**, one endpoint:

| mode | what it does |
|---|---|
| generate | writes the whole thing from the brief; ignores the box |
| extend | continues from what is there, without repeating it |
| edit | changes only what the brief names, returns the rest word for word |

- **MiniMax's structure is enforced, not hoped for.** MiniMax was trained on
  sectioned captions and loses arrangement control without them, so the style
  target generates against that exact skeleton and is then *checked*: three
  headings on their own lines **and** all six field prefixes (`Basic
  Attributes:`, `Sonics & Production Profile:`, `Vocal Gender & Timbre:`,
  `Vocal Style:`, `Primary:`, `Secondary:`). Headings alone are not enough —
  measured, the model will answer with the right headings in markdown over
  bullet lists, which passes a heading-only check and is the wrong format.
  A failed check costs one corrective round trip; if it still fails the text is
  returned with a warning rather than swallowed.
- ACE-Step gets a different template entirely (one keyword line). `model` picks.
- **It is a job, not a sync POST.** 30–180 s is long enough that a phone screen
  lock would kill a plain request — the thing §8 exists to prevent.
- **Device is chosen per request, and it never waits for the card.** Measured
  on this box, Qwen2.5-1.5B: 2–4 tok/s on CPU fp32, 20 tok/s on CUDA fp16 —
  a full lyric is ~3 minutes against ~25 seconds. So: GPU when the lane is idle
  *and* ≥4 GB is free, otherwise its own CPU thread off the lane. Queueing a
  lyric behind a four-minute render would cost more than the CPU run it
  replaced.
- `Registry.submit_cpu` is the off-lane executor. Same succeed/fail/cancel
  conversion as `_drain`, so the SSE contract and §8 cancellation do not fork.
  No queue, because off the GPU there is no scarce resource to serialise.
- The model runs as a subprocess under the MiniMax venv (the studio venv has no
  torch), registered via `job.extra["proc"]` — so a write is cancellable
  through the same path as everything else.
- **Editing lyrics is structural, not prompted.** When the brief names a
  section ("make the chorus angrier", "rewrite verse 2"), the section is cut
  out, rewritten alone, and spliced back — every other section is returned
  byte-for-byte because it was never sent to the model. This is not a
  refinement; it is the only thing that makes edit work at this model size.
  Measured twice on the whole-text approach: asked to rewrite the chorus of a
  two-section draft, the model returned seven sections, and on the second run
  lost the instruction entirely. Tightening the prompt did not fix it.
  - The tag is forced back onto the replacement, so a model that renames the
    section cannot silently restructure the song.
  - A brief that names no section, or an ambiguous one ("rewrite the verse"
    when there are two), falls back to the whole-text path, which detects a
    grown section count and warns.
- **The seed is returned, and the sheet has a seed box.** Random by default:
  `llm_title` derives its seed from the text because a title must be stable, a
  lyricist needs a reroll. The result's seed is written back into the box so
  the roll that just happened is repeatable — a seed the UI cannot accept is
  not a feature.
- The button is **hidden, not disabled, when the weights are absent**, and the
  writer never downloads in-band.

## 11. LoRAs: a folder you drop into, a list you pick from

Adapters are discovered, not configured. Two roots, scanned per request:

    B:\AudioDev\loras          the drop folder (created if absent)
    B:\AudioDev	rain\lora_out  anything trained here, checkpoints included

An adapter is a **directory holding both** `adapter_config.json` and
`adapter_model.safetensors`. That pair is the whole test — PEFT writes them
together, and a directory with one of them is a half-finished save.
`studio/loras.py` imports no torch, no peft, no safetensors: the studio venv
has none of them, and this is what builds the dropdown.

- **The list is never cached.** The point of a drop folder is that a file
  appears in it and then appears in the UI.
- **Per-epoch checkpoints are listed too**, because A/B-ing epoch 5 against
  epoch 10 is a real thing to want.
- **Ids are derived from the path**, so they survive a rescan; a remembered id
  that no longer exists is a 400 at POST, never a job that dies mid-render.
- **The adapter name passed to ACE-Step is explicit.** Its own
  `_default_adapter_name_from_path` uses the directory basename, so
  `train\lora_outinal` and `lorasnythinginal` would both register as
  `"final"` and the second would silently not be what was asked for.
- **A base-model mismatch is shown, not hidden** — a LoRA trained against
  another base loads happily and produces noise.
- LoRAs are **ACE-Step only** (they are adapters on its DiT). Asking for one
  alongside MiniMax is a 400 with the reason. The control stays **visible on
  every model**, disabled where it does not apply, with the reason and the
  adapter count in its hint — the same pattern as the Instrumental checkbox.
  The first version hid it outright, which made the entire feature invisible
  to anyone on the default model (MiniMax) with nothing on screen to explain
  why. Hiding is right for cover mode, which is a *mode the user turns on*;
  it is wrong for a persistent capability that has to be discoverable.

**The worker's state machine is where the danger is.** Every handler call
(`add_lora`, `remove_lora`, `set_use_lora`, `set_lora_scale`) returns a
**string** beginning with a tick or a cross — there are no exceptions to
catch. An unchecked call fails by generating with the wrong model while the UI
reports success, which is the worst outcome this feature has. Every return
value is checked and a cross fails the job with its message.

**Changing adapter restarts the worker (~40 s).** ACE-Step loads a LoRA
reliably and cannot take one off: `remove_lora` reaches PEFT 0.20's
`delete_adapter`, which raises a bare `KeyError('<adapter>')` for the adapter
it was just asked about — measured, twice. So the supervisor treats a
different adapter exactly as it treats a quantization change: stop the process,
start a clean one. Scale changes within the same adapter are in-place and free.
The worker's unload path is deliberately unreachable and raises a message
saying the restart rule regressed, rather than guessing.

Backward compatibility: a request with no `lora_path` key behaves exactly as
before, which is what keeps the Gradio app on :7861 working.

**Provenance.** Which adapter and what scale are read back from the handler
(what was in the forward pass), not from the request (what was asked for), and
written into the sidecar beside the prompt.

**Acceptance is the weights AND the audio.**

"The handler says it loaded" is not evidence. The weight-level check is the
deterministic one: after loading, targeted modules must be PEFT `lora.Linear`
carrying the adapter, the delta `B @ A * scaling` must be non-zero, `scaling`
must track `set_lora_scale` proportionally, and the adapter must appear in
`active_adapters` (`test_lora_weights.py`: 192 adapted modules, delta 1.2359 at
scaling 2.0 -> 0.3090 at 0.5, ratio exactly 0.25).

The audio check is valid too, but only because **the seed is now honoured**.
An earlier version of this spec declared ACE-Step non-deterministic and threw
the audio assertions away. That was wrong: the worker passed a bare
`GenerationConfig()`, whose `use_random_seed=True` makes `prepare_seeds`
discard the seed and roll a random one per item. Every render ignored the UI's
seed. With `use_random_seed=False, seeds=[n]`, a fixed seed reproduces the
audio byte for byte -- three renders, 7-9 s of real work each, one hash -- so
base-vs-LoRA at one seed isolates the adapter, and a base render after a LoRA
must match one from before it.

The end-to-end test then covers what the weights cannot: that the whole
none → A → scale → none cycle completes without killing the worker, and that
switching away actually restarted it.

## 12. Generation config defaults are not our defaults

`GenerationConfig()` ships two defaults that were silently wrong for this app,
and both hid for weeks because nothing failed:

- **`use_random_seed=True`** — `prepare_seeds` then discards the seed argument
  entirely and rolls a fresh random one per item. Every ACE-Step render ignored
  the UI's seed box, and the model looked non-deterministic when it was only
  ever un-seeded. Pass `use_random_seed=False, seeds=[n]`; `config.seeds` is
  preferred over `params.seed` by `inference.py`, so set both.
- **`batch_size=2`** — two takes rendered per request. The worker reported one
  and the sibling was stranded in the library with no sidecar, one per
  generation, forever. Set to 1 unless both takes are surfaced deliberately.

**Read the result, do not diff the folder.** `GenerationResult.audios` is the
real field — a list of dicts with the path and the params that made it.
`audio_files` and `files` do not exist, so code asking for them always fell
through to a folder diff. That fallback breaks the moment seeds work:
`generate_uuid_from_params` is documented as "same parameters will always
generate the same UUID", so a repeat render overwrites its predecessor and the
diff reports "produced no output file" for a generation that succeeded.

## 13. Takes, thinking, and scaling past this card

**Takes.** 1 by default; the ceiling comes from `max_takes()`, which reads the
card (2 on 10 GB, 4 at 16, 8 at 24+). Each take is a different song from one
prompt — ACE-Step reseeds every take after the first — so siblings get their
own sidecar and cover art marked `sibling_of`, rather than being stranded the
way `batch_size=2` stranded them.

**Thinking mode.** ACE-Step's 5Hz LM plans the song before the DiT runs: bpm,
key, time signature, optionally a rewritten caption. Opt-in.

- It was never blocked by nano-vllm having no Windows wheel. That was the
  stated reason and it was wrong. `generate_music(dit_handler, llm_handler, …)`
  takes the LM as its **second argument** and the worker passed `None`, so the
  gate `llm_handler is not None and llm_handler.llm_initialized` could not open
  whatever `thinking` was set to.
- `LLMHandler.initialize(backend="pt")` loads the 1.7B checkpoint with plain
  `AutoModelForCausalLM`. Measured: 23 s, and with `offload_to_cpu=True` it
  costs **~6.8 GB of system RAM, not VRAM** — it reaches the card only while
  thinking. Loaded lazily, so a worker that never thinks never pays.
- Cost per generation once warm: ~25 s on top of a 15 s render.
- `lm_temperature` is the weirdness dial and exists **only** in this mode.
- Acceptance is the audio: at a fixed seed, thinking-off, thinking-on and
  thinking-on-at-a-different-temperature must all differ. That test is only
  meaningful because §12 made the seed real; before that it would have proved
  nothing, exactly like the LoRA audio test did not.

**Scaling is a rule, not a constant.** The 10 GB card is the floor this was
built on. Every limit tied to it reads the hardware instead:

| | 10 GB | 24 GB, no edits |
|---|---|---|
| `run_lora.py --preset auto` | `vram_8gb`, rank 16 | `vram_24gb_plus`, rank 128 |
| `max_takes()` | 2 | 8 |

A limit that hardcodes today's card silently hands a bigger one the small
card's ceiling, and nobody notices because nothing fails.

## 14. Automatic duration

ACE-Step's docstring says `duration < 0` lets the model choose, and it does —
but only through the LM. The length arrives as CoT metadata
(`params.cot_duration`, `inference.py:794`), so with thinking off there is
nothing to make the decision.

Measured, one seed, three prompts of deliberately different scale:

| prompt | lyrics | chosen |
|---|---|---|
| a piano sketch | 3 lines | **13 s** |
| verse/chorus song | 12 lines | **75 s** |
| six-section epic | 14 lines | **192 s** |

179 s of spread — it reads the content and sizes to it. **With thinking off the
same request returns a flat 120 s every time**, which is a default wearing the
costume of a choice.

So auto duration **requires thinking**: a 400 if asked for without it, and
ticking the box in the UI turns thinking on and says so. A control that means
"the model decides" in one configuration and "120 seconds" in another is worse
than no control.

**Continuation is not available on turbo.** `TASK_TYPES_TURBO` is
`[text2music, repaint, cover, cover-nofsq]`; the `complete` task ("Complete the
input track with…") and `lego` are base-model only. Extending a song that ends
mid-phrase therefore needs either the base model — a VRAM question, so it opens
up on a bigger card — or a detect-and-retry loop built here.

## 15. Do not leak file handles in a long-lived server

`sf.info(path)` opens the file to read its header and leaves closing to garbage
collection. In a script that is harmless; in the studio server it is a leaked
handle per call, and cover art is rendered for **every new track** — so every
new track stayed open for the life of the process.

The symptom was not a crash. **Move to trash failed** with "Could not move that
file to the trash folder" on anything recently generated, because Windows will
not rename an open file. It cleared the instant the server was stopped, which
is what identified the holder.

Any audio read on the server path uses `with sf.SoundFile(...) as f:` and takes
both the header and the samples from that one handle. Never `sf.info()`
followed by `sf.read()`.

## 16. Workspaces, playlists, trash

Three different jobs, and conflating them is the mistake this section exists to
prevent. `playlists.py` already states the rule that decides where each kind of
state lives: *a relationship BETWEEN tracks goes in a shared index; a fact
ABOUT one track goes in that track's own sidecar.*

| | what it is | where it lives |
|---|---|---|
| **Workspace** | the project a song was made FOR — cinematic attempts kept apart from pop attempts | a `workspace` field in each sidecar |
| **Playlist** | a curated set with a goal — an album, a mixtape; ordered | `playlists.json` |
| **Trash** | a folder move, reversible | `trash\` |

**A song is born into a workspace.** You do not add to one afterwards — that is
what playlists are for. The tag rides the generate request and is written by
`_run_generate` alongside `quality` and `lora`, for the primary take and its
siblings alike.

- **Resolved server-side at POST**, from `workspaces.json`, never from the
  client's current state. The phone and the desktop both generate, and two
  localStorage copies would tag the same project inconsistently.
- Switching workspace mid-render does not move a song that is already going.
- `active = null` means untagged, and the library's default view filters on
  **nothing at all**. 111 tracks predate workspaces; hiding them behind a
  filter would read as data loss.
- Deleting a workspace forgets the NAME only. Tracks keep their tag and their
  audio — rewriting 90 sidecars because a label was removed would be a
  destructive answer to a cosmetic request. Recreating the name brings them
  back into view.
- The picker is a select, not a chip: a chip filters what you are looking at,
  a workspace decides where new work lands. Create states which one is active
  next to the button that does the tagging — invisible auto-tagging is how a
  month of work ends up in the wrong bucket.

**Trash is a MODE, not a filter.** Trashed items are not library entries: no
id, no rating, nothing to queue, and some have no sidecar at all. Feeding them
through `paintRows`/`setQueue` would break the player, so they get their own
render with one action. `/api/trash` enriches each item from its sidecar when
there is one and degrades to filename + date when there is not.

**Restore only.** Emptying the trash permanently is deliberately not offered:
the ask was to see what had been binned, and restore is the reversible half of
the move that put it there.

## 17. The list keeps your place — except when the filter moves

Rebuilding the row window on every refresh is what threw you back to the top
after adding track #90 to a playlist, binning a dud, or simply finishing a
render — all things you do deep into a night's output, which is exactly where
losing your place hurts.

- a mutation that changes one row **patches that row**, it does not rebuild the
  list: rating, favourite, playlist add, playlist remove and trash all leave
  `scrollTop` untouched — verified 2026-08-21 at row 100 of 130, `scrollTop`
  6830 → 6830, the target row replaced in place at the same index
- `state.shown` only resets when the **filter** moves (`filterKey()`: query,
  favourites, playlist, workspace, trash), never on a plain refresh
- when the filter *does* move the list starts **at the top**. `#main` is the
  scroller — not `#rowlist`, and not `document.scrollingElement`, both of which
  read 0 forever and make a probe pass while the bug is live
- **the trap this exists for:** without the scroll reset the old `scrollTop` is
  merely *clamped* to the shorter list, so typing a query while parked at the
  bottom lands you on match 65 of 73. Worse, the sentinel is then still inside
  its own `rootMargin`, and `IntersectionObserver` fires only on a *transition* —
  no boundary is crossed, so the window sticks at 60 and no amount of scrolling
  grows it. Measured before the fix: 60 rows of 73, `scrollTop` pinned at its
  own maximum across six scroll attempts. After: 73 of 73
- `dropEntry()` patches both numbers in the status line, and the second one
  ("N with a prompt") only moves when the track that left actually had one

## 18. The lock screen is the primary interface

A phone in a pocket with AirPods in is the realistic listening session here, so
the Media Session API is not a nicety — with it unset iOS shows generic controls
with no title and a next button that does nothing.

- metadata is set on every load **after** `audio.play()`: set before playback
  has started it is discarded, because until the element is producing sound
  there is no session to attach it to
- title / the artist name from §21 / workspace-or-`AudioDev Studio` / cover art, and the five
  actions `play`, `pause`, `nexttrack`, `previoustrack`, `seekto`, registered
  once in `init()` rather than per load so they cannot close over a stale queue
- **`play` and `pause` assert a state, they do not toggle it.** A lock screen
  sends `play` because it believes we are paused; bound to a toggle, a wrong
  belief does the opposite of what the button says. Verified 2026-08-21:
  `play` at an already-playing element leaves it playing, `pause` at an
  already-paused one leaves it paused
- `navigator.mediaSession.playbackState` is written on every transition and
  cleared to `'none'` in `stop()`. Left unset it reads `'none'` forever and the
  OS has only the media element to infer from — which is what makes a lock
  screen send the wrong action in the first place
- the dead-file breaker stops a skip loop over a wiped folder at three
  consecutive failures. It is reset by evidence of sound and by a **deliberate**
  pick (`play()`, `next(true)`, `prev()`), never inside `load()` — the
  auto-advance chain runs `error → next(false) → load()`, so resetting there
  would zero the counter between every failure and the breaker could never trip.
  Resetting *only* on sound is the opposite failure: the counter latches at
  three and the next dead file you tap reports "three in a row" on its first
  failure. Both directions verified 2026-08-21, 12/12 in
  `web/scratchpad/probe_fixes.py`

## 19. Search covers what you wrote by hand

The point of a style field you curate is that it becomes findable.

- the haystack is title, filename, prompt, lyrics, model **and** style,
  workspace, adapter name and checkpoint
- terms are ANDed rather than matched as one substring, so word order stops
  mattering. Measured across the change on the same 130 tracks: `dark pop`
  8 → 10, `acestep 110 bpm` 0 → 14, and nothing anywhere returned fewer
- the `#q` placeholder names what is actually searched. It said "titles,
  prompts, lyrics" for a day after the haystack grew, which is the one piece of
  UI whose whole job is to say so
- provenance reaches the wire: `lora_name`, `lora_scale`, `variant`,
  `thinking`, `lm_temperature`, `quality`, `workspace`
- `thinking` is **tri-state on purpose** — `null` means "this track predates the
  field", `false` means "thinking was deliberately off". `bool(x) or None`
  collapsed both to null, which is information loss in the one payload whose
  entire job is provenance
- **`api.py` edits are dormant until a restart.** uvicorn runs without
  `--reload`, so an in-process import test and a `?mock=1` browser run can both
  pass against code the running server has never loaded. Restart with
  `web/restart_studio.ps1` (it refuses while a job is running) and re-check the
  live wire, or the change is not shipped

## 20. Verifying a browser change

Four times this project has shipped a UI change that was verified by reading the
code and never seen in a browser, and four times the user found out instead.

- verify in a real browser at 390×844 against the live server, collect
  `pageerror`, and report measured values
- **the module-map trap:** `api.py` rewrites every import specifier to
  `./x.js?v=<build>`, so `import('/player.js')` from a probe builds a **second,
  never-`init()`ed** module — `entries()` returns 0 while 60 rows are on screen.
  Reach the app's real singletons only through the versioned URL the page itself
  used
- a probe that plays a track to completion writes `played`/`plays` to a real
  sidecar. Use `?mock=1` for anything involving playback — `api.js:22` swaps the
  whole transport, so nothing reaches disk
- toasts linger for seconds and a `MutationObserver` re-scans the DOM on every
  mutation, so clearing a captured list while the old toast is still on screen
  just re-reads it. Wait for the screen to empty before measuring the next run
- assertions must be mutually satisfiable. "`scrollTop` is unchanged" and "the
  row below holds its viewport position" cannot both hold after deleting a
  visible row above it — one row height has to come out of somewhere

## 21. The artist name is entered, never inferred

The lock screen carried a hard-coded name for one day, which credited every
track to a label nobody had typed. The app does not get to decide who made the
music.

- **there is no default artist and there must not be one.** Unset is the honest
  state: the player passes an empty `artist`, iOS shows the title alone, and no
  fallback is substituted. The field's placeholder is the generic "Artist name"
  — pre-filling a suggestion would re-invent the label this exists to remove
- it is a **studio-wide setting**, not a fact about one track, so it lives on
  the server (`Music/studio/settings.json` via `studio/settings.py`) rather
  than in a browser: the phone and the desktop both play, and two localStorage
  copies would disagree about the credit on a lock screen
- entered from **Settings** in the app bar. 44px target, 16px field so iOS does
  not zoom the page — verified 2026-08-21 at 390px, 22/22 in
  `web/scratchpad/probe_artist.py`
- clearing it is a legitimate edit: `PUT {"artist": ""}` stores empty and
  returns it. Only a *missing* key is an error
- changing it repaints the **current** track's credit, not just the next one —
  `setArtist()` re-runs `setMediaSession()`. Verified live: mid-playback edit
  moved the metadata from "The Quiet Hours" to "Second Name"
- names are the user's to choose. Cleaning strips control characters and
  collapses runs of whitespace, and nothing else: punctuation, accents, and
  non-Latin scripts all belong to somebody. Capped at 64 characters, and the
  server *says* when it shortened rather than silently truncating
- the module is `studio/settings.py`, **not `profile.py`** — `profile` is a
  stdlib module (the profiler) and api.py puts the studio folder at the front
  of `sys.path`, so that filename would shadow it for the whole server process
- **not written into the audio's tags**, for the reason library.py gives: a
  Suno track arriving with "made with suno" in its LIST/INFO chunk is what this
  toolkit exists to avoid. Provenance you keep, not provenance you ship

### What the app must not name for you

The same rule generalises. Mock fixtures, placeholders and example data must not
borrow the user's own vocabulary — workspace names in `mock.js` were briefly
taken from their private folder names, which is the same mistake at a smaller
scale. Fixtures use generic names (`Soundtrack`, `Singles`, `Sketches`).

## 22. The design system

The app follows a supplied reference design. Its rule is that **every accent has
a job and nothing is tinted for decoration**, so the palette is not a mood — it
is a legend.

| token | value | job |
|---|---|---|
| `--void` | `#07060E` | the ground |
| `--ink` / `--ink-2` / `--ink-3` | `#0E0C1C` / `#151130` / `#1d1840` | raised surfaces |
| `--rule` | `#241E44` | every hairline |
| `--lumen` | `#C9C4FF` | body text |
| `--muted` | `#8B85B8` | secondary text |
| `--violet` | `#7A5CFF` | the app's own chrome, and **Create** |
| `--rose` | `#FF5CC8` | the music — **Library** |
| `--cyan` | `#35E6E0` | the machine — **Restore**, and measured numbers |

- `--accent` is **re-pointed per pane** from `body[data-pane]`, which `go()`
  sets. One rule set carries all three; no component hard-codes a hue. Verified
  2026-08-21: library → rose, create → violet, restore → cyan
- `--violet` is **4.16:1 on `--surface-2`** — fine for a border or a fill,
  under AA for body text. `--accent-txt` is the lighter tint for words. Use
  `--accent` for ink and edges, `--accent-txt` for text
- the primary button is a **flat accent fill, not a gradient**: a dark label
  measures 4.61:1 on violet at full strength but 3.10:1 against the darker stop
  a gradient introduces. The fill is what passes AA, the gradient is what failed
- **square corners.** `--r-sm/md/lg/xl` are all `0`. The visual language is
  hairlines and viewfinder brackets, and a rounded corner cannot carry a
  bracket. Circles survive only where the *thing* is round — cover art is a
  circular spectrogram, a dot is a dot — and those set `border-radius`
  explicitly rather than through a token
- **the workspace picker holds 16px** because a `<select>` is an editable field
  and iOS zooms the page when it opens the picker. The box got narrower
  instead: at a 168px cap with a fully tracked label it ate 266px of a 390px
  chip row and pushed every filter off screen. Now 193px

### The three devices

1. **the micro-label** — mono, uppercase, `--ls-label` tracking, accent-coloured,
   trailed by a rule that fades to transparent. Names a region. The rule is a
   flex child rather than a border so it takes exactly the width the text leaves
2. **the viewfinder** — four corner marks on `.bracket` that grow from 13px to
   26px and take the accent on hover or `:focus-within`. Frames a panel without
   drawing a box round it
3. **the hairline** — 1px `--rule` wherever a card used to be. Structure comes
   from division, not from fills and shadows

### Type

Three faces, three jobs: `--f-display` for names, `--f-body` for prose,
`--f-mono` for every label and measured number. `--font`, `--mono` and
`--font-display` remain as aliases so no existing rule needed editing.

**Still no `fonts.googleapis.com` link.** The phone reaches this over Tailscale
and may have no route out, so a remote face would sit in the render path of the
app's own headings. The families are *named first* with the system stack behind
them, so dropping self-hosted `woff2` into `static/fonts/` upgrades the whole
app without touching a rule. Until then the system stack renders and the layout
is unchanged — the design's character is carried by the mono micro-labels, the
hairlines and the square geometry, not by the display face alone.

### The trap this section exists for

Replacing `:root` wholesale **deleted the reset block that followed it** —
`box-sizing`, `body { background; color }`, and the `::details-content` fix
whose comment records a real 13px overflow at 390px. Nothing errored; the page
just rendered white-on-transparent and every `<details>` silently reverted to
`content-box`. A token swap is not a self-contained edit: check what sat
*between* the block you replaced and the next one.

## 23. The form changes shape with the workflow

A control the current model cannot use **leaves the screen**. Not greyed out —
a disabled control still costs a line of screen and a moment's reading, and on
iOS a disabled `<select>` cannot even be opened, so "unavailable here" and
"empty" look identical from a phone.

### One source of truth

`/api/config` carries a `capabilities` map, **composed** by the server from the
feature blocks it already emits (`thinking`, `auto_duration`, `takes`,
`variants`, `loras`, `quality` each carry an `applies_to`; the backend carries
`supports_cover` / `supports_instrumental` / `requires_lyrics`). It is never
restated by hand — a second list drifts the first time a feature changes hands.

The client has exactly one reader, `supports(feature, model)`, and feature names
are the config's own block names (`loras`, not `lora`) so there is no
translation layer to get wrong. It falls back to reading `applies_to` directly,
which keeps an older server or an out-of-date mock working.

`requires` records features that are meaningless without another: automatic
duration is gated on `thinking`, because without the LM there is nothing to
make the decision and `-1` is a flat 120 s.

### Capability is per model AND per checkpoint

`variants.items[].supports` comes from ACE-Step's own `constants.py`, not from
this project's memory:

| | text2music | repaint | cover | extract | lego | complete | guidance |
|---|---|---|---|---|---|---|---|
| turbo, turbo-continuous | ✓ | ✓ | ✓ | | | | |
| base, sft | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Turbo can cover.** The variant label used to imply otherwise; what is
base-only is `complete` (continue a track), `extract` (stems), `lego` (layers)
and real CFG. Anything gated on one of those has to consult the active
checkpoint, not just the model.

### Grouping

`MODE → COVER SOURCE → ENGINE → SONG → OUTPUT → ADVANCED`. Checkpoint and
adapter live together under **Engine** because they are one decision — which
weights run. A group whose every child is hidden hides itself, or its label
floats above nothing.

The cover source sits directly under the mode switch: while covering, the
source is the subject of the form and the engine settings are secondary to it.

### Two rules that pull against each other

- **hidden ≠ sent.** The form persists LoRA, checkpoint and thinking in its v2
  blob. Under a model that cannot use them they must not reach the POST body.
  Verified at the wire, not in the DOM: intercepted the request and asserted
  the keys are *absent*
- **hidden ≠ forgotten.** The same values must survive being hidden, so
  switching back restores the selection. Verified: adapter, checkpoint and
  thinking all round-trip through MiniMax unchanged

Both together are the whole risk of capability gating, and either one alone
looks like success.

### The mode switch is deliberately NOT gated

`Cover an existing track` stays visible on a model that cannot cover, because
**turning it on is what pins the model** — gating it would hide the only way in
from Create, leaving the Library's "Cover" action as the sole entry point. It
keeps its `(ACE-Step)` qualifier for that reason; every other qualifier was
removed, because once visibility enforces the rule the parenthetical is a lie.

### What this removed

Picking a LoRA on MiniMax used to switch the model for you. That entry point is
gone by design — the select is no longer there to pick from. The handler
survives as a defensive path, because a restored form blob or a Reuse can still
carry an adapter while MiniMax is active. `probe_ui.py` asserted the old
behaviour and was updated to the new spec rather than left to fail.

Discoverability lost by hiding is repaid by `#c-caps`, a line under the model
select naming what that model adds and what it gives up — hiding a control
otherwise removes the evidence it ever existed.

## 24. An engine with a score stage (YuE2)

YuE2 writes an ABC score — melody, chords, structure, tempo — before it sings,
and that score is text. The Create pane treats it as a first-class artefact:

- A **Score** group appears only for an engine whose capability map says
  `score`. It holds the plan mode (full / melody only / no plan), an ABC
  textarea, **Plan score only** and **Clear score**. Steps and duration leave
  with the engines that have them (`steps`, `duration` capabilities).
- **Plan score only** runs the first stage alone and shows the score in the
  results column with one button, *Edit this score*, that puts it in the
  textarea. Nothing lands in the library. Generate then sings to the edited
  score; the result card shows *Score it sang to* with the same button.
- A pasted score with **No plan** is warned about in the form and refused by
  the server; the form never sends what the engine would ignore.
- The sidecar records the score that was in the forward pass and the plan
  mode; **Reuse** restores both. A record from another engine clears the box.
- An engine this box cannot run (venv missing, weights absent, card too
  small) stays in the model list **disabled, with the reason on the option** —
  hiding it would hide the reason to buy the bigger card. This is the one
  exception to §23's "leaves the screen": there is nothing to open, and the
  reason is the content.

Verified here against a stub pipeline (`probe_score.py`, `probe_yue2_live.py`);
not yet against the model, which needs a 24 GB card.

## Acceptance test

At **390×844** and **360×740**:

- no horizontal scroll (`scrollWidth <= clientWidth`)
- ≥ 5 library rows visible without scrolling
- every control ≥ 44×44 px
- the dock is fixed, ≤ 64 px + inset, and never covers the last row
- `edit` and `more` open their UI and do **not** start playback
- unplayed tracks are visibly distinct from played ones
- list view is what loads on first visit
- Advanced is collapsed by default, and the preset it picks matches the
  detected card
- the cover controls are invisible until cover mode is on, and once on it is
  obvious at a glance which track is being covered
- with jobs queued and one running: force reset cancels all of them, every one
  ends `cancelled`, and the GPU reading drops — verified 2026-08-14 with the
  `STUDIO_TEST_HOOKS` synthetic lane
- a running upscale, cancelled mid-run, ends `cancelled` (not `error`) and
  leaves no orphaned python holding the card — verified 2026-08-14
- all six writer target×mode combinations complete and hold their structure;
  a lyric edit naming a section returns every other section byte-identical —
  `web/scratchpad/test_write_api.py`, 29/29, 2026-08-15
- the library shows a workspace picker, playlist chips and a trash mode; the
  trash lists every binned item with a title where a sidecar survives and a
  filename where it does not — verified in Chromium 2026-08-20: 65 rows, 65
  Restore buttons, first row "Our Paths Diverge, Our Destinies Entwined"
- with a render holding the GPU, a write routes to the CPU and finishes
  *before* the render does — `web/scratchpad/test_write_concurrent.py`, 5/5,
  2026-08-15
