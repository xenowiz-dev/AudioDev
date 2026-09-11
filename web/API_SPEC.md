# AudioDev Studio — HTTP/SSE API Contract (v1)

The replacement for `B:\AudioDev\studio\app.py`. Two agents build against this
document alone: a **backend agent** (FastAPI/uvicorn under the studio venv) and a
**frontend agent** (vanilla ES modules, no build step). Nothing here requires
reading `app.py`.

The old gradio app stays alive and untouched on **:7861** during migration.

---

## 0. Ground rules

| | |
|---|---|
| New code root | `B:\AudioDev\web\` |
| Server | `B:\AudioDev\studio\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 7862` |
| Backend file | `B:\AudioDev\web\server.py` (+ modules); `sys.path.insert(0, r"B:\AudioDev\studio")` and `sys.path.insert(0, r"B:\AudioDev\minimax")` before importing the portable modules |
| Static root | `B:\AudioDev\web\static\` — `index.html`, `app.js`, `app.css`, `manifest.webmanifest`, `icon.png` |
| Bind | `127.0.0.1:7862` **only**. `tailscale serve --bg 7862` proxies it. Never assume a public origin, never emit absolute URLs with a host. |
| Auth | none. Same-origin. No CORS headers needed or wanted. |
| Output dir (`OUTDIR`) | `B:\AudioDev\Music\studio` |
| Derived dirs | `OUTDIR\upscaled`, `OUTDIR\covers`, `OUTDIR\trash`, `OUTDIR\uploads` (new) |
| Assets | `B:\AudioDev\studio\assets` (`icon.png`, `phone_qr.png`, `phone_url.txt`) |
| Palette | Per the supplied reference design, and every accent has a JOB: ground `#07060E`, rule `#241E44`, text `#C9C4FF`; violet `#7A5CFF` = chrome/Create, rose `#FF5CC8` = the music/Library, cyan `#35E6E0` = the machine/Restore. `--accent` is re-pointed per pane. Dark only. See REQUIREMENTS 22. |
| PWA | serve `manifest.webmanifest` with `"background_color": "#07060E"`, `"theme_color": "#07060E"`, `display: standalone`, and the assets icon — parity with gradio's `pwa=True`. |

On process shutdown the server **must** call `SUP.unload()` in a `finally` /
lifespan-shutdown hook, exactly as `app.py` does. A worker left alive holds 8 GB.

### Conventions

* All request and response bodies are `application/json; charset=utf-8` unless
  stated (media, uploads, SSE).
* Windows paths appear in JSON as ordinary strings with escaped backslashes
  (`"B:\\AudioDev\\Music\\studio\\x.wav"`). Never normalise them to forward
  slashes in JSON — they are passed straight back to the portable modules.
* Numbers are JSON numbers. `null` means "absent / unknown", never `0`.
* Booleans are real booleans.

### Identity

* **A library track is identified by its basename** (`"20260814-120301_minimax.wav"`).
  `library.entries()` globs the top level of one folder, so basenames are unique
  there. Call this field `id` everywhere. URL-encode it in paths
  (`/api/library/20260814-120301_minimax.wav`).
* **An arbitrary input file** (Restore tab, cover source, upload) is identified
  by its **absolute path string**, in a JSON field called `path`. These are not
  library members and have no `id`.
* The server resolves `id` → path as `os.path.join(OUTDIR, id)` after rejecting
  any `id` containing a path separator, `..`, or a drive letter.

### Error envelope

Every non-2xx response body:

```json
{"error": {"code": "no_record", "message": "That track has no saved prompt.", "detail": null}}
```

| HTTP | `code` values used | Meaning |
|---|---|---|
| 400 | `bad_request`, `validation` | Malformed or invalid body |
| 404 | `not_found` | Unknown track / job / playlist / file |
| 409 | `gpu_busy`, `no_record`, `not_cancellable`, `exists`, `move_failed` | Valid request, wrong state |
| 413 | `too_large` | Upload over limit |
| 422 | `unmeasurable` | Tool ran but produced no usable answer (e.g. no cliff) |
| 500 | `internal` | Unexpected |

**A failing long-running job is NOT an HTTP error.** `WorkerDied`, an upscaler's
non-zero exit, a dead transcriber — all surface as an SSE `error` event on the
job stream with the job ending in state `error`. The POST that created the job
already returned `202`.

---

## 1. GPU serialization

Exactly one model may be resident (`Supervisor` enforces this internally by
killing the previous worker), and exactly one GPU job may run at a time (the old
`concurrency_id="gpu"`).

The backend maintains **one `GPU_QUEUE`**: a single background worker thread
draining a FIFO. Jobs of kind `generate`, `upscale`, `transcribe` go on it.
`max depth 4`; a 5th create returns `409 gpu_busy`.

Everything else runs on a separate unbounded thread pool and **must never touch
the GPU queue** — these are CPU subprocesses in other venvs and are explicitly
allowed to run *during* a generation:

`pp.bandwidth`, `pp.cliff_of`, `pp.specview`, `pp.spectrogram`, `pp.region_fill`,
`ly.probe`, `art.ensure_cover`, `llm_title.suggest`, everything in `library.py`,
`titles.py`, `playlists.py`.

`llm_title.suggest` is CPU-only (`device="cpu"`) — do not put retitle on the GPU
queue.

### Unload and the RLock trap

`Supervisor.generate()` holds `SUP.lock` (an `RLock`) for the **entire**
generation. Therefore:

* `POST /api/gpu/unload` calls `SUP.unload()`, which takes the same lock and
  would block for minutes. **The endpoint must return `409 gpu_busy` when a GPU
  job is in state `running` or `starting`**, and only call `SUP.unload()` when
  the queue is idle.
* **Cancel must not go through `SUP.unload()`** for the same reason. See §2.5.

---

## 2. Jobs and SSE

### 2.1 Why POST-then-stream

Job creation and the event stream are **separate requests** on purpose. The
driver is a phone over Tailscale: the screen locks, the radio drops, the tab is
backgrounded. The job must survive the client vanishing, and the client must be
able to re-attach and catch up. Streaming the POST response would tie the job's
life to one socket.

```
POST /api/generate            -> 202 {"job_id": "...", "stream_url": "/api/jobs/<id>/events"}
GET  /api/jobs/<id>/events    -> text/event-stream (re-attachable, replayable)
GET  /api/jobs/<id>           -> JSON snapshot (poll fallback if SSE is blocked)
DELETE /api/jobs/<id>         -> cancel
```

### 2.2 Job object (the snapshot shape)

Returned by `GET /api/jobs/<id>` and carried as the `data` of the `hello` event.

```json
{
  "id": "j_7f3a1c",
  "kind": "generate",
  "state": "running",
  "created": 1755180000.12,
  "started": 1755180002.44,
  "ended": null,
  "queue_position": null,
  "seq": 184,
  "log": ["[    0s] starting MiniMax Music 3 (first load takes ~20 s)", "..."],
  "live": {"msg": "decoding 40%", "stage": "decode", "frac": 0.4},
  "artifacts": [
    {"kind": "audio", "label": "generated", "path": "B:\\...\\x.wav", "url": "/api/media?p=..."}
  ],
  "result": null,
  "error": null,
  "vram": {"used_gb": 8.4, "total_gb": 10.0, "free_gb": 1.6, "loaded": "minimax"},
  "request": {"model": "minimax", "duration": 30.0, "...": "the create body, echoed"}
}
```

`state` ∈ `queued` | `starting` | `running` | `done` | `error` | `cancelled`.
Terminal states: `done`, `error`, `cancelled`.

`log` is a ring buffer of the last **1000** flushed lines. `live` is the current
replaced-in-place progress line (see §2.4) or `null`.

### 2.3 SSE wire format

`Content-Type: text/event-stream; charset=utf-8`
`Cache-Control: no-cache, no-store`
`Connection: keep-alive`
`X-Accel-Buffering: no`   ← required; some proxies buffer otherwise
**No gzip/deflate on this route.** Flush after every event.

Every event carries a monotonic `id:` equal to `data.seq`.

```
id: 41
event: progress
data: {"seq":41,"msg":"loading model","stage":null,"frac":null,"t":1.9}

id: 42
event: progress
data: {"seq":42,"msg":"decoding 40%","stage":"decode","frac":0.4,"t":12.3}

: ping
```

A `: ping` comment or an `event: ping` every **15 s** keeps intermediaries from
closing an idle stream.

### 2.4 Event catalogue

| event | when | `data` |
|---|---|---|
| `hello` | first event on a fresh attach (and after a replay gap) | the full job object of §2.2 |
| `queued` | while waiting for the GPU queue | `{"seq":n,"position":1}` |
| `progress` | every `on_progress` callback | `{"seq":n,"msg":str,"stage":str\|null,"frac":number\|null,"t":secondsSinceStart}` |
| `vram` | every ~5 s while running, and once at each phase change | `{"seq":n,"used_gb":8.4,"total_gb":10.0,"free_gb":1.6,"loaded":"minimax"}` |
| `artifact` | a file became available mid-job | `{"seq":n,"kind":"audio"\|"image","label":str,"path":str,"url":str}` |
| `phase` | multi-phase jobs crossing a boundary | `{"seq":n,"phase":"generate"\|"post"\|"upscale"\|"splice"\|"compare","label":"post-processing — Apollo …"}` |
| `result` | terminal success | job-kind-specific, see each endpoint |
| `error` | terminal failure | `{"seq":n,"code":"worker_died","message":"…"}` |
| `cancelled` | terminal cancel | `{"seq":n}` |
| `ping` | keepalive | `{"t":1755180000.0}` |

After a terminal event the server sends nothing more and closes the stream. The
client must **not** let `EventSource` auto-reconnect after a terminal event —
call `es.close()` in the handlers for `result`, `error`, `cancelled`.

### 2.5 Relaying `Supervisor.on_progress`

`SUP.generate(name, params, on_progress=...)` calls `on_progress` with a **dict**
`{"msg": str, "stage": str|None, "frac": float|None}` from a worker thread.
`pp.upscale`, `pp.region_fill` and `ly.transcribe` call `on_progress` with a
**plain string** (already truncated to 160 chars by those modules).

The backend normalises both into one `progress` event:

```python
def emit(m):                       # m is dict OR str
    if isinstance(m, str):
        m = {"msg": m}
    job.push("progress", {"msg": m.get("msg", ""),
                          "stage": m.get("stage"),
                          "frac": m.get("frac"),
                          "t": time.time() - job.started})
```

`on_progress` runs on the worker-reading thread. Pushing an event must be a
non-blocking append to the job's buffer plus a `threading.Event.set()` / queue
put per attached stream. It must never block on a slow HTTP client — a slow
phone must not stall the GPU. Per-subscriber queues are bounded (256); a
subscriber that overflows is dropped and forced to reconnect (it will get a
`hello` snapshot and lose nothing).

### 2.6 The log rendering rule (frontend, verbatim)

This reproduces the old log pane exactly. On each `progress` event:

```
if frac === null:
    if live: log.push(live.text); live = null
    log.push(pad(t) + msg)                 # pad = "[%5.0fs] "
else:
    if live && live.stage !== stage: log.push(live.text)
    live = { text: pad(t) + msg, stage: stage, frac: frac }
```

On a terminal event: `if (live) { log.push(live.text); live = null }`.

Rationale: a percentage line **replaces** the previous one instead of adding a
hundred near-identical rows, but each *stage* keeps its last line. Render `live`
as a progress bar (amber fill) below the log, not as a log row.

### 2.7 Reconnect

`EventSource` sends `Last-Event-ID` automatically on reconnect.

* No `Last-Event-ID` → send `hello` (full snapshot), then live events.
* `Last-Event-ID = k`, and `k` is still inside the job's **2000-event** buffer →
  replay every buffered event with `seq > k`, then live events. **Do not** send
  `hello`.
* `Last-Event-ID = k` older than the buffer floor → send `hello` first (the
  client replaces its whole job state), then replay from the floor.
* Unknown job id → `404`.
* Job already terminal → send `hello` (which contains `result`/`error`), then the
  terminal event, then close. This is how a phone that slept through a
  three-minute generation learns it succeeded.

Terminal jobs are retained in memory for **30 minutes**, then evicted.

### 2.8 Cancel — `DELETE /api/jobs/<id>`

| kind | cancellable | mechanism |
|---|---|---|
| `generate` | **yes** | set `job.cancel_requested = True`, then kill the worker process directly: `p = getattr(SUP.worker, "proc", None); p and p.kill()`. Closing the worker's stdout makes `Worker._read_until` raise `WorkerDied`, which propagates out of `SUP.generate` and **releases `SUP.lock`**. Then call `SUP.unload()` (now safe) to reap. Report the job as `cancelled`, not `error`. |
| `upscale` | no | `409 not_cancellable` |
| `transcribe` | no | `409 not_cancellable` |

**Never** call `SUP.unload()` to cancel — it takes the same RLock the running
generation holds and blocks until the generation finishes.

**Edge case, must be handled:** during `Supervisor.ensure()` the new worker is
only assigned to `SUP.worker` *after* `Worker.start()` returns (~20 s of model
loading), so for that window there is no handle to kill. Cancel therefore always
sets `cancel_requested` first; the job loop checks it at every phase boundary and
the killer retries the handle grab for up to 30 s. A cancel during loading may
take until "ready" to take effect — report `state: "cancelling"` is NOT a state;
keep the job `running` and let the client show a disabled Cancel button until the
terminal event arrives.

A cancel on a `queued` job removes it from the queue immediately → `cancelled`.

Response: `202 {"cancelling": true}` or `409`.

### 2.9 Job listing

`GET /api/jobs?active=1` → `{"jobs": [<job object>, ...]}` newest first.
`active=1` filters to non-terminal. Used on page load so a phone that reopens the
app re-attaches to whatever is running.

---

## 3. Endpoint overview

| # | Method + path | Long-running | GPU | Delegates to |
|---|---|---|---|---|
| 1 | `GET /api/config` | instant | no | `BACKENDS`, `pp.UPSCALERS/DEGRADES/NATURALIZERS`, `ly.WHISPER_MODELS`, `llm_title.is_ready/MODEL_ID`, `wr.is_ready/model_id` |
| 2 | `GET /api/health` | instant | no | — |
| 3 | `GET /api/vram` | instant (~200 ms) | no | `supervisor.gpu_memory`, `SUP.current` |
| 4 | `GET /api/budget` | instant | no | `ar_cache.budget`, `gpu_memory` |
| 5 | `POST /api/gpu/unload` | instant | serialized | `SUP.unload` |
| 6 | `POST /api/generate` | **STREAM** | **yes** | `SUP.generate`, `lib.write`, `art.ensure_cover`, `pp.upscale`, `pp.bandwidth` |
| 7 | `POST /api/upscale` | **STREAM** | **yes** | `pp.upscale`, `pp.bandwidth`, `pp.region_fill`, `pp.spectrogram` |
| 8 | `POST /api/lyrics/transcribe` | **STREAM** | **yes** | `ly.transcribe` |
| 8b | `POST /api/write` | **STREAM** | **when idle** | `wr.write` → `songwriter.py` |
| 8c | `GET /api/loras` | instant | no | `lo.scan` (filesystem only, no torch) |
| 9 | `POST /api/lyrics/probe` | instant (~1 s) | no | `ly.probe`, `ly.describe` |
| 10 | `POST /api/restore/inspect` | slow-sync (1–8 s) | no | `pp.bandwidth`, `pp.specview` |
| 11 | `POST /api/restore/cliff` | slow-sync (1–3 s) | no | `pp.cliff_of` (+ `pp.bandwidth`, `pp.specview`) |
| 12 | `GET /api/library` | instant | no | `lib.entries`, `pls.tracks/is_favorite/names`, `titles.derive`, `art.cover_path/ensure_cover` |
| 13 | `GET /api/library/{id}` | instant | no | `lib.read`, `lib.detail`, `titles.derive` |
| 14 | `PATCH /api/library/{id}` | instant | no | `lib.update` |
| 15 | `POST /api/library/{id}/retitle` | slow-sync (0–30 s) | no | `llm_title.suggest`, `titles.derive`, `lib.update` |
| 16 | `PUT /api/library/{id}/favorite` | instant | no | `pls.set_favorite` |
| 17 | `PUT /api/library/{id}/rating` | instant | no | `lib.update`, `pls.set_favorite` |
| 18 | `POST /api/library/{id}/trash` | instant | no | `pls.trash` |
| 19 | `GET /api/library/{id}/reuse` | instant | no | `lib.read` |
| 20 | `GET /api/library/{id}/cover-source` | instant (~1 s) | no | `lib.read`, `ly.probe/describe`, `titles.derive` |
| 21 | `POST /api/library/covers` | instant (kicks bg) | no | `art.ensure_cover` |
| 22 | `GET /api/playlists` | instant | no | `pls.load/names` |
| 23 | `POST /api/playlists` | instant | no | `pls.create` |
| 24 | `DELETE /api/playlists/{name}` | instant | no | `pls.delete_playlist` |
| 25 | `GET /api/playlists/{name}` | instant | no | `pls.tracks` |
| 26 | `POST /api/playlists/{name}/tracks` | instant | no | `pls.create`, `pls.add` |
| 27 | `DELETE /api/playlists/{name}/tracks/{id}` | instant | no | `pls.remove` |
| 28 | `POST /api/playlists/{name}/reorder` | instant | no | `pls.reorder` |
| 29 | `GET /api/trash` | instant | no | `pls.trashed` |
| 30 | `POST /api/trash/{name}/restore` | instant | no | `pls.restore` |
| 31 | `GET /api/media` | streaming bytes | no | filesystem |
| 32 | `POST /api/upload` | instant | no | filesystem |
| 33 | `GET /api/fs/stat` | instant | no | filesystem |
| 34 | `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/events`, `DELETE /api/jobs/{id}` | see §2 | — | job registry |

---

## 4. Config and status

### 4.1 `GET /api/config`

Instant. No GPU. The frontend must hold **no hardcoded copy** of backends,
upscalers or defaults — they all come from here, once, on boot.

```json
{
  "version": 1,
  "outdir": "B:\\AudioDev\\Music\\studio",
  "default_model": "minimax",
  "backends": {
    "minimax": {
      "label": "MiniMax Music 3",
      "note": "44.1 kHz stereo, sung lyrics. 4-bit LLM. ~7x realtime.",
      "vram_gb": 8.0,
      "requires_lyrics": true,
      "supports_instrumental": false,
      "supports_cover": false,
      "default_steps": 30,
      "default_prompt": "Global Metadata\nBasic Attributes: bpm is 96. …",
      "prompt_lines": 10,
      "prompt_info": "MiniMax was trained on sectioned captions — keep the Global Metadata / Vocal Details / Arrangement headings; short prompts lose arrangement control.",
      "lyrics_info": "Required — MiniMax has no instrumental mode."
    },
    "acestep": {
      "label": "ACE-Step 1.5",
      "note": "48 kHz. Fast, and the only one here that can cover a track.",
      "vram_gb": 6.0,
      "requires_lyrics": false,
      "supports_instrumental": true,
      "supports_cover": true,
      "default_steps": 8,
      "default_prompt": "upbeat indie pop, jangly electric guitar, live drums, warm analog production, 110 bpm",
      "prompt_lines": 4,
      "prompt_info": "Plain keyword-style prompt.",
      "lyrics_info": "Leave blank, or tick Instrumental above."
    }
  },
  "default_lyrics": "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe",
  "upscalers": {
    "none": "None",
    "apollo": "Apollo — fast, trained on codec artifacts (7.00 dB)",
    "audiosr": "AudioSR — most accurate, ~10x slower (4.37 dB)",
    "flashsr": "FlashSR — fastest, least accurate (11.34 dB)"
  },
  "degrades": {
    "off": "Off — upscale the file exactly as it is",
    "auto": "Auto — the damage this upscaler was trained on",
    "cut14": "Cut above 14 kHz — gentle, leaves most of the air",
    "cut11": "Cut above 11 kHz — a big band to rebuild",
    "mp3": "MP3 128k round-trip — real codec artifacts",
    "both": "MP3 128k + cut above 11 kHz — the most to rebuild"
  },
  "naturalizers": {
    "off": "Off", "subtle": "Subtle — …", "medium": "Medium — …", "strong": "Strong — …"
  },
  "fingerprint_note": "AudioSR and FlashSR stamp a 100 Hz neural-vocoder comb on their output (hop 480 @ 48 kHz, confirmed in their source) — upscaling adds a detectable AI fingerprint that was not there before.",
  "whisper_models": ["large-v3", "medium", "small"],
  "writer": {
    "ready": true,
    "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
    "targets": ["lyrics", "style"],
    "modes": ["generate", "extend", "edit"],
    "mode_info": {"generate": "…", "extend": "…", "edit": "…"},
    "note": "Takes the GPU when it is idle (~30 s for a full lyric) and falls back to the CPU when it is not…"
  },
  "defaults": {
    "duration": 30, "steps": 30, "seed": 7, "instrumental": false,
    "post_kind": "none", "cover_strength": 1.0, "noise_strength": 0.75,
    "upscaler": "apollo", "auto_lowpass": true, "fill": true,
    "degrade": "off", "naturalize": "off", "lufs": -14.0,
    "region": {"t0": 0, "t1": 0, "flo": 16000, "fhi": 0},
    "whisper_model": "large-v3", "separate": true, "seconds": 0
  },
  "limits": {
    "duration": [10, 300, 5],
    "steps": [4, 60, 1],
    "cover_strength": [0.0, 1.0, 0.05],
    "noise_strength": [0.0, 1.0, 0.05],
    "preview_seconds": 15,
    "upload_mb": 200
  },
  "llm_title": {"ready": true, "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "hint": "Qwen/Qwen2.5-1.5B-Instruct on CPU, ~9 s. Off uses the instant heuristic."},
  "phone": {"url": "https://box.tailnet.ts.net/", "qr_url": "/api/media?p=B%3A%5CAudioDev%5Cstudio%5Cassets%5Cphone_qr.png"},
  "notes": {
    "post_kind": "Runs after generation. Measured: neither generator leaves a codec cliff (MiniMax 0.6 dB, ACE-Step 3.9 dB), and upresing audio with no cliff makes it worse — so this is normally for lossy source material, on the Restore tab. The bandwidth verdict is logged either way.",
    "cover": "Upload a track to regenerate it in the style above. Noise strength is the parameter that makes it a cover — 0.0 means pure noise and produces an unrelated song. 0.4–0.8.",
    "auto_lowpass": "AudioSR was trained on lowpass-filtered audio only; fed a raw lossy file it hallucinates from codec artifacts. Leave on for anything compressed.",
    "region_fill": "An upscaler rewrites the whole file (~13% relative error below the crossover); splicing adds only the masked correction, so everything outside the box stays bit-identical.",
    "sidecar": "Each track keeps a JSON sidecar holding its prompt, lyrics and settings, so the record travels with the file. Not written into the audio's tags. Cover art is rendered from each track's own spectrogram. Trash is a folder move, never a delete."
  }
}
```

`phone` is `null` when `assets\phone_url.txt` is absent; `phone.qr_url` is `null`
when `assets\phone_qr.png` is absent.

`llm_title.ready` comes from `llm_title.is_ready()` (a pure filesystem walk, no
network, safe to call per request — but cache it for 60 s).

There is **no `ALL_TRACKS` sentinel**. "All tracks" is `playlist: null`.

### 4.2 `GET /api/health`

`{"ok": true, "uptime_s": 1234.5, "jobs_active": 0}`

### 4.3 `GET /api/vram`

Instant (~200 ms — one `nvidia-smi` subprocess). No GPU compute. Replaces the
`every=8` markdown pill.

```json
{
  "available": true,
  "used_gb": 2.1, "total_gb": 10.0, "free_gb": 7.9,
  "loaded": "minimax", "loaded_label": "MiniMax Music 3",
  "warn_low_free": false
}
```

* `available: false` when `gpu_memory()` returns `(None, None)`; then all three
  numbers are `null`. Frontend renders "GPU: unavailable".
* `loaded` is `SUP.current()` (`null` when nothing is resident);
  `loaded_label` is `BACKENDS[loaded]["label"]`.
* `warn_low_free` is `true` iff `loaded is None and free_gb < 8.0`. Frontend text:
  "⚠ under 8 GB free — MiniMax will spill to system memory and run several times
  slower".

**Polling:** the client polls this every 8 s **only while the document is
visible** (`document.visibilityState === "visible"`); pause on `visibilitychange`
to spare the phone's battery and the tailnet. While a job stream is attached, the
stream's `vram` events supersede polling — stop polling for the duration.

### 4.4 `GET /api/budget?model=minimax&duration=180`

Instant. Pure arithmetic plus one `nvidia-smi`. Replaces the fit-note under the
duration slider.

```json
{"applies": true, "need_gb": 8.71, "free_gb": 7.90, "headroom_gb": -0.81, "fits": false}
```

* `model != "minimax"` → `{"applies": false, "reason": "model"}` (the AR-cache
  maths is MiniMax-specific).
* GPU unavailable → `{"applies": false, "reason": "gpu_unavailable"}`.
* Delegates `ar_cache.budget(float(duration), free_gb)` → `(need, fits, headroom)`
  with the module defaults (`base_gb=7.49`, `prompt_tokens=1000`,
  `workspace_gb=0.4`, `kv_bits=16`).

Frontend copy, preserved:
* fits → "needs ~8.7 GB, 9.1 GB free — fits with 0.4 GB spare"
* not → "⚠ needs ~8.7 GB but only 7.9 GB is free (0.8 GB short) — this will spill
  to system memory and run several times slower. Close browsers/ComfyUI, or
  shorten it."

Call on model change, on duration change (debounced 250 ms), and once on load.

### 4.5 `POST /api/gpu/unload`

Serialized against the GPU queue but not itself GPU work.

Request: `{}` (no body required).
Response `200`:
```json
{"unloaded": true, "vram": { …the /api/vram shape… }}
```
Response `409`:
```json
{"error": {"code": "gpu_busy", "message": "A job is running.", "detail": {"job_id": "j_7f3a1c", "kind": "generate"}}}
```

Delegates `SUP.unload()`. See §1 for why the 409 is mandatory rather than
optional.

---

## 4b. Capabilities

`/api/config` carries a `capabilities` map, **composed** from the feature
blocks in the same response rather than restated:

```json
"capabilities": {
  "minimax": {"thinking": false, "auto_duration": false, "takes": false,
              "variants": false, "loras": false, "quality": true,
              "cover": false, "instrumental": false, "lyrics_required": true},
  "acestep": {"thinking": true,  "auto_duration": true,  "takes": true,
              "variants": true,  "loras": true,  "quality": false,
              "cover": true,  "instrumental": true,  "lyrics_required": false}
},
"requires": {"auto_duration": "thinking"}
```

Keys are the config's own block names (`loras`, not `lora`), so a client needs
no translation layer. A feature block gaining an `applies_to` is all it takes
to gate a control; nothing else has to be edited.

`requires` names features that are meaningless unless another is on.

**Per-checkpoint capability** rides on `variants.items[]`, taken from ACE-Step's
`constants.py` (`TASK_TYPES_TURBO` / `TASK_TYPES_BASE`):

```json
{"name": "acestep-v15-base", "tasks": ["text2music","repaint","cover",
                                       "cover-nofsq","extract","lego","complete"],
 "supports": {"cover": true, "repaint": true, "complete": true,
              "extract": true, "lego": true, "guidance": true}}
```

Turbo **can** cover; `complete`, `extract`, `lego` and real CFG are base-only.

The server enforces the same map it publishes — a LoRA sent with MiniMax is
`400 validation`, and a `src_path` is rejected against `supports_cover` rather
than against a hard-coded model name. Hiding a control client-side without the
server refusing it makes the capability map decoration.

---

## 5. Generate / Preview

### 5.1 `POST /api/generate` — LONG-RUNNING, GPU

Replaces both `go.click(generate, …)` and `prev.click(preview, …)`. Preview is
the same endpoint with `"preview": true`.

**Request**

```json
{
  "model": "minimax",
  "prompt": "Global Metadata\n…",
  "lyrics": "[verse]\n…",
  "duration": 30,
  "steps": 30,
  "seed": 7,
  "instrumental": false,
  "post_kind": "none",
  "preview": false,
  "src_path": null,
  "cover_strength": 1.0,
  "noise_strength": 0.75
}
```

**Validation — all `400 validation`, all server-side (never trust the client):**

| rule | message |
|---|---|
| `prompt.strip()` non-empty | `A style description is required.` |
| `model == "minimax"` ⇒ `lyrics.strip()` non-empty | `MiniMax Music 3 requires lyrics — it has no instrumental mode. Use ACE-Step for instrumentals.` |
| `instrumental` true ⇒ `model == "acestep"` | `Instrumental is ACE-Step only.` |
| `model` ∈ `BACKENDS` | `Unknown model.` |
| `post_kind` ∈ `pp.UPSCALERS` | `Unknown upscaler.` |
| `10 ≤ duration ≤ 300`, `4 ≤ steps ≤ 60` | `Out of range.` |
| `src_path` given ⇒ `model == "acestep"` and the file exists | `Cover source is ACE-Step only.` / `No such file.` |

**Normalisation the backend performs**

```python
out_name = f"{datetime.now():%Y%m%d-%H%M%S}_{model}.wav"
if preview:
    duration  = min(float(duration), 15.0)     # PREVIEW_SECONDS
    post_kind = "none"                         # forced
params = dict(prompt=prompt.strip(),
              lyrics="" if instrumental else (lyrics or "").strip(),
              duration=float(duration), steps=int(steps), seed=int(seed),
              instrumental=bool(instrumental), out_name=out_name)
if model == "acestep" and src_path:
    params.update(src_audio=src_path,                 # NOTE the worker's key name
                  cover_strength=float(cover_strength),
                  noise_strength=float(noise_strength))
```

The worker's parameter is `src_audio`, not `src_path`. Do not rename it.

**Response `202`**

```json
{"job_id": "j_7f3a1c", "kind": "generate", "state": "queued",
 "stream_url": "/api/jobs/j_7f3a1c/events", "out_name": "20260814-120301_minimax.wav"}
```

**Job execution, in order**

1. `state = starting`; `emit({"msg": "…"})` for supervisor's own progress
   (`unloading X to free VRAM`, `starting MiniMax Music 3 (first load takes ~20 s)`).
2. `res = SUP.generate(model, params, on_progress=emit)` on the GPU queue thread.
3. On `WorkerDied` or any exception → `error` event, `state = error`, stop. If
   `job.cancel_requested` was set, emit `cancelled` instead.
4. Emit `artifact` `{"kind":"audio","label":"generated","path":res["path"],"url":…}`.
5. Write the sidecar — **a side effect, before any post step**, for the generated
   file (the upscaled derivative lives in `upscaled\` and is reachable via
   `post_kind`):

```python
lib.write(res["path"], model=model, prompt=prompt,
          lyrics=None if instrumental else (lyrics or None),
          instrumental=bool(instrumental), duration=float(duration),
          steps=int(steps), seed=int(seed), post_kind=post_kind,
          preview=bool(preview),
          cover_src=src_path if (model == "acestep" and src_path) else None,
          cover_strength=cover_strength if src_path else None,
          noise_strength=noise_strength if src_path else None,
          seconds=res.get("seconds"), elapsed=res.get("elapsed"),
          sampling_rate=res.get("sampling_rate"),
          vram_peak=res.get("vram_peak"))
```

6. `art.ensure_cover(res["path"])` in a `try/except: pass`. Best-effort: a
   missing cover degrades to a tile with no image, never a failed generation.
7. If `post_kind` not in `(None, "none")`:
   * `phase` event `{"phase":"post","label":"post-processing — <UPSCALERS[post_kind]>"}`
   * emit `pp.bandwidth(final)` as a `progress` line (report the verdict, proceed
     regardless — the user asked for this step)
   * `pp.upscale(post_kind, final, OUTDIR\upscaled, on_progress=emit, auto_lowpass=True)`
   * on success: emit `artifact` `{"kind":"audio","label":"upscaled",…}` and a
     `progress` line with `pp.bandwidth(out)`; `final = out`
   * **on failure: log it and continue to a successful `result`.** `final` keeps
     pointing at the generated file — a bad post step degrades to "you still have
     your generation", it does not fail the job.
8. `result` event; `state = done`.

**`result` payload**

```json
{
  "seq": 210,
  "path": "B:\\AudioDev\\Music\\studio\\20260814-120301_minimax.wav",
  "url": "/api/media?p=…",
  "library_id": "20260814-120301_minimax.wav",
  "final_path": "B:\\AudioDev\\Music\\studio\\upscaled\\20260814-120301_minimax_apollo.wav",
  "final_url": "/api/media?p=…",
  "seconds": 30.2, "elapsed": 214.0, "sampling_rate": 44100,
  "realtime_ratio": 7.09,
  "vram_peak": 8.71, "vram_idle": 0.15,
  "preview": false,
  "post": {"kind": "apollo", "ok": true,
           "bandwidth_before": "…report text…", "bandwidth_after": "…report text…"}
}
```

`realtime_ratio = elapsed / max(seconds, 1e-6)`. The old UI rendered it as
`"{rt:.1f}x realtime"`; keep that label. `post` is `null` when `post_kind` was
`"none"`. `final_path` equals `path` when there was no post step.

**Frontend after `result`:** load `final_url` into the player, and refresh the
library (`GET /api/library`) — the new track's cover may still be rendering, so a
second refresh ~2 s later is worthwhile.

**Preview semantics to preserve in the UI copy:** there is no mid-generation
preview to offer. MiniMax's autoregressive stage emits no audio at all until it
has produced every frame, and that stage is most of the wall time. A 15 s render
at the same seed is the honest substitute — close to the full render's opening,
though not identical, since denoising is chunked differently for a shorter piece.

---

## 6. Restore / Upscale

### 6.1 Region encoding

The UI carries four numbers; the tools take a `t0:t1:flo:fhi` string. **The
server does this conversion**, from the JSON object, exactly:

```python
def region_spec(r):
    t0  = float(r.get("t0") or 0)
    t1  = r.get("t1");  flo = float(r.get("flo") or 0);  fhi = r.get("fhi")
    end = "*" if not t1  or float(t1)  <= 0 else f"{float(t1):.3f}"
    hi  = "*" if not fhi or float(fhi) <= 0 else f"{float(fhi):.0f}"
    return f"{t0:.3f}:{end}:{flo:.0f}:{hi}"
```

`0`/`null` on the high edge means "up to Nyquist"; `0`/`null` on the end time
means "end of file". Both differ per file, so they are passed as `*` and resolved
by the tool, never guessed.

Every request that carries a region uses the same object:
`{"t0": 0, "t1": 0, "flo": 16000, "fhi": 0}`.

### 6.2 `POST /api/restore/inspect` — slow-sync (1–8 s), no GPU

Bandwidth report plus an interactive spectrogram with the selection drawn.
Replaces `inspect()` and the re-render half of `on_click()` / `reset_region()`.

**Request**
```json
{"path": "B:\\AudioDev\\Music\\x.wav",
 "region": {"t0": 0, "t1": 0, "flo": 16000, "fhi": 0},
 "marker": {"t": 12.34, "f": 15800.0}}
```
`marker` is optional/`null`. When present the server passes
`marker=f"{t:.4f}:{f:.1f}"` to `pp.specview` — it draws a crosshair where the
*caller believes* the click landed, so a coordinate-space mismatch shows up
immediately instead of quietly producing wrong regions. **Keep this.**

**Response `200`**
```json
{
  "bandwidth": "…check_bandwidth report text…",
  "image_url": "/api/media?p=B%3A%5C…%5Cupscaled%5C_spec_9f21c4e0.png",
  "geometry": {
    "png": "B:\\AudioDev\\Music\\studio\\upscaled\\_spec_9f21c4e0.png",
    "img_w": 1300, "img_h": 560,
    "x0": 78, "x1": 1248, "y0": 28, "y1": 504,
    "t0": 0.0, "t1": 182.4, "f0": 0.0, "f1": 22050.0,
    "sr": 44100, "full_duration": 182.4, "truncated": false,
    "cliff": 16000.0, "drop": 12.4
  },
  "note": "Click a corner of the region you want to fill.",
  "truncated_note": null
}
```

`geometry` is `pp.specview()`'s return value verbatim — **do not reshape it**.
When `geometry.truncated` is true set
`truncated_note: "showing the first 300s of 412s"` (from `t1` and `full_duration`).

Errors:
* no `path` → `400`.
* `pp.specview` raised → `200` with `image_url: null`, `geometry: null`, and
  `bandwidth` still populated plus `"(spectrogram failed: …)"` appended. A broken
  render must not hide the bandwidth verdict.

**Render filenames.** The old code cycled six fixed names (`_spec_view0..5`) to
defeat gradio's path cache. Replace with **a fresh unique name per render**:
`OUTDIR\upscaled\_spec_<uuid4hex[:12]>.png`, served with `Cache-Control: no-store`.
Two devices rendering at once would collide on a fixed cycle. The backend deletes
`_spec_*.png` files older than 30 minutes on a timer.

**Frontend obligations:**
* Debounce the four number inputs by **400 ms** and abort the in-flight request
  with `AbortController` before issuing the next. The old app fired one render per
  keystroke.
* Server-side single-flight per `path`: a second inspect for the same path while
  one is running cancels/joins rather than spawning a second matplotlib process.

### 6.3 Click-to-select is client-side arithmetic

The two-click corner selection does **not** need a server round trip to compute
coordinates. `geometry` publishes the exact plot rectangle in image pixels.

```js
// evt on the <img>; rect = img.getBoundingClientRect()
const sx = g.img_w / rect.width, sy = g.img_h / rect.height;
const x = (evt.clientX - rect.left) * sx, y = (evt.clientY - rect.top) * sy;
if (!(g.x0 <= x && x <= g.x1 && g.y0 <= y && y <= g.y1))
    return status("Click inside the spectrogram (not the margins).");   // ignore, never clamp
const t = g.t0 + (x - g.x0) / (g.x1 - g.x0) * (g.t1 - g.t0);
const f = g.f0 + (g.y1 - y) / (g.y1 - g.y0) * (g.f1 - g.f0);   // image y grows down
```

* Outside the plot rectangle: **ignore rather than clamp** — a clamped margin
  click silently produces a degenerate region.
* First click stores `anchor = {t, f}` (client state) and re-renders with
  `marker`, status `Anchor at 12.34s / 15.80 kHz — click the opposite corner.`
* Second click sets `t0,t1 = sorted(anchor.t, t)`, `flo,fhi = sorted(anchor.f, f)`
  (rounded: times to 3 dp, freqs to whole Hz), clears the anchor, re-renders with
  the region and the marker, status
  `Region set: 12.34–48.10s, 15.80–22.05 kHz. Click again to start a new one.`
* **Reset region** = `anchor = null`, `{t0: 0, t1: 0, flo: geometry.cliff || 16000, fhi: 0}`,
  then re-render. Status `Reset to whole file above 16.00 kHz.` This is the
  `band_splice` default: whole file, cliff upward.
* Touch: a `pointerdown`/`pointerup` pair under 10 px of movement counts as a
  click; a drag is a two-corner selection in one gesture (nice-to-have, same
  maths). Targets on the image are the whole image, so the 44 px rule does not
  apply, but the ± nudge buttons for the four numbers must be ≥ 44 px.

### 6.4 `POST /api/restore/cliff` — slow-sync (1–3 s), no GPU

**Request** `{"path": "B:\\…\\x.wav", "render": true, "region": {…}}`

`render` (default `false`): when `true`, the response also carries the full
`/api/restore/inspect` payload rendered with `flo` set to the detected cliff —
one round trip instead of two, which matters over Tailscale.

**Response `200`**
```json
{"cliff_hz": 16000.0, "drop_db": 12.4, "weak": false,
 "message": null,
 "inspect": { …the §6.2 response… }}
```

* `weak` is `true` when `drop_db is not None and drop_db < 8`. Then
  `message: "Cliff 16000 Hz but only 5.2 dB — that is not a real codec cliff, so there may be nothing to restore."`
  This is advisory; the value is still returned and still applied.
* No measurable cliff (`pp.cliff_of` → `(None, None)`) → `422 unmeasurable`,
  `"Could not measure a cliff for that file."`
* Missing `path` → `400`, `"Pick a file first."`

Delegates `pp.cliff_of(path)`.

### 6.5 `POST /api/upscale` — LONG-RUNNING, GPU

Replaces `run_upscale()`.

**Request**
```json
{"path": "B:\\AudioDev\\Music\\source.wav",
 "kind": "apollo",
 "auto_lowpass": true,
 "fill": true,
 "region": {"t0": 0, "t1": 0, "flo": 16000, "fhi": 0},
 "compare": true}
```

`kind` ∈ `apollo` | `audiosr` | `flashsr` (**not** `none` — 400 if `none`).
`auto_lowpass` only affects AudioSR. `compare` (default `true`) controls whether
the before/after figure is rendered.

**Response `202`** — same shape as §5.1.

**Job execution**

1. `progress`: `--- <UPSCALERS[kind]> ---`, then `pp.bandwidth(src)`.
2. `phase` `upscale`; `out = pp.upscale(kind, src, OUTDIR\upscaled, on_progress=emit, auto_lowpass=…)`.
   * failure → `error` event `{"code": "upscale_failed", "message": "…"}`, job ends.
3. `artifact` `{"kind":"audio","label":"upscaled","path":out,…}`; `progress` with
   `pp.bandwidth(out)`.
4. If `fill`:
   * `phase` `splice`; `hybrid = OUTDIR\upscaled\<stem(out)>_hybrid.wav`
   * `pp.region_fill(src, out, hybrid, [region_spec], on_progress=emit)`
   * success → `out = hybrid`, `artifact` `{"label":"hybrid",…}`
   * **failure → log `splice FAILED: <Type>: <msg>` and continue.** The upscaled
     file is still a result.
5. If `compare`:
   * `phase` `compare`;
     `panels = [("1. INPUT", src), ("2. <KIND> (whole file)", upscaled)]`
     plus `("3. HYBRID (spliced)", hybrid)` when the splice succeeded.
   * `pp.spectrogram(panels, OUTDIR\upscaled\_spec_<uuid>.png, regions=[region_spec], title="Before / after", subtitle="Box = the selected region. On the hybrid, everything outside it is bit-identical to the input.")`
   * success → `artifact` `{"kind":"image","label":"compare",…}`; failure → log
     `spectrogram failed: …` and continue.
6. `result`.

**`result` payload**
```json
{"seq": 302,
 "path": "B:\\…\\upscaled\\source_apollo_hybrid.wav", "url": "/api/media?p=…",
 "kind": "apollo", "spliced": true,
 "upscaled_path": "B:\\…\\upscaled\\source_apollo.wav", "upscaled_url": "/api/media?p=…",
 "compare_url": "/api/media?p=…" ,
 "bandwidth_before": "…", "bandwidth_after": "…",
 "region": "0.000:*:16000:*"}
```

Outputs are **not** library members (they live in `upscaled\`, which
`lib.entries()` never globs). They are playable via `url` but will not appear in
the library list — that is the intended behaviour, preserved.

---

## 7. Lyrics

### 7.1 `POST /api/lyrics/probe` — instant (~1 s), no GPU

Replaces `find_lyrics()`. Runs automatically when the cover source changes.

**Request** `{"path": "B:\\AudioDev\\Music\\song.mp3", "uploaded": false}`

`uploaded` tells the server the path is a browser upload with no original
neighbours, so the message can say so. The server also forces `uploaded = true`
when `path` is under `OUTDIR\uploads\`.

**Response `200`**
```json
{"ok": true, "found": true,
 "text": "[verse]\n…",
 "message": "**Found lyrics** in file .lrc — matched by stem (24 lines). Apply to use.",
 "best": {"source": "sidecar", "field": ".lrc", "text": "…", "why": "matched by stem"},
 "embedded": [], "sidecars": ["B:\\…\\song.lrc"]}
```

* `message` = `ly.describe(probe_result, uploaded_only)` verbatim (it is markdown
  and may carry the trailing italic caveat about uploads).
* `text` = `best.text` when `found`, else `null`.
* No `path` → `200` with
  `{"ok": false, "found": false, "text": null, "message": "Pick a source track to search for its lyrics."}`
  (not an error — this is the idle state).
* Probe failure → `{"ok": false, "found": false, "message": "Lyrics probe failed: …"}`.

Delegates `ly.probe(path)` + `ly.describe(res, uploaded_only)`.

### 7.2 `POST /api/lyrics/transcribe` — LONG-RUNNING, GPU

**Request**
```json
{"path": "B:\\…\\song.mp3", "separate": true, "model": "large-v3", "seconds": 0}
```
`seconds: 0` = whole track. `model` ∈ `ly.WHISPER_MODELS`. `separate` runs Demucs
before Whisper.

**Response `202`** — same shape as §5.1.

**Job execution:** first `progress` line `transcribing <basename>`, then
`ly.transcribe(path, separate=…, model=…, seconds=float(seconds or 0), on_progress=emit)`.
Progress arrives as plain strings on stderr.

**`result` payload**
```json
{"seq": 88, "text": "…transcribed lyrics…",
 "segments": 42, "language": "en", "language_probability": 0.99,
 "elapsed": 88.0, "separated": true,
 "summary": "42 segments, language en (0.99), 88s · vocals isolated",
 "note": "Structure tags are NOT invented — add [verse] / [chorus] yourself if you want ACE-Step to follow them."}
```

The frontend writes `text` straight into the lyrics box (the old app did the
same) and shows `note`.

### 7.2b `POST /api/write` — LONG-RUNNING, GPU **only when it is free**

The local LLM that writes lyrics and style prompts. Two targets × three modes.

```jsonc
// request
{ "target": "lyrics" | "style",
  "mode":   "generate" | "extend" | "edit",
  "model":  "minimax" | "acestep",   // which house style to write in
  "brief":  "…",                     // the idea, or for edit the instruction
  "existing": "…",                   // required for extend and edit
  "seed":   123456 | null,           // null rolls a new one
  "temperature": 0.9 }

// 202
{ "job_id": "j_ab12cd", "kind": "write", "state": "queued",
  "device": "cuda" | "cpu",          // decided at POST time, see below
  "stream_url": "/api/jobs/j_ab12cd/events" }

// result event
{ "text": "…", "target": "lyrics", "mode": "edit", "model": "minimax",
  "seed": 428913, "tokens": 351, "retried": false,
  "scoped_to": "Chorus" | null,      // set when only one section was rewritten
  "device": "cuda", "warning": null }
```

**Device is chosen, not queued.** `gpu_idle() && free_gb >= 4.0` under the
registry lock → the GPU lane via `_enqueue`; otherwise `Registry.submit_cpu`,
its own thread, off the lane. Measured: 20 tok/s on the card, 2–4 on this CPU.
Waiting for a busy card would cost more than the CPU run it replaced, so it
never waits.

**400s** (before any job exists): unknown target/mode/model; `generate` with no
brief; `extend`/`edit` with no `existing`; `edit` with no instruction.
**503 `model_missing`** when the weights are not on disk — the writer never
downloads in-band, and the button is hidden rather than shown-and-failing.

Structure enforcement and the scoped-edit rule are REQUIREMENTS §10.

### 7.2c `GET /api/loras` — instant, no GPU

```jsonc
{ "loras": [{
    "id": "trained:checkpoints/epoch_10_loss_0.8248",  // stable, path-derived
    "name": "checkpoints/epoch_10_loss_0.8248",
    "label": "… — rank 16, alpha 32, 42 MB",
    "path": "B:\AudioDev\train\lora_out\checkpoints\epoch_10_loss_0.8248",
    "source": "trained" | "drop",
    "rank": 16, "alpha": 32, "peft_type": "LORA",
    "target_modules": ["k_proj","o_proj","q_proj","v_proj"],
    "base_model": "…/acestep-v15-turbo", "base_model_name": "acestep-v15-turbo",
    "compatible": true,          // false = trained against another base
    "size_bytes": 44089608, "mtime": 1755000000.0 }],
  "roots": [{"key": "drop", "path": "B:\AudioDev\loras", "exists": true}, …],
  "applies_to": ["acestep"] }
```

Rescans on every call — the drop folder only works if a file appearing in it
appears here. The same list rides on `/api/config` as `loras.items` so Create
can build its dropdown without a second round trip.

`POST /api/generate` gains two optional fields, ACE-Step only:

```jsonc
{ "lora": "trained:final", "lora_scale": 1.0 }   // scale clamped to [0, 2]
```

400 on an unknown id (resolved at POST time, never a job that dies mid-render)
and 400 when paired with a model that has no LoRA support. The resolved
adapter path, an explicit collision-proof adapter name, and the scale go to the
worker as `lora_path` / `lora_name` / `lora_scale`; **absent keys mean no
LoRA**, which is what keeps the Gradio app on :7861 working. Which adapter
actually ran is read back from the handler and written into the sidecar.

See REQUIREMENTS §11.

### 7.3 Apply found lyrics — **no endpoint**

`apply_lyrics` was pure clipboard work. Client-side:

```
if (!found) toast("Nothing found to apply — try Transcribe.")
else {
  if (current.trim() && current.trim() !== found.trim())
      toast("Replaced the lyrics box with the lyrics that were found.")
  lyricsBox.value = found
}
```

Never silently clobber typed lyrics without the notice.

### 7.4 Cover-source resolution — **client-side rule**, one server check

The old `cover_source()` rule: *a real on-disk path beats the browser's copy*,
because an upload lands in a temp folder with no sidecar neighbours.

```
typed = typedPath.trim().replace(/^"|"$/g, "")
if (typed && await fsStat(typed).is_file)  use {path: typed, uploaded: false}
else if (uploadedServerPath)               use {path: uploadedServerPath, uploaded: true}
else                                       no source
```

`fsStat` = §11.3.

---

## 8. Library

### 8.1 The entry object

Returned in the list **complete enough to render both the tile and the detail
pane with no follow-up request** — an N+1 over Tailscale on a phone is not
acceptable.

```json
{
  "id": "20260814-120301_minimax.wav",
  "name": "20260814-120301_minimax.wav",
  "path": "B:\\AudioDev\\Music\\studio\\20260814-120301_minimax.wav",
  "title": "Softly The World Begins",
  "audio_url": "/api/media?p=…",
  "cover_url": "/api/media?p=…",
  "mtime": 1755180181.4,
  "created": "2026-08-14T12:03:01",

  "recorded": true,
  "shared_rec": false,
  "take": 1, "takes": 2,

  "favorite": false,
  "rating": 0,

  "model": "minimax",
  "prompt": "Global Metadata\n…",
  "lyrics": "[verse]\n…",
  "instrumental": false,
  "duration": 30.0, "steps": 30, "seed": 7,
  "seconds": 30.2, "sampling_rate": 44100, "elapsed": 214.0, "vram_peak": 8.71,
  "preview": false,
  "post_kind": "none",
  "cover_src": null, "cover_strength": null, "noise_strength": null,

  "title_override": "", "style": "",
  "caption": "Softly The World Begins\ntake 1/2 · 30s · minimax"
}
```

Field semantics that the implementers **must** get right:

| field | source | meaning |
|---|---|---|
| `title` | `titles.derive(rec, name)` | Never empty. Uses `rec["title"]` when set, else derives from lyrics/prompt, else the filename. |
| `title_override` | `rec.get("title", "")` | What the user typed. Empty string = "derived". This is what the Title input is bound to, **not** `title`. |
| `recorded` | `entry["recorded"] and not rec.get("inherited_from")` | A real sidecar with real provenance. **`Reuse` is refused when false.** |
| `shared_rec` | `entry["shared_rec"] or bool(rec.get("inherited_from"))` | The prompt shown was borrowed from the sibling take. ACE-Step writes two takes per run and only one carries the record. |
| `take` / `takes` | `library.group_takes` | 1-based index within a run, and the run's size. Rendered in the caption only when `takes > 1` — otherwise every take in a pair looks identical. |
| `favorite` | `pls.is_favorite(path)` | Per file, never through an inherited record. |
| `rating` | `-1` \| `0` \| `1` | Tri-state. Derivation below. |
| `cover_url` | `art.cover_path(path)` if it exists, else `null` | A missing cover renders as a tile with no image, never blocks the list. |
| `caption` | computed | `f"{('★ ' if favorite else '')}{title}\n{' · '.join(bits)}"` where `bits` = `take i/n` (only if `takes>1`), `f"{seconds:.0f}s"` (if present), `model` (if present). Two lines; the second must not be clipped. |

**`rating` derivation** (from `rowview._rating`, keep in one place):

```python
def rating_of(entry):
    if entry["shared_rec"]:              # a verdict must not ride on a lent record
        return 0
    r = (entry["rec"] or {}).get("rating")
    if isinstance(r, bool):   val = 1 if r else 0
    elif isinstance(r, (int, float)): val = 1 if r > 0 else (-1 if r < 0 else 0)
    elif isinstance(r, str):
        s = r.strip().lower()
        val = 1 if s in ("up","like","liked","+1","1") else (-1 if s in ("down","dislike","disliked","-1") else 0)
    else: val = 0
    if val == 0 and (entry["rec"] or {}).get("favorite"): val = 1
    return val
```

### 8.2 `GET /api/library` — instant, no GPU

Query params, all optional:

| param | default | meaning |
|---|---|---|
| `q` | `""` | case-insensitive substring search |
| `fav` | `0` | `1` = favourites only |
| `playlist` | absent/empty = all tracks | playlist name |
| `limit` | `500` | page size |
| `offset` | `0` | |

**Filter order — must match exactly:**

1. `es = lib.entries()` (newest first, take-grouped).
2. If `playlist`: `want = pls.tracks(playlist)`; keep entries whose `name` is in
   `want`; **sort by playlist order, not date**.
3. If `fav`: keep `pls.is_favorite(e["path"])`.
4. If `q`: keep entries where `q.lower()` is a substring of
   `" ".join([title, name, rec.prompt, rec.lyrics, rec.model]).lower()`.
5. Paginate.

**Response**
```json
{
  "entries": [ …entry objects… ],
  "total": 128,
  "returned": 128,
  "offset": 0,
  "with_prompt": 96,
  "awaiting_cover": 4,
  "status": "128 tracks · 96 with a prompt  ·  4 awaiting cover art",
  "playlists": ["late night", "demos"]
}
```

* `with_prompt` counts entries where `recorded or shared_rec` (pre-pagination).
* `awaiting_cover` counts entries with no file at `art.cover_path(path)`.
* `status` is the ready-made line; the frontend may render its own instead.
* `playlists` = `pls.names()` — saves a second request for the filter dropdown.

**Cover backfill.** This GET fires the background renderer, exactly as the old
refresh did: collect entries whose `art.cover_path` is missing; if a module-level
`threading.Lock` is not already held, spawn a daemon thread that takes it and
calls `art.ensure_cover(p)` for at most **60** of them, each in `try/except:
pass`. Never block the response — each render is a ~0.6 s subprocess in the media
venv. The covers appear on the next refresh. Generation only ever covers the file
the worker hands back, so this is what keeps the library self-healing for
ACE-Step's second take and for anything produced outside the UI.

### 8.3 `GET /api/library/{id}` — instant

The entry object plus:

```json
{"entry": {…}, "detail_md": "### 20260814-120301_minimax.wav\n\nmodel **minimax** · seed **7** · …",
 "shared_note": "*Prompt inherited from take 2 of the same run — ACE-Step writes two takes and only one carries the record.*"}
```

`detail_md` = `lib.detail(entry)`. `shared_note` is non-null only when
`shared_rec`; the take number is `1 if take == 2 else 2`.

Not needed for normal browsing (§8.2 is sufficient) — provided for a deep link
and for re-reading one row after a mutation.

### 8.4 `PATCH /api/library/{id}` — instant

Commit the detail-pane edits.

**Request** `{"title": "New Name", "style": "doom folk, post rock"}`
Either key may be omitted (leave unchanged). **An empty string clears the
override** — send `null` to `lib.update`, which removes the key:

```python
lib.update(path, title=(title or "").strip() or None,
                 style=(style or "").strip() or None)
```

**Response** `{"entry": {…updated…}, "message": "Saved details for 20260814-120301_minimax.wav"}`

### 8.5 `POST /api/library/{id}/retitle` — slow-sync (0–30 s), no GPU

**Request** `{"use_llm": true}`

* `use_llm: false` → `t = titles.derive({**rec, "title": None}, name)`; `how = "heuristic"`. Instant.
* `use_llm: true` → `got = llm_title.suggest(lyrics=rec.get("lyrics"), prompt=rec.get("prompt"), n=1, timeout=30)`.
  `[]` means the model is absent, slow, or the input had no substance — **every
  one of those falls back** to the heuristic rather than failing the action.
  `how = "LLM"` when `got`, else `"heuristic (LLM declined)"`.

Then `lib.update(path, title=t)`.

Pass `timeout=30` (the module default is 120) so a phone request cannot hang for
two minutes. The fallback is `titles.derive`, which cannot fail.

**Response** `{"title": "Softly The World Begins", "how": "LLM", "entry": {…}, "message": "Titled **…** (LLM)"}`

Frontend: the returned `title` goes into the **Title override** input.

### 8.6 `PUT /api/library/{id}/favorite` — instant

**Request** `{"on": true}` — absolute, not a toggle. The client knows the current
state from the entry.
**Response** `{"favorite": true, "entry": {…}}`
Delegates `pls.set_favorite(path, on)`. Label in the UI: `★ Favourited` / `☆ Favourite`.

### 8.7 `PUT /api/library/{id}/rating` — instant

**Request** `{"rating": 1}` — one of `1`, `0`, `-1`. Absolute, not a toggle; the
client computes the new value from the current one (clicking an active thumb
sends `0`).

**Server-side coupling — the ★ and the 👍 are the same gesture:**

| new rating | previous | actions |
|---|---|---|
| `1` | any | `lib.update(rating=1)` **and** `pls.set_favorite(path, True)` |
| `-1` | any | `lib.update(rating=-1)` **and** `pls.set_favorite(path, False)` |
| `0` | was `1` | `lib.update(rating=None)` **and** `pls.set_favorite(path, False)` |
| `0` | was `-1` | `lib.update(rating=None)` only — **do not touch the favourite** |

**Response** `{"rating": 1, "favorite": true, "entry": {…}, "message": "Liked — 20260814-120301_minimax.wav"}`

### 8.8 ⚠ The `shared_rec` write trap — REQUIRED handling

Writing *anything* to a `shared_rec` entry (rating, favourite, title) creates a
sidecar for that file. `library.entries()` then reports `recorded: true` for it,
and `group_takes` only lends the donor's record to entries where `recorded` is
false — **so the borrowed prompt silently disappears from that row.**

Before the first write to any entry with `shared_rec == true`, the backend must
seed the new sidecar from the donor:

```python
def ensure_own_record(entry):
    """Give a take that is borrowing its sibling's record a record of its own."""
    if not entry["shared_rec"] or lib.read(entry["path"]):
        return
    donor = dict(entry["rec"] or {})
    donor.pop("audio", None); donor.pop("audio_path", None)
    donor.pop("favorite", None); donor.pop("rating", None)
    donor["inherited_from"] = <donor entry's basename>
    lib.write(entry["path"], **donor)
```

The API then reports `recorded = entry["recorded"] and not rec.get("inherited_from")`
and `shared_rec = entry["shared_rec"] or bool(rec.get("inherited_from"))`, so the
row keeps showing the prompt, still refuses `Reuse`, and still shows `rating: 0`
until a rating is explicitly written to it. No portable module is modified — this
is a derivation in the API layer.

### 8.9 `POST /api/library/{id}/trash` — instant

Move to `trash\` — reversible, **never** `os.remove`.

**Response `200`** `{"trashed": true, "dest": "B:\\…\\studio\\trash\\x.wav", "message": "Moved **x.wav** to trash\\ (recoverable)."}`
**Response `409 move_failed`** when `pls.trash()` returns `None`:
`"Could not move that file to the trash folder."`

The sidecar moves with the audio under the same stem. Frontend: refresh the
library, clear the selection, and stop the player if the trashed track was
playing.

### 8.10 `GET /api/library/{id}/reuse` — instant

Load a track's prompt/lyrics/params back into Generate.

**Response `200`**
```json
{"model": "minimax", "prompt": "…", "lyrics": "…",
 "duration": 30.0, "steps": 30, "seed": 7, "instrumental": false,
 "message": "Loaded **20260814-120301_minimax.wav** into Generate ✓"}
```
Defaults when a field is absent: `model "minimax"`, `duration 30`, `steps 30`,
`seed 7`, `instrumental false`, `lyrics ""`.

**Response `409 no_record`** when `recorded` is false:
`"That track has no saved prompt — it predates the library, or is an alternate take."`

### 8.11 `GET /api/library/{id}/cover-source` — instant (~1 s)

Load a library track into Generate as an ACE-Step cover source. Unlike Reuse this
works for **any** track, recorded or not.

**Response `200`**
```json
{
  "model": "acestep",
  "prompt": "…", "lyrics": "…", "instrumental": false,
  "src_path": "B:\\AudioDev\\Music\\studio\\x.wav",
  "src_url": "/api/media?p=…",
  "cover_strength": 1.0,
  "noise_strength": 0.75,
  "lyrics_probe": { …the §7.1 response… },
  "message": "Covering **Softly The World Begins** — source loaded in Generate. Edit the style, then Generate."
}
```

The `src_path` is the **real on-disk path**, deliberately: it lets the lyrics
probe read the track's own JSON sidecar and offer the lyrics it was generated
with, which for a cover is usually exactly what you want to keep. ACE-Step is the
only backend with a cover task, so the frontend switches the model to `acestep`
and opens the cover panel — **without** re-applying the model's stock default
prompt (that would overwrite the very prompt being restored).

### 8.12 `POST /api/library/covers` — instant (kicks background)

**Request** `{"force": false, "max": 60}`
**Response** `{"queued": 4, "running": true}`
Explicit trigger for the same backfill §8.2 runs implicitly. `force: true` passes
`force=True` to `art.ensure_cover` (re-render existing covers).

---

## 9. Playlists, trash

All instant, no GPU, all delegating to `playlists.py`. Playlist membership is by
**basename**, not path — the folder is the namespace.

| Method + path | Request | Response | Delegates |
|---|---|---|---|
| `GET /api/playlists` | — | `{"playlists":[{"name":"demos","count":7}]}` | `pls.load()` |
| `POST /api/playlists` | `{"name":"demos"}` | `201 {"name":"demos","created":true}` (`created:false` if it already existed — not an error) | `pls.create` |
| `DELETE /api/playlists/{name}` | — | `{"deleted":true}` / `404` | `pls.delete_playlist` |
| `GET /api/playlists/{name}` | — | `{"name":"demos","tracks":["a.wav","b.wav"]}` in playlist order | `pls.tracks` |
| `POST /api/playlists/{name}/tracks` | `{"id":"a.wav"}` | `{"added":true,"count":8,"message":"Added to **demos** (8 tracks)"}` — `added:false` when the track was already in the playlist, which is **not** an error | `pls.create` then `pls.add` |
| `DELETE /api/playlists/{name}/tracks/{id}` | — | `{"removed":true}` | `pls.remove` |
| `POST /api/playlists/{name}/reorder` | `{"id":"a.wav","delta":-1}` | `{"ok":true,"tracks":[…]}` | `pls.reorder` |
| `GET /api/trash` | — | `{"items":[{"name":"x.wav","url":"/api/media?p=…","mtime":1755…}]}` newest first | `pls.trashed` |
| `POST /api/trash/{name}/restore` | — | `{"restored":true,"path":"B:\\…\\x.wav"}` / `409 exists` | `pls.restore` |

`POST /api/playlists/{name}/tracks` creates the playlist if it does not exist
(the old "+ Add" button did `create` then `add`). Reject `name` that is empty
after stripping → `400 "Type a playlist name, or pick an existing one."`

Playlist entries are deliberately **not** pruned when a track is trashed — the
dangling entry is the price of `restore()` putting the track back where the
playlist still expects it. Do not add pruning.

`pls.reorder` returns `True` for a no-op nudge at either end; that is success,
not an error.

---

## 9b. Settings and workspaces

Instant, no GPU. Both are small JSON registries in `OUTDIR` with the same
contract as `playlists.py`: pure stdlib, atomic write via `mkstemp` + `replace`,
and a missing or corrupt file reads as empty rather than raising.

| Method + path | Request | Response | Delegates |
|---|---|---|---|
| `GET /api/profile` | — | `{"artist":"The Quiet Hours"}` | `settings.load()` |
| `PUT /api/profile` | `{"artist":"The Quiet Hours"}` | `{"artist":"The Quiet Hours","note":null}` | `settings.set_artist` |
| `GET /api/workspaces` | — | `{"active":"Soundtrack","names":[…],"counts":{…},"untagged":12,"note":"…"}` | `wsp.load` + `wsp.counts` |
| `POST /api/workspaces` | `{"name":"Soundtrack"}` | `{"created":"Soundtrack","names":[…],"active":…}` | `wsp.create` |
| `PUT /api/workspaces/active` | `{"name":"Soundtrack"}` or `{"name":null}` | `{"active":…,"names":[…]}` | `wsp.set_active` |
| `DELETE /api/workspaces/{name}` | — | `{"deleted":true,"names":[…]}` | `wsp.delete` |

### `/api/profile`

Studio-wide settings. One flat object rather than a bare string, so the next
global has somewhere to go without a migration.

**There is no default artist and there must not be one.** `""` is the honest
unset state, not a value to fill in — the lock screen carried a hard-coded name
for one day, which credited every track to a label nobody had typed. Clients
must send an empty `artist` rather than substituting a fallback.

- **clearing is a legitimate edit.** `PUT {"artist":""}` stores empty and
  returns it. Only a *missing* `artist` key is `400 validation`
- `note` is non-null when the stored value differs from what was sent —
  currently only `"Shortened to 64 characters."` and `"That name had nothing
  storable in it."` Clients show it rather than letting the field silently
  disagree with the box it came from
- cleaning strips control characters and collapses whitespace runs, and nothing
  else. Punctuation, accents and non-Latin scripts all belong to somebody
- the module is `studio/settings.py`, **not `profile.py`** — `profile` is a
  stdlib module and `sys.path.insert(0, …\studio)` would shadow it process-wide
- **not written into audio tags.** A Suno track arriving with "made with suno"
  in its LIST/INFO chunk is what this toolkit exists to avoid

### `/api/workspaces`

The project a generation was made *for*, as opposed to a playlist, which is a
set you curate afterwards. Membership is a **sidecar field**, so the counts are
derived from the library rather than from an index that could drift; this file
holds only what membership cannot express — the list (so an empty workspace can
exist) and which one is active.

`active: null` means untagged, which is also the migration story: every track
predating the feature has no workspace and must stay visible, so any view that
filters by workspace needs an "Everything" option.

Active lives on the **server**, not in a browser: the phone and the desktop both
generate, and two localStorage copies would disagree about where tonight's songs
went.

---

## 10. Media, upload, filesystem

### 10.1 `GET /api/media?p=<url-encoded absolute path>`

Serves audio, covers, spectrogram PNGs and assets.

* **Allowlist**: after `os.path.realpath` + `os.path.normcase`, the path must sit
  under `OUTDIR` (which covers `covers\`, `upscaled\`, `trash\`, `uploads\`) or
  under `ASSETS`. Anything else → `404 not_found` (report as not-found, never as
  forbidden; do not confirm existence outside the roots). This is the exact
  parity of gradio's `allowed_paths=[OUTDIR, ASSETS]`.
* **Range requests are mandatory** — `Accept-Ranges: bytes`, `206 Partial
  Content` with a correct `Content-Range`. Without them a phone cannot scrub a
  four-minute WAV.
* `Content-Type` by extension: `.wav → audio/wav`, `.flac → audio/flac`,
  `.mp3 → audio/mpeg`, `.m4a/.aac → audio/mp4`, `.ogg/.opus → audio/ogg`,
  `.png → image/png`.
* `Cache-Control`: `private, max-age=300` for audio and covers;
  `no-store` for `_spec_*.png`.
* `HEAD` must work (the audio element probes with it).

**Consequence to state plainly:** a file the user typed a path to (e.g.
`D:\Music\some.mp3`) is **processable but not playable**. The Restore and cover
flows read it server-side; the browser cannot stream it. The frontend must not
offer a play button for an out-of-root source — show the path and its bandwidth
report instead. To make such a file playable, upload it (§10.2).

### 10.2 `POST /api/upload`

`multipart/form-data`, single field `file`.

* Destination `OUTDIR\uploads\<uuid4hex[:8]>_<sanitised original name>`.
  **`uploads\` is a subdirectory, so `library.entries()` never sees uploads** —
  the same mechanism that hides `upscaled\`. This is intentional; do not change it.
* Accepted extensions: `.wav .flac .mp3 .m4a .aac .ogg .opus`. Others → `400`.
* Max 200 MB → `413 too_large`.

**Response `201`**
```json
{"path": "B:\\AudioDev\\Music\\studio\\uploads\\9f21c4e0_song.mp3",
 "url": "/api/media?p=…", "name": "song.mp3", "size": 8412345, "uploaded_only": true}
```

`uploaded_only: true` is carried into `POST /api/lyrics/probe` so `ly.describe`
emits its caveat: sidecar files cannot be seen for an upload, because the browser
only sends a copy — paste the file's real path to search beside it.

Uploads older than 7 days are swept on startup.

### 10.3 `GET /api/fs/stat?p=<url-encoded absolute path>`

The typed-path check. **No allowlist** (this is a local single-user tool and the
Restore tab must accept any path), but it returns *nothing* beyond existence and
size — never directory listings.

```json
{"exists": true, "is_file": true, "size": 8412345, "playable": false,
 "name": "some.mp3"}
```
`playable` = the path passes §10.1's allowlist.

---

## 11. Operations with NO endpoint — pure client-side

The frontend agent must implement these locally; the backend agent must **not**
build routes for them.

| Old handler | New home | Rule |
|---|---|---|
| `on_model_change` | client | On a **human** model change: set prompt to `config.backends[m].default_prompt`, steps to `default_steps`, show/hide the instrumental checkbox (`supports_instrumental`), show/hide the cover panel (`supports_cover`), swap `prompt_info` / `lyrics_info` / `note`. **Never run this cascade when the model is set programmatically** by Reuse, Cover, or settings restore — it would overwrite the very prompt being restored. |
| `apply_lyrics` | client | §7.3 |
| `cover_source` | client | §7.4 (uses `GET /api/fs/stat`) |
| `on_click` spectrogram picking | client | §6.3 — pure arithmetic on `geometry` |
| `reset_region` | client | §6.3 — `flo = geometry.cliff \|\| 16000`, then re-inspect |
| `region_specs` | server | client sends numbers; server formats (§6.1) |
| `pick_track` | client | The entry is already in the list from `GET /api/library` |
| `play_step` / `play_next` / `play_prev` | client | §12 |
| `toggle_shuffle` / `toggle_repeat` | client | §12 |
| `_at` / now-playing text | client | `**{title}** · {i+1}/{n}` plus a muted sub-line of `model` and `take i/n` |
| `_label` / `_gallery` | client | `caption` is precomputed in the entry (§8.1) |
| `_bar` progress bar | client | render from `live.frac` |
| `vram_text` string | client | format from `GET /api/vram` (§4.3) |
| `duration_hint` string | client | format from `GET /api/budget` (§4.4) |
| single-player enforcement | client | Only one `<audio>` in the DOM. If a second is ever added, pause every other `HTMLMediaElement` on a capture-phase `play` listener. |

---

## 12. State: what was in `gr.State`, and where it lives now

| Old | Where | Notes |
|---|---|---|
| `lib_entries` (`gr.State([])`) — the filtered entry list backing the gallery | **Client**, from `GET /api/library`. Server holds no per-session copy. | The list is re-fetched after every mutation (rating, trash, retitle, generation). Never index a stale list: after a mutation, re-find the track **by `id`**, not by the old index — the list can shrink (trash, ★ filter, search). |
| `lib_row` (`gr.State(None)`) — selected row index | **Client**, stored as the selected **`id` string**, not an index. | Derive the index when needed. If the id is no longer in the refreshed list, clear the selection and show "Track left the current view." |
| `up_geom` — geometry of the current spectrogram render | **Client**, replaced wholesale by every `/api/restore/inspect` response. | Must be discarded whenever the source file changes. A click mapped through a stale rect produces wrong regions. |
| `up_anchor` — first corner of a pending selection | **Client** only. | §6.3 |
| `lyr_found` — probed lyrics text | **Client**, from `POST /api/lyrics/probe`.`text` | |
| `play_shuffle`, `play_repeat` | **Client** (was server-side purely as a gradio artifact). Persist in `localStorage` under `audiodev.studio.player.v1`. | |
| **Play queue** | **Client.** The queue *is* the current filtered library list, in its displayed order, plus a cursor (the selected `id`). | `next` = cursor + 1; at the end, wrap only when `repeat`, else stop ("End of playlist."). `prev` = cursor − 1; below zero → wrap to last when `repeat`, else clamp to 0. `shuffle` applies to **forward** moves only and picks uniformly from the other indices — never the current one. Auto-advance fires on the `<audio>` element's `ended` event. |
| **Remembered generate settings** (was `gr.BrowserState`) | **`localStorage`**, key **`audiodev.studio.generate.v2`**, plain JSON, no obfuscation. | Per-device by design: the phone and the desktop each keep their own last session. |

### 12.1 The settings blob

```json
{"v": 2, "model": "minimax", "prompt": "…", "lyrics": "…",
 "duration": 30, "steps": 30, "seed": 7, "instrumental": false,
 "post_kind": "none"}
```

* Fields: exactly `model, prompt, lyrics, duration, steps, seed, instrumental,
  post_kind` — the same eight as before.
* **Written on every edit**, debounced 400 ms. Never in response to reading it
  back (that is an infinite loop).
* **Restored per FIELD, not all-or-nothing**: a blob written by an older version
  can be missing keys, and defaulting the whole dict would throw away the fields
  it does have. `value = saved[k] ?? configDefault[k]`.
* Bump `v` and the key when the shape changes; an old blob under the same key
  would restore fields that no longer mean the same thing.
* **Clean break from v1.** The old gradio `BrowserState` blob is written under
  `audiodev.studio.generate.v1` wrapped in gradio's own secret-based encoding and
  is not readable by our code. Do not attempt to migrate it.
* After restoring, apply the ACE-only panel visibility from the restored `model`
  **without** running the `on_model_change` cascade (§11).

Also persisted client-side, separate keys, same rules:
* `audiodev.studio.library.v1` — `{q, fav, playlist, view}` (last search / filter /
  grid-or-list).
* `audiodev.studio.restore.v1` — `{kind, auto_lowpass, fill}`.
* `audiodev.studio.player.v1` — `{shuffle, repeat, volume}`.

---

## 13. Frontend flows that must survive the port

Behavioural contracts the two agents both have to honour. These are the things
the old app got right and are easy to lose.

1. **One player, one song.** Every result — a generation, an upscale, a library
   track — lands in the same dock player. Three per-tab players is how you end up
   with two songs going at once.
2. **The GPU is one lane.** Two jobs must never run together. The UI must show
   *what* is running and offer Cancel only where §2.8 says it works.
3. **Trash is a move.** No endpoint anywhere calls `os.remove` on a track.
4. **The sidecar is the record.** Nothing is written into the audio's tags.
5. **Post-processing failure is not generation failure.** A generation that
   completed is a success even if the upscaler died.
6. **Cover art is best-effort.** A missing cover is a tile with no image.
7. **A cliff verdict is advisory, never a block.** Report and proceed.
8. **Probes never overwrite typed text.** Lyrics detection is automatic; applying
   is an explicit, separate action.
9. **`Reuse` is refused without a real record; `Cover` is not.**
10. **The spectrogram marker.** Every click re-render draws a crosshair where the
    client *believes* the click landed. Do not drop it as decoration — it is the
    only thing that makes a coordinate-space bug visible.
11. **Mobile.** 390×844 first. Touch targets ≥ 44 px. No horizontal page scroll at
    any width — wide content (the spectrogram, the log) scrolls inside its own
    `overflow-x: auto` container. The library detail pane and the Generate form
    are one column on a phone and a two-pane split at ≥ 900 px. The log is
    collapsed by default on a phone with the live progress bar always visible.
12. **Offline-capable chrome.** No CDN, no external fonts. System font stack:
    `ui-sans-serif, system-ui, "Segoe UI", sans-serif`; mono
    `ui-monospace, "Cascadia Code", Consolas, monospace`.

---

## 14. Worked example — a generation from the phone

```
GET  /api/config                                  -> backends, defaults, upscalers
GET  /api/vram                                    -> pill
GET  /api/library                                 -> tiles + playlists
GET  /api/jobs?active=1                           -> [] (nothing running)
      (restore settings from localStorage)
GET  /api/budget?model=minimax&duration=180       -> {"fits": false, …}  -> warn

POST /api/generate {…}                            -> 202 {"job_id":"j_7f3a"}
GET  /api/jobs/j_7f3a/events
     event: hello      {state:"queued", …}
     event: queued     {position:0}
     event: progress   {msg:"starting MiniMax Music 3 (first load takes ~20 s)", frac:null}
     event: vram       {used_gb:8.4, loaded:"minimax"}
     event: progress   {msg:"decoding 12%", stage:"decode", frac:0.12}
     …
     (phone sleeps; socket dies; EventSource reconnects with Last-Event-ID: 137)
     event: progress   {seq:138, …}                ← replayed, no hello needed
     …
     event: artifact   {kind:"audio", label:"generated", url:"/api/media?p=…"}
     event: result     {seconds:180.4, elapsed:1280.0, …}
     (client calls es.close())

GET  /api/library                                 -> new track at the top
```
