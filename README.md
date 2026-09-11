# Audio Super-Resolution / Bandwidth Extension

Three installed tools that rebuild high frequencies destroyed by lossy compression.

## TL;DR — which one do I use?

Ranked by a **ground-truth test**: a real track degraded to 96 kbps, restored, then
compared against the untouched original (full method under "Which is actually best").

| | AudioSR | Apollo | FlashSR |
|---|---|---|---|
| **Accuracy** — mean error vs original, 15.3–20 kHz | **4.37 dB** 🥇 | 7.00 dB 🥈 | 11.34 dB |
| Speed, 30 s clip | 231 s | **22 s** 🥇 | 69 s |
| Output rate | 48 kHz | 44.1 kHz (**resamples**) | 48 kHz |
| Stereo | preserved (via wrapper) | preserved | preserved |
| Model size | 5.9 GB | 63 / 140 MB | 3.1 GB |
| Type | latent diffusion, 50 steps | band-split GAN | 1-step distilled diffusion |

- **Best quality → AudioSR** (`audiosr.ps1`). Slowest by a wide margin, but clearly
  the closest reconstruction, and it keeps 48 kHz.
- **Best speed → Apollo** (`apollo.ps1`). ~10× faster and still second on accuracy.
  Only real drawback: it resamples 48 kHz sources down to 44.1 kHz.
- **FlashSR installed but not recommended.** It is the one-step distillation of
  AudioSR and should have been the machine-fit winner; measured, it was neither the
  fastest nor the most accurate. Kept for A/B — see its caveat below.

AudioSR's README warns it does badly on raw MP3 input (trained on clean low-pass
filtering only). `-AutoLowpass` applies the authors' recommended workaround — detect
the true cutoff, filter there first — and that is how the winning number above was
produced. Use it on any lossy source.

Timing caveats, all measured:
- **Apollo's first run of a session takes ~110 s** regardless of file length — CUDA
  context + cuDNN autotune warmup, not the model. Subsequent runs are ~7 s per 15 s.
- **Stock AudioSR took 19 minutes on a 15-second clip**, spilling into shared system
  memory at 76× slower than realtime. Always go through `audiosr.ps1`, which chunks.
- Speeds above were measured while ComfyUI was also holding the GPU, so they are
  pessimistic in absolute terms — but all three tools were measured under the same
  contention, so the *ranking* is fair.

---

## Getting it onto another machine

The code lives at `github.com/xenowiz-dev/AudioDev` (private). Clone it
anywhere and run `.\install.ps1`; the tree discovers its own location, so the
`B:\AudioDev` paths in the examples below are just where this box keeps it.
`install\README.md` lists what the installer rebuilds and what has to be copied
by hand (the library, the trained adapters).

## Layout

```
<root>\                        # wherever it was cloned
  audiosr.ps1                  # AudioSR wrapper (chunked, handles long files)
  apollo.ps1                   # Apollo wrapper
  check_bandwidth.py           # run this FIRST — is upresing even worth it?
  spectrogram.py               # before/after spectrogram figures (--fmin to zoom)
  band_splice.py               # keep the original, add ONLY the band above a cutoff
  region_fill.py               # same idea, but time x frequency rectangles
  lyrics_probe.py              # find existing lyrics: embedded tags or sidecars
  transcribe.py                # demucs vocal isolation -> faster-whisper
  lyrics\
    .venv\                     # Python 3.10, torch cu128 + demucs + faster-whisper
  watermark\                   # AI-audio provenance: watermarking + detection
    AI_AUDIO_PROVENANCE.md     #   >> the writeup: how AI music is marked & caught
    ai_audio_forensics.py      #   all three layers in one report
    .venv\                     #   Python 3.10, CPU torch (models are small)
  studio\                      # >> LOCAL WEB UI over both music generators
    README.md                  #   why it is not a ComfyUI backend; the VRAM design
    app.py                     #   gradio front-end (own venv, no torch)
    supervisor.py              #   one warm worker at a time, JSON-lines protocol
    postprocess.py             #   upscalers as a post step (calls the wrappers)
    workers\                   #   one per model, each in that model's venv
    cli.py                     #   headless driver — test without the UI
    studio.ps1                 #   launcher -> http://127.0.0.1:7861
  minimax\                     # MiniMax Music 3 generation (44.1 kHz stereo)
    README.md                  #   >> how it fits in 10 GB; do NOT follow the model card
    gen.py                     #   generate; --dry-run reports VRAM without generating
    download.ps1               #   pulls only the ~27 GB diffusers needs, not all 57 GB
    .venv\                     #   Python 3.12, torch 2.7.1+cu128, diffusers @ pinned commit
    hf\                        #   HF cache (HF_HOME) — 26.6 GB of weights
  flashsr\
    .venv\                     # Python 3.10, third venv
    FlashSR_Inference\         # jakeoneijk/FlashSR_Inference + ModelWeights\ (3.1 GB)
    flashsr_long.py            # chunked runner + accurate cliff detection
  audiosr\
    .venv\                     # Python 3.10, numpy 1.23.5 pinned stack
    audiosr_long.py            # chunked runner + cutoff detection + lowpass
  msst\
    .venv\                     # Python 3.10, modern stack
    MSST\                      # ZFTurbo Music-Source-Separation-Training
    checkpoints\
      apollo_mp3_jusperlee.bin        # MP3 restoration (JusperLee)
      apollo_universal_sr.ckpt        # Universal Super Resolution (Lew)
      config_apollo_universal.yaml
  testaudio\                   # test pair + outputs
```

Two separate venvs are required: AudioSR is frozen on a 2022 stack
(`numpy==1.23.5`, `transformers==4.30.2`, `librosa==0.9.2`) that cannot coexist
with Apollo's modern dependencies.

---

## Usage

### Apollo (recommended for compressed audio)

```powershell
# single file (any format — transcoded automatically)
B:\AudioDev\apollo.ps1 -i "B:\music\track.mp3" -o "B:\music\out"

# whole folder
B:\AudioDev\apollo.ps1 -i "B:\music\album" -o "B:\music\album_restored"

# the Universal SR model (Lew) instead of the MP3-specific one
B:\AudioDev\apollo.ps1 -i "track.mp3" -o "out" -Model universal
```

Output lands in `<out>\<inputname>\restored.wav`.

Two checkpoints are installed:
- `-Model mp3` — JusperLee's, trained on 24–128 kbps MP3. Best for genuinely
  low-bitrate sources.
- `-Model universal` — Lew's Universal Super Resolution, a larger model
  (feature_dim 384 vs 256). Better general-purpose HF restoration.

### AudioSR

```powershell
# band-limited source (not codec-compressed)
B:\AudioDev\audiosr.ps1 -i "old.wav" -o "restored.wav"

# lossy source — detect the real cutoff and lowpass there first
B:\AudioDev\audiosr.ps1 -i "track.mp3" -o "restored.wav" -AutoLowpass

# faster
B:\AudioDev\audiosr.ps1 -i "in.wav" -o "out.wav" -DdimSteps 25

# fewer seams, more VRAM (only if nothing else is using the GPU)
B:\AudioDev\audiosr.ps1 -i "in.wav" -o "out.wav" -Chunk 10.24
```

`-Chunk` defaults to **5.12 s**, verified end-to-end at that default. AudioSR pads
each window up to the next 2.5 s multiple, so `-Chunk 10.24` is really a 12.5 s
window — raise it only on an idle GPU.

A note on the VRAM numbers: `nvidia-smi` shows ~9.5 GB during the chunked run too,
because PyTorch's caching allocator reserves aggressively and doesn't hand memory
back. That reserved figure is not the working set. The difference chunking makes is
empirical, not in the reserved number: same ~9.5 GB reserved, but the chunked run
finished a 15 s stereo file in ~3 minutes at default 50 steps, while the unchunked
run took 19 minutes on the same clip in mono. The monolithic batch overflows into
shared system memory and crawls; the chunked one doesn't.

**Why the wrapper exists:** stock `audiosr` builds one batch for the entire file,
so the working set scales with duration — a 15-second clip already saturated the
3080 and ran 76× slower than realtime. A full track would be far worse (and the
stock CLI also downmixes to mono). `audiosr_long.py` splits into overlapping
windows with an equal-power crossfade, processes channels separately so stereo
survives, and frees the cache between chunks.

AudioSR is generative — change `-Seed` for a different reconstruction.

### FlashSR

```powershell
B:\AudioDev\flashsr\.venv\Scripts\python.exe B:\AudioDev\flashsr\flashsr_long.py `
  -i "track.wav" -o "out.wav"

# bigger batch if the GPU is free; force a known cutoff; use FlashSR's own detector
... -o out.wav --batch 8
... -o out.wav --cutoff 15305
... -o out.wav --detect flashsr
```

FlashSR only accepts exactly 245760 samples (5.12 s at 48 kHz), so `flashsr_long.py`
windows the input and stitches it with the same equal-power crossfade as the AudioSR
wrapper. Its `forward()` takes `[batch, time]` where the leading dim is a real batch,
so windows *and* both stereo channels go through together — that's where its speed
comes from. Note its internal lowpass asserts batch ≤ 2, which is why the wrapper
applies the lowpass itself up front and calls the model with `lowpass_input=False`.

---

## Verified result

Test pair: broadband source (sweep + noise, 22 kHz full-band) encoded to 64 kbps
MP3, which cut it to 11.2 kHz.

| file | rate | ch | cutoff | energy >12 kHz |
|---|---|---|---|---|
| source (original) | 44.1 k | 2 | 22050 Hz | 33.97 % |
| degraded (64 kbps MP3) | 44.1 k | 2 | 11245 Hz | **0.01 %** |
| Apollo (mp3 model) | 44.1 k | 2 | 22024 Hz | **11.84 %** |
| AudioSR stock, unchunked | 48 k | 1 | 24000 Hz | **21.03 %** |
| AudioSR via `audiosr.ps1` | 48 k | 2 | 24000 Hz | **32.57 %** |

The chunked wrapper (`-AutoLowpass`, 25 steps, 5.12 s windows) landed at 32.57 %
against the original's 33.97 % — closer to the source than the unchunked run, in
stereo, in ~2 minutes instead of 19. A repeat at the shipped defaults (50 steps)
gave 29.83 % in ~3 minutes, also exactly 15.000 s and in stereo.

Seam check on the wrapper output — peak sample-to-sample delta at each chunk
boundary, relative to the track's own 99.9th-percentile delta:

| seam | position | vs baseline |
|---|---|---|
| 1 | 4.12 s | 1.09× |
| 2 | 8.24 s | 0.39× |
| 3 | 12.36 s | 1.13× |

No discontinuity spikes — the crossfades are clean. Output duration was exactly
15.000 s, confirming no timeline drift.

Both genuinely reconstruct the missing band rather than just resampling. AudioSR
puts back more energy and reaches higher (24 kHz vs 22 kHz), but collapses stereo
to mono and is generative — it invents plausible detail rather than inferring it,
so results vary by seed. Apollo is conservative and repeatable.

The synthetic test signal is deliberately HF-heavy (sweep + noise); real music has
far less energy above 12 kHz, so don't read the percentage gap as a quality
verdict. Trust your ears on your own material.

Test files are in `B:\AudioDev\testaudio\`. Regenerate the degraded pair with:

```powershell
ffmpeg -y -i source.wav -c:a libmp3lame -b:a 64k degraded.mp3
ffmpeg -y -i degraded.mp3 -ar 44100 -ac 2 -c:a pcm_s16le degraded.wav
```

---

## Surgical high-band splice (`band_splice.py`)

The problem with running any of these tools on an already-good file: they rewrite
the *whole* signal. Measured on this material, they introduce 12–19 % relative error
below the crossover while adding content above it — you pay in the band you can hear
to gain one you can't.

`band_splice.py` avoids that. It splices in the frequency domain: every bin below
the crossover comes from the original, only bins above it come from the model.

```powershell
# 1. make the model extend ONLY above the real cliff
B:\AudioDev\flashsr\.venv\Scripts\python.exe B:\AudioDev\flashsr\flashsr_long.py `
  -i ref.wav -o gen.wav --cutoff 20010

# 2. keep the original below 20 kHz, take only the generated band above it
B:\AudioDev\msst\.venv\Scripts\python.exe B:\AudioDev\band_splice.py `
  --orig ref.wav --restored gen.wav --cut 20010 -o outdir --tag flashsr
```

Emits three files per model:
- `hybrid_<tag>.wav` — original + generated high band. **Audible band drift measured
  at 1.5e-07, i.e. the 24-bit quantisation floor.**
- `band_only_<tag>.wav` — the generated band alone. Above 20 kHz, so it reads as
  silence to human ears; load it in an analyser.
- `band_slowed4x_<tag>.wav` — the same band written at a quarter sample rate, which
  maps 20–24 kHz down to 5–6 kHz. This is the only way to actually *hear* what the
  model invented up there.

Measured on a 30 s excerpt of `Paycheck to Paycheck` (true cliff 20010 Hz):

| | added band level | isolated band RMS | damage avoided below crossover |
|---|---|---|---|
| original | −97.0 dB | — | — |
| + AudioSR | −65.2 dB | −47.3 dBFS | −11.9 dB |
| + FlashSR | −68.6 dB | −42.8 dBFS | −18.7 dB |

Verification that the splice is clean: `hybrid − original` equals the isolated band
exactly (−47.3 / −42.8 dBFS in both measurements), so nothing outside the target
region moved.

### Time-bounded version (`region_fill.py`)

`band_splice.py` applies one crossover to the whole file. `region_fill.py`
generalises it to **rectangles in the time-frequency plane**, so the model's band
can be taken only where the source needs it. It adds a masked *correction* rather
than replacing, which makes untouched audio bit-identical (measured max
difference exactly `0.000e+00` across 74% of a test file), and it handles the
sample-rate change the upscalers introduce. See `studio\README.md` for the
numbers and the UI. Run it from the watermark venv (numpy + scipy + soundfile).

Visually (`Music\spectrogram_20khz_band.png`), the two models differ in character:
AudioSR's fill continues the texture of the band below it, while FlashSR's is louder
but visibly striped — a more synthetic-looking reconstruction.

**Caveat worth stating plainly:** 20 kHz+ is above the hearing limit of essentially
every adult, and most playback chains roll it off anyway. This technique is
technically clean — that is its whole point — but do not expect it to sound
different. It is most useful when the cliff sits *lower* (a 128 kbps file cut at
16 kHz), where the same surgical approach adds audible content without touching what
was already correct.

---

## Which is actually best — the ground-truth test

The synthetic sweep test above shows the tools *do something*, but it can't say
which is **right**, because a generated signal has no "correct" high end. This test
does, by using real music as its own reference:

1. Took a 30 s excerpt of `Paycheck to Paycheck.wav` → **reference** (cliff 20.0 kHz).
2. Encoded it to 96 kbps MP3 and back → **degraded** (cliff 15.3 kHz).
3. Restored the degraded copy with each tool.
4. Measured mean absolute error, in dB, against the reference across **15.3–20 kHz** —
   exactly the band the codec destroyed.

| | mean \|error\| vs reference | rate | notes |
|---|---|---|---|
| degraded 96 kbps (baseline) | 57.07 dB | 48 k | nothing there to measure |
| **AudioSR** | **4.37 dB** | 48 k | closest by a clear margin |
| Apollo universal | 7.00 dB | 44.1 k | good, but overshoots above 20 kHz |
| FlashSR (cliff detector) | 11.34 dB | 48 k | under-restores |
| FlashSR (own detector) | 12.12 dB | 48 k | worse still — see below |

Per-band, the ranking has structure worth knowing: FlashSR is closest just above the
cliff (15–17 kHz), Apollo holds up best at 17–20 kHz, and AudioSR is the most
consistent across the whole band, which is why it wins overall.

Rendered comparison: `Music\spectrogram_comparison.png`.

### ⚠ FlashSR's built-in cutoff detector under-reads

FlashSR lowpasses the input before super-resolving, using its own
`find_cutoff_freq()` — a fixed 98.3rd-percentile of spectral energy. On real music
that lands well below the true cliff:

| file | true cliff | FlashSR's estimate | real audio discarded |
|---|---|---|---|
| 96 kbps excerpt | 15305 Hz | 12820 Hz | 2.5 kHz |
| your 48 kHz track | 20010 Hz | 15773 Hz | 4.2 kHz |

Under-estimating means it filters away real content and then asks the model to
re-invent it. `flashsr_long.py` therefore defaults to `--detect cliff` (steepest
rolloff, the same method as `check_bandwidth.py`), which found 15304 Hz against a
true 15305 Hz. That change alone improved its error from 12.12 → 11.34 dB. Pass
`--detect flashsr` to use the original behaviour.

---

## Before you process anything: check if it needs it

Upresing a file that isn't actually band-limited makes it worse, not better.
Check the real cutoff first:

```powershell
B:\AudioDev\msst\.venv\Scripts\python.exe B:\AudioDev\check_bandwidth.py "yourfile.wav"
```

A hard cliff means lossy encoding and a real candidate. A gentle rolloff usually
means the recording just has no HF content, and there's nothing to restore.

Rough guide to what the cliff position tells you:

| cliff | source | worth upresing? |
|---|---|---|
| 11–13 kHz | 64–96 kbps MP3 | yes, big audible win |
| 15–16 kHz | 128 kbps MP3 | yes, audible |
| 16–19 kHz | 192–256 kbps MP3 | marginal — near the edge of hearing |
| 20 kHz | 320 kbps MP3 / 256k AAC | **no** — that's already above what you can hear |
| none | lossless | no |

### ⚠ Apollo resamples 48 kHz input down to 44.1 kHz

Apollo's config is fixed at `sample_rate: 44100`, and MSST resamples to match.
Feeding it a 48 kHz file **lowers** the ceiling from 24 kHz to 22.05 kHz.

Measured on `Music\Paycheck to Paycheck.wav` (48 kHz, hard cliff at 20 kHz):

| band | original | after Apollo | change |
|---|---|---|---|
| 16–17 kHz | −53.4 dB | −57.2 dB | −3.7 |
| 19–20 kHz | −56.7 dB | −62.9 dB | −6.1 |
| 20–21 kHz | −79.8 dB | −66.6 dB | **+13.2** |
| 23–24 kHz | −82.8 dB | *gone* (past new Nyquist) | — |

It did fill above the 20 kHz cliff, but everything it added is above the limit of
human hearing, while the audible 16–20 kHz region lost 4–6 dB and the sample rate
dropped. **For that file, don't process it.** Use AudioSR (48 kHz out) if you have
a 48 kHz source you genuinely need extended.

---

## Other tools considered (not installed)

| Tool | What it is | Why not installed |
|---|---|---|
| [A2SB](https://github.com/NVIDIA/diffusion-audio-restoration) (NVIDIA) | Diffusion Schrödinger bridge, 44.1 kHz music, bandwidth extension **and** inpainting | **Assessed and deliberately skipped** — see below. |
| [BABE-2](https://github.com/eloimoliner/BABE2-music-restoration) | Blind bandwidth extension for historical recordings | Aimed at unknown/analog degradation (78s, tape), not codec cutoff. |
| [LavaSR](https://github.com/ysharma3501/LavaSR) | Fast Vocos-based speech restoration | Speech-focused. |
| [AERO](https://arxiv.org/pdf/2211.12232) | Spectral-domain SR baseline | Older; Apollo outperforms it on music. |
| [BEHM-GAN](https://github.com/eloimoliner/bwe_historical_recordings) | GAN bandwidth extension for historical recordings | Same niche as BABE-2. |

### Why A2SB was skipped (not yet installed)

It is the strongest technical alternative on paper, but every axis argues against it
for **this** machine and **this** source material:

- **Compute.** It runs an *ensemble of two models* at 50 diffusion steps by default.
  Single-model AudioSR at 50 steps already takes 231 s per 30 s clip here; A2SB is
  structurally ~2× that, on a GPU already shared with ComfyUI.
- **Sample rate.** 44.1 kHz output — it would downsample your 48 kHz sources, the
  same flaw that makes Apollo second-best for your file.
- **License.** NVIDIA OneWay **NonCommercial** — fine personally, a problem if any
  of this touches paid work.
- **Size.** 6.79 GB of checkpoints on top of the 9 GB already downloaded.

It does have one capability nothing else installed has: **inpainting** (filling gaps
/ dropouts, not just extending bandwidth). Worth installing if you ever need that, or
if you want the quality comparison regardless of runtime — say the word and I'll add
it. It is skipped for now rather than judged inferior.

If you want a no-install option to A/B against, [MVSep](https://mvsep.com/algorithms/52)
hosts both Apollo enhancers and the Universal SR model as a web service.

---

## Install notes / gotchas hit

These are already fixed, recorded in case you rebuild:

1. **Python 3.14 will not work.** AudioSR pins `numpy<=1.23.5`, which has no wheels
   past cp311. Installed Python 3.10.11 alongside your 3.14.
2. **`setuptools>=81` breaks librosa 0.9.2** — `pkg_resources` was removed.
   Pinned `setuptools==69.5.1`.
3. **pip resolves numba/scipy to numpy-2-era builds.** Pinned
   `numba==0.56.4`, `llvmlite==0.39.1`, `scipy==1.10.1`.
4. **`audiosr==0.0.7` is missing `matplotlib` from its requirements** — imports
   fail without it. Added `matplotlib==3.7.5`.
5. **The audiosr wheel installs no console script.** Use `python -m audiosr`,
   not `audiosr`.
6. **MSST's `requirements.txt` is the full training stack** (wxpython, pyaudio,
   bitsandbytes, sageattention). Apollo's inference path only needs torch, numpy,
   soundfile, librosa, pyyaml, omegaconf, ml_collections, tqdm, einops,
   pandas, matplotlib — model imports in `settings.py` are lazy.
7. **The Universal SR checkpoint ships no config.** Derived it from the tensor
   shapes (`feature_dim: 384`, `layer: 6`) and verified a clean `load_state_dict`
   with zero missing/unexpected keys.
8. **Both tools write progress bars to stderr**, which PowerShell promotes to a
   terminating `NativeCommandError` under `ErrorActionPreference = 'Stop'` — a
   successful run looks like a failure. Both wrappers now judge success by
   `$LASTEXITCODE` instead. If you script around them yourself, don't pipe them
   through `2>&1`.
9. **AudioSR pads every window to the next 2.5 s multiple** (`round_up_duration`
   in `pipeline.py`), so a 10.24 s chunk returns 12.5 s. `audiosr_long.py` trims
   each window back before crossfading; without that, every seam splices in
   ~2.3 s of padding and the timeline drifts progressively later.
10. **FlashSR's setup scripts are conda/bash** — don't run them. Its deps are just
    the `install_requires` in `setup.py`; `TorchJaekwon` is vendored in the repo, so
    `pip install -e . --no-deps` plus the listed packages is enough (skips wandb and
    tensorboardX, which are training-only).
11. **FlashSR's weights live in a HuggingFace _dataset_ repo**, not a model repo —
    `hf_hub_download(..., repo_type='dataset')` or it 404s.
12. **FlashSR documents `forward()` as `[batch, time]` but its internal lowpass
    asserts the first dim is 1 or 2**, so `lowpass_input=True` silently caps the
    batch. The wrapper does the lowpass itself and passes `lowpass_input=False`.
13. **VRAM readings after 4:41 PM on install day were inflated by ComfyUI**, which
    was holding the GPU concurrently. The pre-ComfyUI measurement (stock AudioSR at
    9.5 GB / 19 min on a 15 s clip) is clean; later peak figures are not, and speed
    numbers across all three tools are pessimistic but mutually comparable.
