# Training a LoRA on your own library

ACE-Step only. **MiniMax-Music3 cannot be finetuned** with what ships today:
its vocoder is decoder-only (121 tensors, all transposed convolutions), there
is no audio encoder anywhere in the roster, and both trainable stages need
audio converted into the model's space — RVQ tokens for the LLM, Flow-VAE
latents for the DiT. Nothing shipped can do that conversion. The same gap is
why `supports_cover=False`: it can't take audio in at inference either. Its
licence permits modification, so if MiniMax publish an encoder this becomes a
code problem rather than a wall. Worth re-checking the repo.

ACE-Step 1.5 is MIT, ships a complete LoRA pipeline, and has the VAE encoder
MiniMax lacks.

## The three steps

```powershell
# 1. build the dataset (CPU, ~2 min for 200 tracks)
cd B:\AudioDev\train
..\studio\.venv\Scripts\python.exe build_dataset.py `
    --src "C:\Users\Kevin\Downloads\Suno Playlist 8-11-2026" `
    --tag xenowiz

# 2. sanity-check it against ACE-Step's own loaders (CPU, instant)
B:\AudioDev\acestep\.venv\Scripts\python.exe validate_dataset.py

# 3. GPU work — the gate refuses unless the card is genuinely free
..\studio\.venv\Scripts\python.exe run_lora.py check
..\studio\.venv\Scripts\python.exe run_lora.py preprocess
..\studio\.venv\Scripts\python.exe run_lora.py train --preset vram_8gb
```

`--src` repeats, earlier sources winning ties:

```powershell
build_dataset.py --src "...\Suno Playlist 8-11-2026" --src "...\Suno Downloads" --tag xenowiz
```

## What build_dataset.py handles

**Sidecars.** Suno writes `<track>.wav.txt` next to each file. The prose part
of it loses newlines and has no play count; the real data is the JSON after
`--- Raw API Response ---`, which carries the style prompt (`metadata.tags`),
the lyrics (`metadata.prompt`), duration, play count, upvotes and clip id.

**The duplicate-download naming trap.** Chrome inserts ` (1)` before the LAST
extension, and the sidecar's name already ends in `.wav`:

```
Song (1).wav          <- second copy of the audio
Song.wav (1).txt      <- its sidecar
```

Neither obvious pattern pairs those. Missing this silently dropped 47 of 215
tracks — 22% of the library.

**Deduplication, in two passes.**

| pass | matches on | why |
|---|---|---|
| exact repeat | same clip `id` | byte-identical audio, keep one |
| sibling take | same title **and** same caption | two clips from one prompt |

The caption half of that key is not optional. Suno titles anything you never
named "Untitled", and this library has **twelve completely different tracks**
sharing that title — different folders, 69 s to 480 s. Keying on title alone
merged all twelve into one and threw away eleven real songs. Requiring the
style prompt to match as well errs toward keeping a near-duplicate, which
costs a little training time, over deleting a distinct song, which is
unrecoverable.

Within a group: highest `play_count` wins, then upvotes, then liked, then the
longer take, then clip id — every step deterministic, so the same folder always
produces the same dataset.

**Durations are probed, not trusted.** A download that stopped halfway still
reports its full length in metadata, and a truncated file is a silent quality
bug. `ffprobe` reads what is actually on disk.

**Length limits.** `--min-seconds 20` drops the short stuff; the
`Game_SFX-*` jingles in this library are 4–5 s and fall out. That is right for
a music LoRA and wrong if you ever want an SFX one — that's a separate dataset,
one `--min-seconds 3 --out sfx.json` run away. At the other end, preprocess's
`--max-duration` **truncates rather than skips** (`audio[:, :max_samples]`), so
the 25 tracks over 4 minutes still train, on their first 4 minutes.

## Labelling: bpm, keyscale, timesignature

**These three are not decoration.** `acestep/training_v2/preprocess_prompt.py:63-73`
formats them into the `# Metas` block of `SFT_GEN_PROMPT`, which reaches the text
encoder in training *and* at inference
(`core/generation/handler/conditioning_text.py:116,168`). The first LoRA run wrote
the literal string `"N/A"` into all three for all 184 samples, so 6.8 GPU-hours
taught the adapter that the trigger word co-occurs with *unspecified* tempo, key and
meter — while the 5Hz planner supplies real values when you generate. Off-distribution
by construction.

```powershell
.\.venv-label\Scripts\python.exe label_audio.py     # ~40 min, CPU only
.\.venv-label\Scripts\python.exe ..\train\resolve_octaves.py
.\..\studio\.venv\Scripts\python.exe build_dataset.py --src "<suno folder>" --tag xenowiz
```

`label_audio.py` writes `labels.json`; `build_dataset.py` reads it and falls back to
`"N/A"` for anything unmeasured. **That fallback is deliberate** — a track whose tempo
could not be established is honestly described as unspecified, whereas a guessed value
binds the trigger word to a fact the audio does not have. Wrong is worse than silent.

`.venv-label` is a separate environment on purpose: `beat_this` pulls its own
torch/torchaudio, and the installer's own rule is that mixing torch builds is how this
box breaks. Nothing in it touches CUDA.

### How each field is decided

| field | method | notes |
|---|---|---|
| bpm | `beat_this` (ISMIR 2024) cross-checked against `librosa.beat.beat_track` | Agreement within 3 BPM → high. Within 5% → mean, high. 2:1 apart → the octave rule below |
| timesignature | bar length ÷ beat length from `beat_this` **downbeats** | Rounded to `VALID_TIME_SIGNATURES` `[2,3,4,6]` with 0.18 beats of slack; anything else refused |
| keyscale | Krumhansl-Kessler correlation over 24 profiles | The model's vocabulary is only major/minor (`KEYSCALE_MODES`), so 24 profiles is the exact shape |

**The octave rule.** Half/double-time is the failure mode of every beat tracker on a
catalogue spanning doom folk and trap EDM. When the two trackers land 2:1 apart they
have agreed on the grid and disagree only about which level is the beat. The candidate
nearest 120 BPM wins — the standard convention, and the one piece of ground truth we
have supports it: *Factory Soul* states 135 BPM in its Suno prompt, the candidates were
68.2 and 136.0, and nearest-120 picks 136.0. Near-equidistant pairs are kept but marked
`confidence: medium` so they can be excluded without re-running the 40-minute pass.

**Relative major/minor.** C major and A minor have identical pitch-class content, so
chroma correlation cannot separate them. When the runner-up *is* the relative key, bass
energy at the two candidate tonics breaks the tie; when it is something else, a small
margin means genuinely undecided and the field stays `"N/A"`.

### Validation

Seven tracks state a BPM in their own Suno prompt — the only external check available.
**6 of 7 land within 6%.** The miss (*Maj*, stated 115, measured 103) is as likely to be
Suno not honouring the request as a tracker error; there is no way to tell from here.

### Coverage, as measured

| field | measured | N/A |
|---|---|---|
| bpm | 165/183 (90%) | 18 — trackers reported unrelated tempi |
| timesignature | 179/183 (98%) | 4 |
| keyscale | 141/183 (77%) | 42 — undecided, runner-up unrelated |

### Two data bugs found while doing this

1. **Lyrics were being used as a style prompt.** Suno's `metadata.tags` is the style
   field and `metadata.prompt` is the lyrics; when `tags` was empty the sidecar fallback
   read the human-readable "Prompt:" section — the lyrics. The fallback now refuses text
   carrying three or more section markers.
2. **One song was in the set twice**, once with its real caption and once with its own
   lyrics as the caption, because the two variants had different captions and so were
   different dedup keys. Fixing (1) collapsed them. 184 → 183 samples, 10.37 h.

### Known limit, not fixed

Lyrics are tokenised at `max_length=512` in training
(`training/dataset_builder_modules/preprocess_lyrics.py:9`) but `2048` at inference
(`conditioning_text.py:142`). 35 of 183 tracks exceed 512, so the model never trains on
their tails but sees full-length lyrics when generating. Trimming would delete real
lyrics; this is ACE-Step's mismatch to live with, recorded rather than papered over.

Six tracks still lose their `# Metas` block to the 256-token caption cap. Measured with
the real Qwen3 tokeniser, not a chars÷3.6 estimate — which had put this at ~21%.

## The GPU gate

`run_lora.py` refuses to start unless *all* of these hold:

- the studio's job lane is idle
- no model is resident in the studio
- ComfyUI is not running
- `nvidia-smi` reports enough free memory

The thresholds are **derived from ACE-Step's own constants**, not guessed:
`gpu_utils.py` puts the 2B decoder weights at 4096 MiB and a conservative
full-finetune forward+backward at 1200 MiB per sample, with a 0.8 safety
factor — so preprocess needs ~3.8 GB and training ~6.5 GB. (A LoRA at
batch_size=1 with gradient checkpointing is comfortably under that batch
figure, so 6.5 GB is an upper bound.) An earlier hardcoded 8.0 GB would have
refused runs that actually fit.

Training is the only thing on this box that runs for hours and cannot yield,
and a run that starts against a busy card doesn't fail cleanly — it OOMs an
hour in, or quietly thrashes to shared system memory and takes ten times as
long. `--force` exists; don't.

**On this box the studio is rarely the blocker.** Unity, two browsers and
Discord sit on ~3.7 GB of the 10 GB card between them, which is the difference
between a preset fitting and not, so the gate names the apps to close rather
than just printing a number.

## Presets

`acestep/training_v2/presets/`. On a 10 GB 3080 with a clean desktop:

| preset | rank | notes |
|---|---|---|
| `quick_test` | 16 | 10 epochs. Use this first — proves the pipeline, not the LoRA |
| `vram_8gb` | 16 | aggressive savings, encoder offloaded to CPU |
| `vram_12gb` | 32 | 8-bit optimizer; needs `bitsandbytes` (installed) |

`run_lora.py --preset NAME` reads those JSONs and **expands them into CLI
flags itself**, because `train.py fixed` has no `--preset` option — the presets
are data the UI consumes. Every key does have a matching flag, so the expansion
is mechanical, but passing the name straight through fails instantly at argparse
(discovered before the first run, not after it).

Use `train.py fixed`, never `vanilla` — upstream's own docstring calls vanilla
bugged, kept only for backward compatibility. `train.py estimate` runs a
gradient-sensitivity analysis without training.

## Known gaps

- **The gate checks once, at the start.** Training then holds the card for
  hours without registering anywhere the studio can see, so the studio's own
  `gpu_idle()` check does not know training exists. Start a render from your
  phone mid-training and MiniMax will happily load into a card that is 95%
  full and thrash to shared system memory. **Don't generate while training
  runs.** A real lock is possible; it isn't built.
- **Nothing loads the LoRA at inference yet.** `load_lora_weights` exists in
  `acestep/training/lora_checkpoint.py`, but neither ACE-Step's pipeline nor
  this studio's `acestep_worker.py` takes a LoRA argument. Training and using
  are two separate tasks today.
- **The source material is AI-generated**, so the LoRA learns Suno's character
  as well as yours — including whatever vocoder artefacts it leaves. That's a
  property of the dataset, not a bug in the pipeline.
- A LoRA on the DiT's `q/k/v/o` is a **style and production adapter**. It is
  not a voice clone and it will not reproduce specific songs.
