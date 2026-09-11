# RESEARCH — emotional control, timing/structure, and whether to build our own

Read-only investigation, 2026-08-22. Nothing was trained, generated, installed or
modified. Every ACE-Step claim below cites `path:LINE` from the local source and was
read, not recalled. External claims carry URLs. Sentences that infer rather than
confirm say so.

---

## 0. The short answer

**A — what is missing.**

*Before the detail, one external fact that should reset expectations.* Across two
independent 2025-26 studies, **arousal is controllable in text-to-music and valence is
essentially not.** AImoclips (991 clips, 6 systems, 111 participants, 6,162 ratings)
finds every system "exhibits a centralizing tendency toward emotional neutrality", with
valence differentiation **collapsing entirely at low arousal** — "calm" and "gloomy"
come out indistinguishable. LARA-Gen measures text-only conditioning at **valence CCC
0.06, arousal CCC 0.23**. Full citations in §7.1. Practical consequence: chase
**energy, intensity and contrast** — those respond. Treat "make it sadder" as an open
research problem, not an engineering task, and spend the effort budget accordingly.

*And one about the vendor.* **There is no ACE-Step 2.0**; the last release was XL on
2026-04-02. Two community issues describing precisely our complaint — flat vocals, no
pitch bending, uniform vibrato, **"insufficient difference between verse and chorus
intensity"**, timing **"overly grid-aligned… lacks human-like micro-timing variation"**
— were **closed as "not planned" with no maintainer reply** (§7.3). Nothing here gets
fixed by waiting.

*Emotion.* The architecture has exactly three conditioning channels into the DiT
(`models/turbo/modeling_acestep_v15_turbo.py:1513-1556`): a text stream, a lyric
stream, and a **timbre stream fed from reference audio**. Emotion can only enter
through prose in the caption or through words inside the lyrics. There is no
valence/arousal input, no emotion embedding, and the 5Hz planner's output schema is a
hard-coded FSM with six fields — bpm, caption, duration, keyscale, language,
timesignature (`constrained_logits_processor.py:55-79`). **There is no emotion field
anywhere in the stack.** What is missing is not model capacity; it is (1) a
conditioning channel we are not using at all — the timbre/reference-audio path is fed
literal zeros during training (`training/dataset_builder_modules/preprocess_encoder.py:17`)
— and (2) any emotion vocabulary in our data, which is worse than we thought (see B).
Note that ACE-Step's *documented* tag vocabulary is far richer than our data uses —
`[Chorus - anthemic]` modifier syntax and inline `[melancholic]` / `[building energy]`
tags are official, supported, and absent from 180 of our 184 samples (§7.2).

*Timing/structure.* Two real limits. First, 12 of the DiT's 24 self-attention layers
are **sliding-window, ±128 tokens** (`checkpoints/acestep-v15-turbo/config.json`
`layer_types` + `sliding_window: 128`; applied at
`modeling_acestep_v15_turbo.py:290,368`). At 25 Hz latents with `patch_size: 2` that is
±10.2 seconds of context. Only the 12 full-attention layers can relate a chorus at 0:30
to a chorus at 2:30. Second, and much more actionable: the thing that actually plans
long-range structure is the **5Hz LM**, which emits a semantic sketch of the whole song
as audio codes that are decoded to 25 Hz and injected as the DiT's source latents
(`modeling_acestep_v15_turbo.py:1685-1696`). We can neither train it (ACE-Step's own
tooling explicitly skips it, `training_v2/model_discovery.py:62`) nor, in the studio,
even reliably turn it on with the caption-rewrite enabled.

**B — how to fine-tune/expand beyond the LoRA we have.**

The LoRA is not the problem. It wrapped **both** self- and cross-attention correctly
(verified from the safetensors header — 384 tensors, 24 layers × {self_attn,
cross_attn} × {q,k,v,o}). The problems are in the data and in the run config:

1. **`bpm`, `keyscale` and `timesignature` are the literal string `"N/A"` in all 184
   samples.** Hard-coded at `train/build_dataset.py:290`. These fields *do* condition
   the model — they are formatted into the prompt at
   `training_v2/preprocess_prompt.py:63-73` and encoded by the text encoder at
   `training_v2/preprocess.py:271`. We built a live channel and fed it nothing for
   every single sample. This is the single cheapest fix on this page.
2. **~21% of training prompts overflow the 256-token caption cap** (estimated,
   chars/3.6), and because `SFT_GEN_PROMPT` puts `# Metas` **last**
   (`constants.py:160-168`), truncation deletes bpm/key/timesig *first*.
3. **~22% of lyric blocks overflow the 512-token training cap**
   (`training/dataset_builder_modules/preprocess_lyrics.py:9`) while inference allows
   2048 (`core/generation/handler/conditioning_text.py:142`) — a train/infer mismatch
   that lands squarely on long songs, i.e. exactly where structure matters.
4. Rank 16 was the floor preset for a 10 GB card, and loss was flat from epoch ~30
   (0.7760) to epoch 100 (0.7588) — **~4.8 of the 6.8 hours bought 0.017 of loss.**
5. MLP (`gate_proj`/`up_proj`/`down_proj`) was never targeted — that is the real
   expansion, not cross-attention.

**C — build our own?** From scratch, no — by a factor of 45× on data and ~$50–82k on
compute (§6.2d, real numbers, ACE-Step v1 discloses enough to price). But **two of the
"build" options are far cheaper than they sound, and one of them I initially got wrong:**

- **A full fine-tune of the 2B DiT is a ~$100 experiment, not a $50,000 one.** MusicGen
  Songstarter v0.2 full fine-tuned a **3.3B** model on **7–8 hours** of audio in **48
  GPU-hours ≈ $62–96** at today's prices. We have 9.88 hours. The real objection is
  catastrophic forgetting and checkpoint management, not cost. Worth one rented
  8×A100 day **after** the data is fixed, as a benchmark against the best LoRA (§6.2c).
- **A personal reward model on our 215 play counts is published, not speculative.**
  MuQ-Eval's own ablation shows a LoRA-adapted evaluator reaching **SRCC 0.761 from 150
  clips**, with the abstract stating that this "enables personalized quality evaluators
  from individual listener annotations." I wrote this off in an earlier draft; it is the
  best use of the play counts, and §5.1 has the correction and the caveats.

**The one generative component worth training ourselves is still the 5Hz planner** — it
is a stock `Qwen3Model` 1.7B
(`checkpoints/acestep-5Hz-lm-1.7B/config.json`), the audio→codes encoder we need to
build its training set already exists on disk
(`core/generation/handler/audio_codes.py:68`), and it is the only part of the stack that
reasons about whole-song structure. It is also the **industry-standard pattern**, not a
guess: SegTune fine-tunes a Qwen3-4B for precisely this job — predicting section
durations — and Seed-Music argues from first principles that separating the arrangement
planner from the dense token model is what makes structure control work at all (§7.2).
Numbers in §6.

**And one thing to try before any of that.** The tech report says the planner emits a
`structure` field; the shipped constrained decoder allows six fields and `structure` is
not one of them. `use_constrained_decoding` is a parameter with a default of `True`
(`inference.py:207`). If the planner was trained to plan sections and a logits mask is
suppressing it, that is the timing feature we want, already paid for. **One generation
settles it.** Option 0 in the table.

---

## 1. Ranked options

Wall-clock figures for LoRA work are extrapolated from our own measured run:
**246.8 s/epoch** at rank 16, 184 samples, batch 1 × grad-accum 8, 23 optimizer
steps/epoch, 11,010,048 trainable params (`train/logs/train-20260820-170000.log`).
Rank changes the adapter size, not the base forward/backward, so epoch time is
dominated by epochs, not rank — that is an inference, not a measurement.

### DO NOW — fits the 10 GB card

| # | Option | What it buys | Data needed | VRAM + wall-clock | Risk |
|---|---|---|---|---|---|
| 0 | **Turn `use_constrained_decoding` off once and read the planner's raw output** | Possibly the whole timing feature, for free. The tech report says the planner emits a **`structure`** field; the shipped FSM (`constrained_logits_processor.py:55-79`) has no such state and masks everything outside six fields. If the model was trained to plan sections, a logits mask is currently suppressing it | 0 | One generation. `inference.py:207` | **Zero.** Five minutes. Do this before anything else on this page |
| 1 | **Label bpm / keyscale / timesignature** and rebuild the dataset | Real tempo, key and meter control. Today the adapter has learned "trigger word ⇒ these are unspecified", which is actively off-distribution when the planner supplies real values. **Values must land inside `VALID_KEYSCALES` (70 options) and `VALID_TIME_SIGNATURES` `[2,3,4,6]`** (`constants.py:35-41,66`) | 0 new audio. Label the 184 we have — **beat-this** for beats/downbeats and **SongFormer** for sections (§7.2); key needs a third tool | Labelling: CPU or ~4 GB. Retrain 30–50 epochs ≈ **2.0–3.4 h**, ~6.5 GB | **Low.** Pure data fix |
| 1b | **Add a structured mood prefix inside the caption** — `xenowiz, [mood: X \| arousal: Y \| valence: Z], ...` | A consistent, greppable emotion channel with **zero code change**. The caption is free text on both the training and inference side (§2.9); the `# Metas` block is not | Label 184 moods by hand (~2 h) or via option 8 | Rides on the option 1 retrain, no extra cost | Med. That it yields a usable dial is inferred, not verified |
| 1c | **Add per-section emotion to the lyric structure tags** — `[chorus – resigned, heavy]` | Emotion that is *time-localised* rather than global. 4× the token budget of the caption, and time-aligned by construction. 4 of our samples already do this (§2.8) | Edit lyrics in `dataset.json` | Rides on the option 1 retrain | Med. Inferred |
| 2 | **Trim captions under ~200 est. tokens; cap lyrics ~450 est. tokens** | Stops 21% of samples losing their metadata block and 22% losing their lyric tail | 0 | Preprocess only, ~30 min, 3.8 GB | Low |
| 3 | **Wire `repaint` into `studio/workers/acestep_worker.py`** | Section-level surgery: "regenerate 1:04–1:32". This is the most direct timing/structure control in the stack and it works on **turbo** | 0 | Code only. No training | Low. Already declared in `web/api.py:1657` |
| 4 | **Wire `base` variant tasks: `complete`, `extract`, `lego`** | `complete` = continue a track that ends mid-phrase; `extract` = stems; `lego` = add a named instrument layer over existing audio; plus real CFG | 0 | Base needs **7.3 GB** (`web/api.py:1665`) — it fits today. Slower: 32 steps vs turbo's 8 | Low–med. Our LoRA is turbo-only |
| 5 | **Expose `flow_edit_morph`** | Re-colour an existing render toward a new caption while keeping its arrangement. The closest thing to an "make this sadder" dial that exists | 0 | Code only | Med. v1 disables DCW/heun/ADG inside the loop (`inference.py:174-183`) |
| 6 | **Turn thinking on and set `use_cot_caption=True`** | Planner writes a richer caption than ours, picks bpm/key/meter, and emits a whole-song structural sketch that conditions the DiT | 0 | ~6.8 GB **system RAM**, +25 s/gen (`web/REQUIREMENTS.md` §13) | Low. Currently defaults to `False` in the worker |
| 7 | **Best-of-N reranking** | Quality lift with **zero training**, and the **best-evidenced item in this report**: TuneJury measured N=1→32 on **ACE-Step v1.5 Turbo specifically**, reward rising monotonically, **+0.178 to +0.291 per doubling at N=4→8** (§5.1). ACE-Step's own PMI + lyric-alignment scorer is already on disk (§2.2) | 0 | N× generation time. Takes already exist | **Low.** Not wired into our studio at all. Stop around N=8–16; the same paper shows structural drift at extreme N |
| 7b | **A personal reward model — LoRA-adapt MuQ-Eval on the 215 play counts** | Turns the play counts into a *reranker* instead of gradient updates. **Published floor is 150 clips → SRCC 0.761** (§5.1). Pair it with an off-the-shelf scorer, which sees generation artifacts our library never will | The 215 play counts we already have | Small; MuQ-Eval weights are open (`https://github.com/dgtql/MuQ-Eval`) | Med. Our labels are play counts, not expert MOS — noisier than the paper's |
| 8 | **Auto-recaption the catalogue** with `understand_audio_from_codes` | Captions in the *model's own* caption distribution instead of Suno style-prompt distribution, plus structure-tagged lyrics derived from the audio | 0 new audio | ~4–6 GB, est. 10–20 s/track ⇒ ~1 h for 184 | Med. Needs an A/B against the current captions |
| 9 | **Stop at epoch 30–50; go rank 32 (`vram_12gb` preset)** | Same or better adapter in ~⅓ the time; rank 32 doubles capacity | 0 | ~6.5 GB, **2.0–3.4 h** | Low. `bitsandbytes` installed; preset needs a manual `--preset vram_12gb` |
| 10 | **Add MLP targets** (`gate_proj up_proj down_proj`) | The genuine capacity expansion. Attention-only LoRA moves *routing*; MLP moves *content* | 0 | **+9,437,184 params at rank 16** (11.0M → 20.4M, +86%); VRAM +~0.3 GB est. | Med. `train.py estimate` **cannot rank MLP** — it only scans q/k/v/o (`training_v2/estimate.py:236-246`) |
| 11 | **Curate by play count — drop the bottom decile — then duplicate the top, capped at ~3×** | Bends the fit toward the tracks you actually replay. **Curation is the stronger half**: Emu got an 82.9% win rate from 2,000 curated images (§5). No code change either way: the loss is a plain unweighted MSE (`training_v2/fixed_lora_module.py:354`), so repetition is the only weighting hook | 0 | Longer epochs proportional to duplication | Med. Over-duplication overfits the top ~20 tracks; reward-weighted regression has a published record of *degrading* quality |
| 12 | **Train 3–5 small mood LoRAs, switch/blend at runtime** | An actual emotion dial. Per-adapter scaling is implemented and in-place free (`core/lora/registry.py`, `core/lora/scaling.py`) | Partition the 184 by mood | 3–5 × 2 h | **Med–high.** ~40–60 tracks per mood is thin, and see §2.4: true *simultaneous* blending needs a one-line patch, and switching costs a 40 s worker restart |
| 13 | **SFT the 5Hz planner (QLoRA on 1.7B)** | The only lever that touches whole-song structure. Teach it *your* section lengths, intro/outro habits and arrangement pacing. **This is the industry pattern** — SegTune fine-tunes Qwen3-4B for exactly this job (§7.2) | 184 (codes ↔ caption/lyrics) pairs — buildable locally | Est. 8–10 GB QLoRA; encode ~1 h + train ~1–2 h | **High.** No official tooling, confirmed at three levels externally (§7.3). We would write the trainer. See §6 |
| 14 | **Run ACE-Step 1.5 XL (4B) quantised, today** | The 4B quality tier on the current card. GGUF Q4_K_M is **2.8 GB**; `acestep.cpp` runs the LM and the DiT with LoRA support (§7.4) | 0 | ~3–9 GB depending on quant | Med. README does not list XL variants despite the GGUF repo publishing them; our turbo LoRA will **not** transfer to XL |
| 15 | **Add DiffRhythm 2 as a second engine** (Apache-2.0, ~6–8 GB) | **Timestamped LRC lyrics as a primary input** — real per-line timing control that ACE-Step structurally cannot accept (§7.2, §7.4). The studio is already multi-engine | 0 | Fits today | Med. New engine, new worker, new quality profile to learn |

### WAITS for the bigger card

| # | Option | What it buys | Data | VRAM + wall-clock | Risk |
|---|---|---|---|---|---|
| 16 | XL turbo / XL base (4B) in bf16 | Higher ceiling on everything, no quantisation loss | 0 | 11.5–12 GB (`install/models.json`); the upstream README says ~9 GB — **unresolved discrepancy, neither figure measured** (§8.14); 20 GB on disk each | Low |
| 17 | **HeartMuLa-oss-3B** (`https://huggingface.co/HeartMuLa/HeartMuLaGen`) | **Mood as a first-class structured tag + per-section natural-language style prompts** — the closest thing in the field to the requirement. Apache-2.0, the cleanest licence here | 0 | 14–20 GB full bf16; 10–14 GB with model swap; a 4-bit build exists but is validated only for 16 GB | Med. **No training code.** Community traction thin (924 downloads/mo) |
| 18 | Rank 64 / 128 presets | Capacity for the whole palette in one adapter | 0 | 16 GB → rank 64; 24 GB+ → rank 128, batch 2 | Low |
| 19 | Re-train the LoRA on XL | Our style on the better base | 0 | Est. 16 GB+; est. 2–3× our epoch time | Med |
| 20 | **Unfreeze the condition encoder** (lyric + timbre encoders, text projector) | The only way to change how *text and reference audio* are interpreted rather than how latents are denoised. `stable-audio-controlnet` is a working reference implementation of this shape on a frozen audio DiT (§7.2) | 0 | Requires patching `training_v2/preprocess.py` to stop pre-baking `encoder_hidden_states` — see §3.4. Est. 16–24 GB | **High.** We would be off ACE-Step's supported path |
| 21 | **Full fine-tune of the 2B DiT — as a one-off benchmark** | Everything a LoRA cannot reach, and it answers "is the ceiling the adapter or the data?" | Our 9.88 h is enough: a 3.3B model was full fine-tuned on **7–8 h for 48 GPU-h** (§6.2c) | Does not fit 10 GB. **~24 GB floor**; a rented 8×A100 day ≈ **$62–96** | High. Forgetting risk + a 4.8 GB checkpoint to version. **Do it after options 1/2/9, never before** |
| 22 | LoKr instead of LoRA | Higher expressivity per parameter | 0 | Needs `pip install lycoris-lora` — **currently not installed** (`train/logs/train-20260820-170000.log`) | Med |
| 23 | **Muse + MuseCritic GRPO** (`https://arxiv.org/abs/2608.11755`) | The **only open route to train *toward* expressiveness** rather than around it: a released aesthetic reward model plus a GRPO directory. Different base model, so it is a migration, not an addition | Muse's own corpus + ours | Unpublished VRAM | **Very high.** A reproducibility artifact, not a tool — ~120 stars, all issues from Jan–Feb 2026 |

---

## 2. What we already have and are not using

This is the most valuable section. Everything here is on this disk, paid for, and dark.

### 2.1 The base checkpoint is installed and its three extra tasks are unreachable

`acestep-v15-base` is present (`acestep/checkpoints/acestep-v15-base/`, 4.79 GB,
downloaded 2026-08-20) and `web/api.py:1665` prices it at **7.3 GB to run** — it fits
the 10 GB card *today*. `constants.py:88` gives base `extract`, `lego` and `complete`
on top of the turbo four. `web/api.py:1711-1716` already **declares** those
capabilities to the client.

But `studio/workers/acestep_worker.py` only ever sends two task types — `"cover"`
(line 253) and `"text2music"` (line 260). **Repaint, complete, extract and lego are
declared and never issued.** What each does, traced:

- **`repaint`** — `core/generation/handler/conditioning_masks.py:52-58` builds a boolean
  mask over latent frames `[start*48000/1920, end*48000/1920)`, silences the source
  inside it (line 92), regenerates only that span, then crossfades the boundary
  (`modeling_acestep_v15_turbo.py:1531-1556`, `repaint_latent_crossfade_frames`
  defaults to 10 frames ≈ 0.4 s). **Works on turbo.** This is "the second chorus is
  wrong, redo 1:04–1:32" — bar-accurate structural editing, available now, unexposed.
- **`lego`** — the same masking, but line 90-92 skips the silencing when
  `task_type == "lego"`, so the source audio stays under the new material. Combined
  with the instruction `"Generate the {TRACK_NAME} track based on the audio context:"`
  (`constants.py:145`) and the 12 legal track names (`constants.py:153-156`:
  woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums,
  backing_vocals, vocals), this is **"add a string section under bars 33–48"**. Base only.
- **`complete`** — instruction `"Complete the input track with {TRACK_CLASSES}:"`
  (`constants.py:147`, built at `core/generation/handler/task_utils.py:92-97`).
  Continuation of a track that ends mid-phrase. Base only. `REQUIREMENTS.md` §14
  already identified this gap.
- **`extract`** — `"Extract the {TRACK_NAME} track from the audio:"`. Stem separation
  through the model itself rather than Demucs. Base only.

### 2.2 A complete reward model, already written, never called

`acestep/core/scoring/` ships three scorers and a combiner:

- `lm_score.py:377` `calculate_reward_score(scores, weights)` — a weighted composite,
  default **caption 50% / lyrics 30% / metadata 20%**, with dynamic renormalisation
  when a component is missing.
- `lm_score.py:21` `pmi_score` — pointwise mutual information between the generated
  audio codes and the condition, i.e. "did this render actually match the prompt",
  measured by the 5Hz LM. Normalised through a sigmoid (`lm_score.py:64`).
- `dit_score.py:15` `MusicLyricScorer` — lyrics-to-audio alignment quality from the
  DiT's **cross-attention energy matrices**: Coverage, Monotonicity, Path Confidence.
  It even masks structural tags out of the scoring (`dit_score.py:32-44`).
- `dit_alignment.py:39` `MusicStampsAligner` — per-token and per-sentence timestamps in
  LRC format via bidirectional consensus + DTW.

Public entry points exist on the handler: `get_lyric_score`
(`core/generation/handler/lyric_score.py:15`) and `get_lyric_timestamp`
(`core/generation/handler/lyric_timestamp.py:15`). PMI scoring is wired into ACE-Step's
own Gradio UI at `ui/gradio/events/results/scoring.py:102`. **Our studio calls none of
it.**

One practical caveat, verified from the signature
(`core/generation/handler/lyric_timestamp.py:15-27`): it takes `pred_latent`,
`encoder_hidden_states`, `encoder_attention_mask`, `context_latents` and
`lyric_token_ids` — the live tensors from a generation. It cannot be run on a saved
`.flac` after the fact. The scorer has to be called **inside the worker**, right after
`generate_music` returns, while `extra_outputs` still holds the tensors. That is a
wiring job in `studio/workers/acestep_worker.py`, not a standalone script.

Two consequences, both large:

1. **Best-of-N for free.** Render N takes, score them, keep the best. No training, no
   labels, no preference data. This is the honest answer to "what can 215 play counts
   support" — see §5. **The N already exists**: `web/REQUIREMENTS.md` §13 documents the
   takes mechanism, `max_takes()` returning 2 on 10 GB / 4 at 16 / 8 at 24+, with each
   take reseeded into a genuinely different song. Best-of-N is "score the takes we
   already render and surface the winner first", not a new feature.
2. **A timing metric we can actually optimise against.** `MusicStampsAligner` gives
   real numbers for whether the lyrics land where they should. Right now "timing is
   off" is a feeling; this turns it into a measurement, which is the precondition for
   improving it deliberately.

### 2.3 Inference levers in `GenerationParams` that the studio never sets

Read from `inference.py:99-212`. The worker sets nine of these
(`studio/workers/acestep_worker.py:235-262`). The rest are dark:

| Parameter | `inference.py` line | What it does |
|---|---|---|
| `flow_edit_morph` + `flow_edit_source_caption` / `_source_lyrics` | 174-184 | Integrate `V_target(new caption) − V_source(old caption)` over a noise interval. **Morph an existing take toward a different emotional description while keeping the song.** |
| `repainting_start` / `repainting_end` / `repaint_mode` / `repaint_strength` | 163-168 | Section surgery. `repaint_mode` ∈ conservative/balanced/aggressive |
| `chunk_mask_mode` | 165 | `"explicit"` = honour the repaint range; `"auto"` = mask 2.0, model decides |
| `retake_seed` / `retake_variance` | 170-172 | Variance-preserving variation — a *nudge* rather than a reroll |
| `lm_cfg_scale` (2.0), `lm_negative_prompt` ("NO USER INPUT") | 199-200 | **Classifier-free guidance on the planner.** A negative prompt for structure/vibe |
| `use_cot_caption` | 205 | Planner rewrites the caption. Default `True` upstream, **`False` in our worker** (`acestep_worker.py:248`) |
| `use_cot_lyrics` | 206 | Planner writes lyrics |
| `instruction` | 100 | Overrides `"Fill the audio semantic mask based on the given conditions:"`. Free-text steering of the whole task framing; exposed in ACE-Step's own Gradio UI (`ui/gradio/interfaces/generation_tab_primary_controls.py:61`) |
| `guidance_scale` (7.0), `use_adg`, `cfg_interval_start/end` | 136-139 | Prompt adherence vs. musicality. Only honoured on base |
| `velocity_norm_threshold`, `velocity_ema_factor` | 143-144 | Sampler stabilisation; docstrings suggest 2.0 and 0.1 |
| `dcw_*` (5 params) | 152-156 | Wavelet-domain per-band correction, tuned by upstream grid search |
| `timesteps` | 158-160 | Custom noise schedule, overrides steps and shift |
| `latent_shift` / `latent_rescale` | 132-133 | Post-DiT latent scaling |
| `reference_audio` | 84 | **The timbre channel.** Distinct from `src_audio` |

### 2.4 Multi-LoRA with per-adapter scaling — mostly there, with two sharp edges

`core/generation/handler/lora/lifecycle.py:191` `add_lora(path, adapter_name)` tracks
adapters in a `_active_loras` dict (line 290); line 328 adds a dedicated `"voice"` slot;
`core/lora/registry.py:13` `build_lora_registry` builds an explicit adapter→target map
and `core/lora/scaling.py:14` `apply_scale_to_adapter` sets a per-adapter scale
deterministically. Our worker loads exactly one (`acestep_worker.py:223`).

Two edges, both verified:

- **Activation is single-adapter as written.** Line 300 calls
  `self.model.decoder.set_adapter(effective_name)` with a bare string. PEFT accepts a
  list there, so genuine simultaneous blending is a one-line change — but it *is* a
  change, not a flag. Loading a second adapter today switches the active one.
- **`remove_lora` is broken and this is already known.** `web/REQUIREMENTS.md` §11
  records it: PEFT 0.20's `delete_adapter` raises a bare `KeyError` for the adapter it
  was just given, measured twice, so the supervisor **restarts the worker (~40 s)** on
  any adapter change. **Scale changes within the same adapter are in-place and free** —
  which is the good news for a mood dial: load once, vary the scale continuously.

Same section records the weight-level acceptance test worth reusing: 192 adapted
modules, delta `B@A*scaling` = 1.2359 at scale 2.0 → 0.3090 at 0.5, ratio exactly 0.25.
192 modules × 2 tensors = the 384 in §4.1 — the numbers agree.

### 2.5 An audio→caption→metadata auto-labeller

`core/generation/handler/audio_codes.py:68` `convert_src_audio_to_codes(audio_file)`
turns any audio file into a 5Hz code string. `llm_inference.py:1850`
`understand_audio_from_codes(codes)` runs the planner in reverse and returns
**bpm, caption, duration, keyscale, language, timesignature and lyrics** — the docstring
example at 1891-1894 shows lyrics coming back with `[Intro: ...]` structure tags.
`training/dataset_builder_modules/label_single.py:36-90` is the batch driver, and there
are HTTP routes for it (`api/train_api_dataset_auto_label_*.py`).

This is the fix for option 1 and the enabler for options 8 and 13, and it is already
written.

### 2.6 `train.py estimate`

`training_v2/estimate.py:26` `run_estimation()` measures per-module gradient
sensitivity on **our** preprocessed tensors using the real training loss surface, and
ranks the top-K modules. Free, data-driven target selection. Caveat, verified:
`_find_attention_modules` (line 236-246) only matches `q_proj|k_proj|v_proj|o_proj`, so
**it cannot tell us whether MLP is worth targeting.** Use it to prune, not to expand.

### 2.7 Preference signal we parse and then throw away

`train/build_dataset.py:81-82` reads `upvote_count` and `is_liked` from the Suno
sidecar. Line 150 uses them as dedup tiebreaks. They are then **not written to
`dataset.json`** — verified: sample keys are exactly `_ch, _id, _instrumental, _model,
_play_count, _sr, _title, audio_path, bpm, caption, custom_tag, duration, filename,
keyscale, lyrics, timesignature`. Two preference signals discarded at the door.

### 2.8 Emotion annotations already sitting in the lyrics

171 of 184 samples carry bracketed structure tags. 180 `[chorus]`, 89 `[pre-chorus]`,
75 `[verse 2]`, 64 `[instrumental]`, 62 `[bridge]`. And a handful already carry
**emotion in the section tag itself**: `[chorus – viscous, heavy]`,
`[verse 1 - sexy and sassy]`, `[pre-chorus - bitterness rising]`,
`[instrumental / wordless hook]`. The lyric stream gets 2048 tokens at inference — four
times the caption's budget — and it is time-aligned by construction. **Per-section
emotion tags in the lyrics are a structure-aware emotion channel that costs nothing and
is already partially populated.** That it works is an inference; that the channel
exists and has the capacity is verified.

### 2.9 The `# Metas` block is free text at inference — but not in training

`core/generation/handler/metadata_utils.py:41-54` `_parse_metas`: if the caller passes a
**string**, it is used verbatim (lines 47-48). Only a *dict* gets normalised down to the
four fixed keys by `_dict_to_meta_string` (lines 22-38). So at inference nothing stops us
writing:

```
- bpm: 128
- timesignature: 4
- keyscale: F# minor
- duration: 180 seconds
- emotion: resigned, low arousal
- structure: intro 0-12 | verse 12-45 | chorus 45-70 | ...
```

**But training does not have that freedom.** `training_v2/preprocess_prompt.py:68-73`
hard-codes exactly four lines. Adding a fifth field to the *training* prompt means
patching an ACE-Step file — a fork, not a flag — and a conditioning field the model
never saw in training does approximately nothing at inference.

**The zero-fork route is the caption.** It is free text, it is the field the LoRA
actually reshapes, and it is under our control on both sides. A machine-readable prefix
inside the caption — `xenowiz, [mood: resigned | arousal: low | valence: low], ...` —
gives a consistent, greppable emotion channel with **no code change anywhere**, in both
`dataset.json` and the studio prompt box. That this would produce a usable dial is an
inference; that the channel exists and costs nothing is verified.

### 2.10 Free settings we are leaving off

- `--sample-every-n-epochs` defaults to 0 (`training_v2/cli/args.py:314`). Turning it on
  gives audible A/B checkpoints during a 7-hour run instead of a loss curve.
- `vram_12gb` (rank 32, `adamw8bit`, encoder offload) is one flag away and
  `bitsandbytes` is installed; `run_lora.py`'s `auto` maps our 10 GB card to `vram_8gb`
  (rank 16) because the threshold is 11.0 GB (`train/run_lora.py`, `PRESET_BY_TOTAL_GB`).
- `--attention-type` accepts `self` / `cross` / `both` (`training_v2/cli/args.py:290`,
  resolved at `training_v2/cli/validation.py:60-98`). A **cross-attention-only** LoRA is
  a one-flag experiment that isolates text conditioning — plausibly the right shape for
  a mood adapter. That it is the right shape is an inference.

---

## 3. Where emotion and timing actually enter — the verified trace

### 3.1 Three conditioning streams, packed into one cross-attention sequence

`AceStepConditionEncoder.forward` (`modeling_acestep_v15_turbo.py:1531-1556`):

```
text_hidden_states  --text_projector (Linear 1024->2048)------.
lyric_hidden_states --AceStepLyricEncoder (8 layers)----------+--pack_sequences--> encoder_hidden_states
reference audio     --AceStepTimbreEncoder (4 layers)---------'                          |
                                                                                         v
                                                              DiT cross_attn (24 layers, always full attention)
```

Layer counts from `checkpoints/acestep-v15-turbo/config.json`:
`num_lyric_encoder_hidden_layers: 8`, `num_timbre_encoder_hidden_layers: 4`,
`num_hidden_layers: 24`, `hidden_size: 2048`, `text_hidden_dim: 1024`,
`patch_size: 2`, `timbre_fix_frame: 750`.

A fourth channel bypasses cross-attention entirely: `context_latents` is concatenated
onto the noisy latents at `modeling_acestep_v15_turbo.py:1349`. That is where
repaint masks, cover source audio and the planner's structural sketch arrive.

### 3.2 The text channel and its 256-token ceiling

The caption is not sent raw. It is formatted into `SFT_GEN_PROMPT`
(`constants.py:160-168`):

```
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
{custom_tag}, {caption}

# Metas
- bpm: {bpm}
- timesignature: {timesignature}
- keyscale: {keyscale}
- duration: {duration} seconds
<|endoftext|>
```

Built identically in training (`training_v2/preprocess_prompt.py:63-74`) and inference
(`core/generation/handler/conditioning_text.py:116`). Encoded by
**Qwen3-Embedding-0.6B** (28 layers, hidden 1024,
`checkpoints/Qwen3-Embedding-0.6B/config.json`) and truncated at **256 tokens** in both
paths (`training/dataset_builder_modules/preprocess_text.py:18-24`;
`core/generation/handler/conditioning_text.py:131`).

Two consequences the brief asked about directly:

- **Can the text encoder represent an emotional distinction?** Yes. It is a full
  general-purpose Qwen3 embedding model, not a genre tag vocabulary. Prompt-side
  emotion *can* reach the DiT. What it cannot do is carry an unbounded caption: the
  budget is 256 tokens for instruction + tag + caption + metas.
- **Metas come last, so they truncate first.** Our own prompts run **56 / 158 / 591
  estimated tokens** (min/median/max; estimate = characters ÷ 3.6, so treat as
  approximate, not measured). **38 of 184 (~21%) exceed 256.** For those, the model
  never saw the metadata block at all — but see §3.5, because for us it would not have
  mattered.

### 3.3 The lyric channel — and a 512 vs 2048 train/inference mismatch

Lyrics are formatted as `# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>`
(`core/generation/handler/prompt_utils.py:26-28`), embedded with
`text_encoder.embed_tokens(...)` — the **embedding table only**, no transformer pass
(`training/dataset_builder_modules/preprocess_lyrics.py:24`) — then processed by the
8-layer `AceStepLyricEncoder` inside the DiT.

- Training truncates at **512** tokens (`preprocess_lyrics.py:9`).
- Inference truncates at **2048** (`conditioning_text.py:142`).

Our lyric blocks estimate at 11 / 292 / 912 tokens (min/median/max, chars ÷ 3.6,
approximate). **40 of 184 (~22%) exceed the 512 training cap; none exceed 2048.** So on
long songs the adapter was trained on the first ~60% of the lyric and is asked at
inference to place all of it. Structure and timing are exactly what breaks under that
mismatch. Inferred, but the mechanism is verified.

### 3.4 What LoRA can and cannot reach — a hard architectural ceiling

`training_v2/preprocess.py` is a two-pass pipeline. Pass 1 encodes text and lyrics
(lines 270-276). **Pass 2 runs the condition encoder once and freezes the result**
(`training_v2/preprocess.py:429-441` calling
`training/dataset_builder_modules/preprocess_encoder.py:33-41`, inside
`torch.no_grad()`), writing `encoder_hidden_states` into the `.pt` files. Training then
only ever forwards through `self.model.decoder`
(`training_v2/fixed_lora_module.py:342-350`).

Therefore, in ACE-Step's supported pipeline:

- **Trainable:** the DiT decoder's 24 layers only.
- **Never trainable:** the Qwen3 text encoder, `text_projector`, `AceStepLyricEncoder`,
  `AceStepTimbreEncoder`, `condition_embedder`, the VAE, the 5Hz LM.

Two more things pass 2 bakes in, both of which scope our adapter tightly:

- `preprocess_encoder.py:17` passes **zeros** as the reference-audio tensor. The timbre
  channel is dark for every training sample.
- `preprocess_context.py:11-34` builds `context_latents` from **silence** plus an
  all-ones chunk mask.

So **the LoRA we trained is strictly a text2music adapter.** It has never seen a cover
source, a repaint mask, a reference timbre, or a planner sketch in its context latents.
Expect it to underperform under cover and under thinking-on. Verified from the source;
not measured in audio.

### 3.5 The planner: what it can and cannot plan

`checkpoints/acestep-5Hz-lm-1.7B/config.json`: `Qwen3Model`, 28 layers, hidden 2048,
16 heads / 8 KV, `vocab_size: 217204` (stock Qwen3 vocab plus a 64000-entry audio
codebook — `constrained_logits_processor.py:47` gives `MAX_AUDIO_CODE = 63999`).
3.71 GB on disk in bf16.

Its output is forced through an FSM (`constrained_logits_processor.py:55-79`) that
allows exactly:

```
<think>
bpm: [30-300]
caption: [free text]
duration: [10-600]
keyscale: [A-G][#/b]? [major|minor]
language: [en|zh|ja|...]
timesignature: [2|3|4|6]
</think>
<audio codes ...>
```

**No emotion field. No section list. No structure timeline.** The `caption` field is the
only place emotion can live, and it is free text.

Then it generates audio codes, which are dequantised to a 5 Hz latent sketch, decoded
to 25 Hz and **substituted for the source latents**
(`modeling_acestep_v15_turbo.py:1690-1696`). Note the gate is `is_covers`, but
`core/generation/handler/task_utils.py:110-113` sets `is_cover_task = True` whenever
code hints are present — so **the planner's sketch reaches text2music too, through the
cover pathway, whenever thinking emits codes.** This is the whole-song structural
prior, and it is the highest-leverage timing lever in the stack.

ACE-Step ships **no training code for it**: `training_v2/model_discovery.py:62` skips
any checkpoint whose directory name starts with `acestep-5Hz`, and the only
`AutoModelForCausalLM` in the tree is in `llm_inference.py` (inference).

### 3.6 The receptive-field limit

`config.json` alternates `sliding_attention` / `full_attention` across the 24 layers,
`sliding_window: 128`, `use_sliding_window: true`. Applied at
`modeling_acestep_v15_turbo.py:289-290` and passed to the kernel at line 368 — and
**only for self-attention**: line 368 reads
`sliding_window=self.sliding_window if not self.is_cross_attention else None`. The mask
is bidirectional (`|i-j| <= window`, line 108).

Latent rate is 48000/1920 = 25 Hz (`conditioning_masks.py:52`), and `patch_size: 2`
halves it to 12.5 Hz inside the DiT. **±128 tokens ≈ ±10.2 seconds.** Half the
self-attention stack is musically near-sighted. Cross-attention is always full, so
text and lyrics reach everywhere; it is *audio-to-audio* long-range structure that is
constrained. This is why the planner matters more than the DiT for structure.

---

## 4. What is actually wrong with the LoRA we trained

### 4.1 Verified: cross-attention was wrapped. The brief's suspicion was wrong.

Read directly from the safetensors header of
`train/lora_out/final/adapter_model.safetensors` (header parse only, no torch,
no model load) — **384 tensors**:

| count | key pattern | A shape | B shape |
|---|---|---|---|
| 24 | `base_model.model.layers.N.self_attn.q_proj.lora_{A,B}` | [16, 2048] | [2048, 16] |
| 24 | `...self_attn.k_proj...` | [16, 2048] | [1024, 16] |
| 24 | `...self_attn.v_proj...` | [16, 2048] | [1024, 16] |
| 24 | `...self_attn.o_proj...` | [16, 2048] | [2048, 16] |
| 24 | `...cross_attn.q_proj...` | [16, 2048] | [2048, 16] |
| 24 | `...cross_attn.k_proj...` | [16, 2048] | [1024, 16] |
| 24 | `...cross_attn.v_proj...` | [16, 2048] | [1024, 16] |
| 24 | `...cross_attn.o_proj...` | [16, 2048] | [2048, 16] |

All 24 DiT layers, both attention types, all four projections. `--attention-type both`
was in the launch command (`train/logs/train-20260820-170000.log`) and
`training_v2/cli/validation.py:60-98` leaves the suffix list unprefixed in that mode, so
PEFT matched both. The `k_proj`/`v_proj` B-shape of 1024 is GQA (8 KV heads × 128).

**Not present:** any `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, and anything
outside `model.layers` — no lyric encoder, no timbre encoder, no projectors. Consistent
with §3.4: adapters attach to `model.decoder` only
(`core/generation/handler/lora/lifecycle.py:270,278`).

The arithmetic checks out exactly, which confirms the reading. Per layer, rank 16,
hidden 2048, KV 1024: q 65,536 + k 49,152 + v 49,152 + o 65,536 = 229,376 per attention
block; × 2 blocks × 24 layers = **11,010,048** — the precise figure the training log
reports. Adding the MLP (`Qwen3MLP`, `intermediate_size: 6144`, instantiated at
`modeling_acestep_v15_turbo.py:472`) would add 393,216/layer × 24 = **9,437,184**, an
86% increase, and `--target-modules` accepts arbitrary suffixes
(`training_v2/cli/args.py:289`), so `gate_proj up_proj down_proj` need only be appended.

### 4.1b The fullest fine-tune ACE-Step supports

`training/lora_injection.py:204-213` builds `LoraConfig(r, alpha, dropout,
target_modules, bias, task_type=FEATURE_EXTRACTION)` and calls
`get_peft_model(decoder, ...)` at line 213 — on `model.decoder`, after
`_unwrap_decoder` (line 186). There is **no `modules_to_save`**, so nothing outside a
LoRA-wrapped module is ever trainable.

**The answer to "what is the fullest fine-tune it supports" is: a LoRA or LoKr on the
DiT decoder's linear layers, and nothing else — ever.** Not the text encoder, not the
lyric or timbre encoders, not the VAE, not the planner. Anything beyond that is a fork,
not a flag. Options 13 (planner SFT) and 20 (unfreezing the condition encoder) are both
forks; that is their real cost.

### 4.2 The dataset defect that matters most

```
bpm            → 'N/A'  ×184
keyscale       → 'N/A'  ×184
timesignature  → 'N/A'  ×184
```

Hard-coded at `train/build_dataset.py:290`. Not missing, not null — the literal string
`"N/A"`, which is truthy, so nothing downstream flags it. Every `# Metas` block in
training read:

```
- bpm: N/A
- timesignature: N/A
- keyscale: N/A
- duration: 196.76 seconds
```

`duration` is real. The other three are not. The channel is live —
`training_v2/preprocess_prompt.py:63-73` formats them, `preprocess.py:271` encodes them
— so we spent 6.8 GPU-hours teaching the model that `xenowiz` co-occurs with
**unspecified tempo, key and meter**. When the planner (or the UI) then supplies a real
bpm at inference, that combination is off-distribution for the adapter.

This also means the §3.2 truncation finding is, for *this* dataset, harmless in effect —
truncating `bpm: N/A` costs nothing. It becomes load-bearing the moment option 1 is
done, which is why options 1 and 2 should ship together.

### 4.3 The run was ~3× longer than it needed to be

From `train/logs/train-20260820-170000.log` and the checkpoint names:

| epoch | loss | wall clock |
|---|---|---|
| 10 | 0.8409 / 0.8248 | 17:40 |
| 20 | 0.8017 | 18:21 |
| 30 | 0.7760 | 19:02 |
| 40 | 0.7768 | 19:43 |
| 50 | 0.7660 | 20:23 |
| 60 | 0.7703 | 21:04 |
| 70 | 0.7678 | 21:45 |
| 100 | 0.7588 | 23:47 |

Flat from epoch 30. Epoch 1 took 246.8 s. 11,010,048 trainable params, 23 optimizer
steps/epoch. **Epochs 30→100 cost ~4.8 hours for 0.017 of loss.**

Also in that log: **LyCORIS not installed** (LoKr unavailable) and **Lightning /
Lightning Fabric not installed** (basic training loop only, no Fabric mixed-precision
plumbing).

### 4.4 Other dataset facts worth knowing

- 184 samples, **10.43 raw hours**; `--max-duration 240` truncates rather than skips, so
  **9.88 hours actually train** and 25 tracks over 4 minutes are cut. Those 25 are the
  long, most structurally interesting songs, **and the adapter never saw their endings.**
- 94 of 184 are instrumental.
- All 184 at 48 kHz.
- `_model` is `chirp-crow` ×108, `chirp-fenix` ×29, `chirp-v4` ×21, `chirp-auk` ×14,
  `chirp-bluejay` ×11, `chirp-chirp` ×1 — the source library is Suno output, as
  `train/README.md` already notes. The captions are therefore **Suno style prompts**,
  which is a different distribution from ACE-Step's own caption format. Option 8
  addresses this.
- Play counts: min 1, median 18, mean 34.3, max 397, over 184 tracks.
- `custom_tag` is confirmed as a trigger-token mechanism, not a metadata field:
  `training_v2/preprocess_prompt.py:55-61` prepends/appends/replaces it into the caption
  before encoding (`training/dataset_builder_modules/models.py:56-64` upstream). Ours is
  `"xenowiz"`, `tag_position: "prepend"`, so every training caption began
  `"xenowiz, ..."`. To fire the adapter, the prompt must contain that word.

---

## 5. Preference data: what 215 play counts can and cannot support

Local facts first:

- The training loss is `F.mse_loss(decoder_outputs[0], flow)`
  (`training_v2/fixed_lora_module.py:354`) — unweighted, no per-sample term, no hook.
  **The only weighting mechanism available without patching ACE-Step is repeating
  `.pt` files in `train/preprocessed/`.**
- `upvote_count` and `is_liked` are parsed and discarded (§2.7). Recovering them costs
  one line in `build_dataset.py` and gives two more signals.
- ACE-Step already ships a reward function (§2.2) that needs **zero** preference data.

Scoping, with the literature check that follows each:

| Method | Viable at n≈215? | Why |
|---|---|---|
| **Curation / thresholding** (drop the bottom decile) | **Yes — and this is the strongest form of weighting** | Emu got an **82.9% win rate over its base model from 2,000 curated images** on top of a 1.1B-pair pre-train (`https://arxiv.org/pdf/2309.15807`). **Filtering beats weighting.** 4 of our tracks have a play count of 1 |
| **Duplication weighting** by play count | **Yes, with a cap** — but expect less than curation | Log-scaled, capped ~3×. Linear weighting on a 1→397 range would make the top 20 tracks most of the run. Reward-weighted regression is real but published results show *"reduced image quality, such as over-saturation"* and DPOK beats it |
| **Score-aware timestep routing** | **Yes, and it is published on ACE-Step** | Rather than discarding low-scoring segments, route them to **high-noise timesteps**: `α(S) = 1 + λ(1−S)`, λ=1.0, filter below 0.20. Ablated at 2,000 samples on a 450M model **using ACE-Step 1.5 as the frozen codec**; CLAP +0.018, FAD 0.2856→0.2767 (`https://arxiv.org/html/2606.07387`). Needs a patch to `sample_timesteps`, so it is a fork |
| **Quality-tag conditioning** (a `"favourite"` token on high-play tracks) | **Weaker than I assumed** | Standard LoRA practice is the *opposite* of intuition — quality tags belong in the **generation prompt, not the training captions**, where `masterpiece` does "more harm than good". SDXL's aesthetic-score conditioning is "subtle and often swamped"; Pony's `score_9` works only because it was baked into training captions. The one working audio precedent is HunyuanVideo-Foley (`https://arxiv.org/pdf/2508.16930`), which tags >16 kHz clips `"high-quality"` in training and appends it at inference. **And ACE-Step 1.5 itself uses reward-score *filtering*, not quality tags** — which points back to row 1 |
| **Best-of-N reranking** | **Yes — do this first** | Zero training data, and measured on our exact checkpoint. See §5.1 |
| **DPO on our own generations** | **Only after collecting new ratings, or synthetically** | DPO needs preference *pairs over generations*. Play counts on the catalogue are not pairs, and the catalogue is not model output |
| **A learned reward model** | **Yes — I was wrong about this** | See §5.1. The published floor is **150 clips**, not thousands |

**Where the literature agrees with the pessimistic reading: training the *generator* on
215 preferences.** The floors:

| System | Preference data |
|---|---|
| Diffusion-DPO (SDXL) | **851,293 pairs** / 58,960 prompts |
| MusicRL | ~300,000 pairs (proprietary) |
| LeVo | 60,000 pairs (human component only 4,000 crowd-ranked) |
| SegTune | ~20,000 per round |
| **Tango 2 / Audio-Alpaca** | **15,025 triplets** — and *synthetically constructed* |
| **DRAGON (music)** | **1,676 human-rated pieces** → 60.95% win rate |
| AWS Nova DPO guidance | "minimum of 1,000 preference pairs for effective training" |

**The lowest number that produced a real improvement in a music generator is DRAGON's
1,676** (`https://arxiv.org/abs/2504.15217`) — still 8× our 215. The 2025 survey
(`https://arxiv.org/html/2511.15038v1`) names few-shot preference learning an open,
unsolved problem with no concrete methods for small datasets, and DPO is documented as
"prone to overfitting on limited preference datasets".

Worse than the count is the **shape**: "A outplayed B" compares different songs under
different prompts, which is not what DPO consumes. Beware the "few-shot personalisation"
headlines too — PPD's "as few as four preference examples… 76% win rate" is *adaptation*
on top of a model meta-trained on a large multi-user corpus, not training from four
examples.

**But note *how* Tango 2 got its 15,000: synthetically, with no human labelling at all.**
Multiple inferences per prompt, perturbed prompts, then CLAP-score filtering to pick
winner and loser. **DiffRhythm+ does the same thing in music**, building its DPO pairs
fully automatically by scoring its own generations with SongEval + Audiobox-Aesthetics —
on-distribution, zero human labels (`https://arxiv.org/html/2507.12890v1`). We have both
ingredients: a generator, and an automatic scorer (§2.2, plus option 7b).

Generating 15,000 pairs at ~15 s per render is roughly **60 GPU-hours** on this card — a
long unattended weekend, not an impossibility. And Tango 2's entire DPO training run was
**7 A100-hours ≈ $8–13** (§6.2c). **That is the credible preference path from here, and
it does not use the play counts at all.** Filed as a real option, not a recommended one.

A cheaper variant that *does* fit our data shape: the Tango 2 construction with
ground-truth audio as winner and our model's generation for the same caption as loser.
184 captions × 5–10 losers ≈ **900–1,800 pairs**, right at the AWS floor, with high beta
to stay near the reference model. Unpublished at this scale — flagged as untested.
Alternatively **DRAGON's exemplar mode, which needs zero human labels and in which our
215 tracks simply *are* the target distribution** — the only method found whose input is
literally what we own.

One more thing the literature says that cuts against all of this: ACE-Step's own
alignment stage used **intrinsic geometric rewards with no external reward model and no
DPO** (§7.0). Bolting an external preference signal onto a model aligned that way is
working against the grain, and the shipped PMI scorer is the in-grain alternative. The
tech report puts its Attention Alignment Score at ">95% correlation with human
judgments" (`https://arxiv.org/html/2602.00744v3`) — vendor claim, unverified.

### 5.1 The correction: a *personal reward model* at n=215 is not only viable, it is published

I initially reasoned that 215 examples could not support a learned reward model. **The
literature says otherwise and gives an exact floor.**

**MuQ-Eval** (`https://arxiv.org/html/2603.22677v1`, weights at
`https://github.com/dgtql/MuQ-Eval`) is a music quality evaluator trained on 2,748 clips
/ 13,740 expert ratings; system-level SRCC 0.957, per-clip 0.838 (Audiobox Aesthetics
manages r = 0.200 on the same task). Its Section V-H ablates training-set size for a
**LoRA-adapted** variant:

| N_train | SRCC (base) | SRCC (LoRA-adapted) |
|---|---|---|
| 100 | 0.635 | 0.757 |
| **150** | 0.685 | **0.761** |
| 250 | 0.731 | 0.788 |
| 500 | 0.781 | 0.808 |

The abstract states it plainly: *"LoRA-adapted models trained on as few as 150 clips
already achieve usable correlation, enabling personalized quality evaluators from
individual listener annotations."* **We have 215.** Interpolating gives SRCC ≈ 0.78 —
that interpolation is mine, not the paper's.

**Two caveats that decide the recipe, both raised by the research and both real:**

1. Those 150 clips carried **expert MOS ratings**. Play counts are noisier — recency,
   mood and track length all confound them. Our labels are weaker than the paper's.
2. MuQ-Eval was trained on **generated** music. A scorer trained only on our human-made
   library never sees generation artifacts and therefore **cannot penalise them**.

**So the recipe is two scorers, not one:** an off-the-shelf evaluator (MuQ-Eval base, or
TuneJury) for the artifact floor, plus a personal LoRA on our 215 play counts for taste.
Rank by the product, or gate on the first and rank by the second.

**And best-of-N was measured on our exact model.** TuneJury
(`https://arxiv.org/html/2606.17006v1`) evaluated N ∈ {1,2,4,8,16,32} on four frozen
backbones **including ACE-Step v1.5 Turbo**. Reward rose monotonically with N on every
backbone: **+0.178 to +0.291 per doubling at N=4→8**, decaying to +0.060 to +0.144 at
N=16→32. This is not a plausible mechanism any more; it is a measured one, on this
checkpoint. **Option 7 is the best-evidenced item in this report.**

Its warning is worth carrying too: the same paper's aggressive variant (expert
iteration, 900 candidates/prompt, top decile) gave a bigger reward lift but **structural
distributional drift** — MAD rose +0.293 to +0.669, and the authors call the drift
"structural under instance-level reward optimization." Best-of-8 is safe; best-of-900
turns into a different model.

**Also revised: the minimum preference-set size for a music generator.** DRAGON
(`https://arxiv.org/abs/2504.15217`) reached a **60.95% human win rate from 1,676
human-rated pieces** — well below Tango 2's 15,000, though still 8× our 215. DRAGON also
works with **zero** human labels using exemplar distributions, where our 215 tracks *are*
the exemplar set. That is a genuinely open door, and it is the one preference method
whose data shape matches what we own.

---

## 6. Build our own — the numbers

Wall-clock inputs measured here, extrapolations flagged.

### 6.1 What our box does today

| Quantity | Value | Source |
|---|---|---|
| Epoch time, rank-16 LoRA, 184 samples | 246.8 s | `train/logs/train-20260820-170000.log` |
| Full 100-epoch run | 408 min = **6.8 h** | `train/logs/last_run.json` |
| Trainable params | 11,010,048 | training log |
| Preprocess VRAM floor | ~3.8 GB | `train/README.md`, derived from `training_v2/gpu_utils.py:160-164` |
| Training VRAM (upper bound) | ~6.5 GB | same |
| DiT decoder weights | 4096 MiB | `training_v2/gpu_utils.py:164` |
| Forward+backward per sample (full FT) | 1200 MiB | `training_v2/gpu_utils.py:160` |
| Effective dataset | **9.88 audio-hours**, 184 clips | computed from `dataset.json` |

### 6.1b Cloud GPU prices, observed 2026-08-22

USD per GPU-hour. Needed for the numbers below; the owner asked for figures, not a
recommendation to rent.

| GPU | RunPod on-demand / spot | Lambda OD | Cheapest primary found | Market median¹ |
|---|---|---|---|---|
| RTX 4090 | 0.74 / **0.34** | — | RunPod **$0.34** | $0.44 |
| RTX 5090 | 0.99 / 0.69 | — | RunPod **$0.69** | $0.46 |
| A100 40 GB | — | 1.99 | Verda spot **$0.645** | — |
| A100 80 GB | 1.39 / 1.19 | 2.79 | Thunder **$1.09** OD | $1.79 |
| H100 PCIe | 2.89 / **1.99** | 3.29 | RunPod/VoltagePark **$1.99** | — |
| H100 SXM | 3.29 / 2.69 | 3.99–4.29 | TensorDock **$2.25** | **$3.38** |
| H200 | 4.59 / 3.59 | — | Atlas **$3.50** | $4.29 |
| B200 | 6.79 / 5.98 | 6.69–6.99 | Verda spot $3.06 | $6.49 |
| GB200 | — | — | **no primary source found** | $18.77 |

¹ Secondary aggregate, `https://getdeploying.com/gpus` (3,280 prices / 74 providers).
Primary sources: `https://www.runpod.io/pricing`, `https://lambda.ai/pricing`,
`https://verda.com/pricing`, `https://www.thundercompute.com/pricing`,
`https://www.voltagepark.com/pricing`, `https://www.atlascloud.ai/pricing/gpu`.

**Market note: the H100 median is UP 12% year-on-year ($3.02 → $3.38) and +1.5% in the
last four weeks. Prices are rising, not falling.** Traps flagged by the research:
Together's "reserved" tiers are 7-day-to-180-day commitments, not spot; Lambda's
1-Click Clusters need 2 weeks to 1 year; DigitalOcean/Paperspace's advertised $2.24 H100
is a **3-year** price against $5.95 real on-demand.

### 6.2 The options, costed

**(a) More/better LoRA — recommended.**
Data: 0 new hours. Compute: 2–3.4 h per run on the existing card, ~0 GPU-hours rented.
Dollars: **$0.** This is where options 1, 2, 9, 10, 11 live. Every one of them is a data
or config change on hardware we own.

**(b) SFT the 5Hz planner — recommended as the one component worth building.**
This is the answer to "build our own, in part". Rationale, all locally verified:
it is a stock `Qwen3Model` 1.7B; the encoder that turns our audio into its training
targets already exists (`convert_src_audio_to_codes`); the reverse-direction labeller
exists (`understand_audio_from_codes`); and it is the only component that plans
whole-song structure.

- Data to build: 184 pairs of `(caption + metas + lyrics) → 5Hz codes`. At 5 Hz, 240 s
  of audio ≈ 1200 code tokens; plus prompt, call it ~2000 tokens/sample ⇒ **~370k
  training tokens**. Tiny by LLM standards; this is a style-transfer SFT, not a
  knowledge injection.
- Encoding pass: est. 10–20 s/track ⇒ **~1 hour** on the 3080. Estimate.
- Training: QLoRA (4-bit base + rank-16/32 adapter) on 1.7B at seq 2048, batch 1.
  Est. **8–10 GB, ~5–10 min/epoch, under 2 hours for 10 epochs.** Estimate — no
  measurement, and this is the number most likely to be wrong.
- Dollars: **$0** if it fits; est. $2–6 on a rented 24 GB card if it does not.
- **The real cost is engineering, not compute.** There is no trainer for this in the
  repo. We would write a HF `Trainer`/TRL loop against the checkpoint, get the prompt
  format right (`SFT_GEN_PROMPT`-style with the `<think>` block from
  `constrained_logits_processor.py`), and keep the constrained decoder working
  afterwards. Estimate: a few focused days. That is the honest risk.

**(c) Full fine-tune of the 2B DiT — cheaper than I assumed, and worth one experiment.**

Memory arithmetic for 2B params, mixed precision, AdamW: bf16 weights 4 GB + bf16 grads
4 GB + fp32 optimizer moments 16 GB + fp32 master weights 8 GB ≈ **32 GB**. With 8-bit
Adam, moments drop to ~4 GB ⇒ **~20 GB**, plus activations (1200 MiB/sample per
`gpu_utils.py:160`, less with gradient checkpointing). So: **does not fit 10 GB; ~24 GB
is the floor; 40–80 GB is comfortable.** This arithmetic is standard, not read from the
repo — inference, not verification.

**I initially wrote this off on cost. That was wrong, and the counter-example is close
to our exact situation.** MusicGen Songstarter v0.2 **full fine-tuned a 3.3B model on
1,700–1,800 samples ≈ 7–8 hours of audio**, on 8× A100-40GB for ~6 hours = **48
GPU-hours**. At today's prices (§6.1b) that is **$62–96**. The author reports paying
~$180 at 2024 rates. **We have 9.88 hours. This is a $100 experiment, not a $50,000
one.**

Comparable published fine-tune costs, all converted at §6.1b prices:

| Work | Base | Data | Hardware | GPU-h | USD |
|---|---|---|---|---|---|
| MusicGen Songstarter v0.2 | 3.3B | **7–8 h** | 8× A100-40, ~6 h | 48 | **$62–96** |
| Tango 2 (the whole DPO run) | Tango | 15,025 pairs | 2× A100, 3.5 h | 7 | **$8–13** |
| SAO-Instruct | Stable Audio Open | 150k samples | 2× A6000, 80 h | 160 | $56–174 |
| Low-resource adapters | MusicGen / Mustango | 157–209 h | 1–2× A6000 | 40–45 | $14–49 |
| Full TuneJury preference pipeline | 120M | SFT + expert-iter + ranker | 1× A5000 | 40 | **$14–44** |

**So the honest objection is not money — it is forgetting.** 9.88 hours against
ACE-Step's pre-training corpus is a rounding error, and a full fine-tune at that ratio
risks catastrophic forgetting for the price of lunch. It also produces a ~4.8 GB
checkpoint we must store, version and re-merge on every upstream update, where a LoRA is
44 MB and stacks.

**Revised recommendation: do it once, on a rented 8× A100 day, after the data is fixed —
as a benchmark against the best LoRA, not as a replacement for it.** ~$100 to learn
whether the LoRA ceiling is the adapter or the data is cheap information. Do not do it
before options 1, 2 and 9, or you will be measuring the `"N/A"` bug at 8× the cost.

**(c2) Continued pre-training — no, and the reason is data, not money.**

At roughly $60–100 per 100 data-hours (inferred below), money is not the barrier. Our
9.88 hours against ACE-Step v1's disclosed **100,000-hour** pre-training corpus is
**0.01%**. We would induce catastrophic forgetting for the price of a lunch. Skip.

The only cost estimate available is inferred, and I am labelling it as such because no
paper publishes it: ACE-Step v1's fine-tune phase was ~240k of ~700k total steps ⇒
≈10,900 A100-hours for 20,000 data-hours ⇒ **~0.55 A100-h per data-hour at 3.5B**.
Linear-scaled that gives 1,000 h ≈ 550 A100-h ≈ **$600–985**, and 100 h ≈ 55 A100-h ≈
**$60–98**. Small sets need more epochs, so treat those as lower bounds — realistically
500–2,000 A100-h (**$550–3,600**) for 1,000 hours. **The paper does not split pre-train
from fine-tune; this is a step-share inference, not a published figure.**

**(d) From scratch — no, and here is the arithmetic that closes it.**

ACE-Step **v1** is the only model in this class that discloses enough to price:

| Component | Data | Hardware | GPU-hours | USD at §6.1b |
|---|---|---|---|---|
| Diffusion model (3.5B) | 100,000 h pre-train / 20,000 h FT | 120× A100, 264 h | **31,680** | $34.5k–56.7k |
| Music DCAE | 100,000 h | 120× A100, 5 days | **14,400** | $15.7k–25.8k |
| **Total** | 100,000 h | 120× A100 | **≈46,080** | **≈$50k–82k** |

GPU counts and durations are disclosed; the GPU-hour products and the USD are
arithmetic. For comparison, Stable Audio Open cost ≈36,224 A100-hours (**$39.5k–64.8k**)
on 7,300 hours of data of which only 970 were music, and Stable Audio 2 ≈104,000
A100-hours (**$113k–186k**).

Corpus sizes across the field, for scale:

| Model | Training data |
|---|---|
| Qwen-Music | **>5,000,000 hours** |
| YuE | 650,000 hours mined |
| LeVo 2 | ~500,000 hours |
| MusicLM | 280,000 hours |
| ACE-Step v1, SongBloom | 100,000 hours |
| DiffRhythm 2 | 70,000 hours |
| JAM | 54,000 hours |
| SegTune | 27,000 h pre-train + 4,000 h fine-tune |
| MusicGen | 20,000 hours |
| Muse (smallest complete open stack) | ~7,771 hours |
| SongGen — **first credible vocals+lyrics** | **~2,000 hours** |
| FluxAudio-S / Prefix SiMBA — smallest *coherent instrumental* | **457–464 hours** |
| **Us** | **184 clips / 9.88 hours** |

**The two floors that matter:**

- **~450–465 hours** is the smallest published corpus that produced coherent
  *instrumental* text-to-music — and notably, on **one consumer GPU**: Prefix SiMBA
  (281M params, 1× RTX 3090, `https://arxiv.org/html/2601.14786v1`) and FluxAudio-S
  (120M, 1× A6000, ~2 days, ≈$18, `https://arxiv.org/html/2605.21538v2`, whose authors
  write that "even the partial 464-hour subset enabled competitive submissions").
- **~2,000 hours** is the smallest published corpus that produced credible **vocals with
  lyrics** (SongGen, 1.3B, 16× A100).

We have **2.2%** of the instrumental floor and **0.5%** of the vocal floor. **No paper
publishes a hard minimum**, but nothing below 450 hours exists, and we are 45× under
even that. **From scratch is closed**, and it would still be closed if the GPUs were
free.

### 6.3 The one-line version

> Spend the next month on data and on wiring what is already installed. Then spend $100
> on a full fine-tune to find out whether the ceiling is the adapter or the data. Spend
> the month after that on the planner, if timing is still the complaint. Do not spend a
> penny on any of it until `bpm` stops saying `N/A`.

And the single sentence about the play counts:

> 215 genuine personal preference labels cannot move a 2B model's gradients, but they
> can train a **personal reranker** — the published floor is 150 clips. Spend them on
> choosing between takes, not on training.

### 6.4 A concrete order to do it in

Nothing below needs a new GPU, a rented card, or a dollar. Ordered by
(value ÷ effort), with the checks that decide whether to continue.

**Day 1 — three experiments, no training, ~1 hour total.**
1. Option 0: one generation with `use_constrained_decoding=False`, thinking on, and read
   the planner's raw `<think>` block. **If a structure/section plan appears, stop and
   re-plan around it** — that changes the whole timing answer.
2. Option 6: same prompt, `thinking=True, use_cot_caption=True`. Compare against
   thinking-off at a fixed seed (§12 of `REQUIREMENTS.md` made the seed real, so this
   test is now meaningful).
3. Option 5/3 spike: one `flow_edit_morph` call and one `repaint` call from a script, to
   confirm they run on our checkpoints before any UI work.

**Week 1 — fix the data. This is the highest-value block on the page.**
4. Run **beat-this** over the 184 tracks → beats and downbeats, from which `bpm` and
   `timesignature` fall out directly; run **SongFormer** for section boundaries and
   labels (§7.2). **Neither does key** — get `keyscale` from Essentia/librosa key
   detection, or from `understand_audio_from_codes` (§2.5), and cross-check the two on a
   handful of tracks before trusting either. Remember `VALID_KEYSCALES`
   (`constants.py:35-41`) is a 70-value closed set of the form `"F# minor"`, and
   `VALID_TIME_SIGNATURES` is `[2, 3, 4, 6]` (`constants.py:66`) — anything outside those
   is as good as `N/A`.
5. Add `upvote_count` and `is_liked` back into `dataset.json` (§2.7) — one line.
6. Add the mood prefix (1b) and per-section emotion tags (1c), using `music2emo`
   quadrants (§7.1) as a first pass and hand-correcting. **Coarse bins only** — the
   literature says fine-grained valence is not recoverable.
7. Trim captions and lyrics under the caps (option 2).
8. Retrain: `--preset vram_12gb` (rank 32), **50 epochs not 100**,
   `--sample-every-n-epochs 10`, `--target-modules q_proj k_proj v_proj o_proj
   gate_proj up_proj down_proj`. ~2–3 hours. **Check: does the mood prefix move the
   output at a fixed seed?** If not, options 1b/1c are dead and the emotion answer is
   reference-audio and flow-edit, not prompting.

**Week 2 — wire the dark features and build the reranker.**
9. Expose `repaint` (option 3) and `reference_audio` (§2.3) in the worker and the form.
10. Wire the built-in scorer into the worker and rank takes by it (option 7). **This is
    the best-evidenced item in the report** — measured on ACE-Step v1.5 Turbo
    specifically, +0.178 to +0.291 reward per doubling of N from 4 to 8 (§5.1). Stop at
    N=8–16.
11. LoRA-adapt MuQ-Eval on the 215 play counts (option 7b) and rank by the product of
    the two scorers. **This is what the play counts are actually for.**
12. Add `base` as a selectable checkpoint so `complete` / `extract` / `lego` become
    reachable (option 4).

**Week 3+, budgeted, in this order:**
13. **~$100, one rented 8×A100 day:** full fine-tune the 2B DiT on the *fixed* dataset
    and A/B it against the best LoRA (option 21). This answers "is the ceiling the
    adapter or the data?" for the price of a takeaway, and it is the single most
    informative experiment available. Do **not** run it before step 8.
14. Only if timing is still the complaint: option 13, the planner SFT.
15. And if it still is after that, the honest answer is option 15 — **DiffRhythm 2 takes
    a timed plan as input and ACE-Step cannot.** That is a model choice, not a training
    problem, and no amount of LoRA fixes it.

---

## 7. The field, August 2026

### 7.0 What the local model card says about ACE-Step itself

From `acestep/checkpoints/acestep-v15-base/README.md` (shipped with the weights — this
is the **vendor's claim**, not something I verified):

- Tech report: **arXiv:2602.00744**; project page
  `https://ace-step.github.io/ace-step-v1.5.github.io/`; collection
  `https://huggingface.co/collections/ACE-Step/ace-step-15`.
- **Licence: MIT**, and the card explicitly states commercial use of generated music is
  permitted, with training data described as licensed + royalty-free + synthetic
  (MIDI-to-audio). Relevant if any of this ends up published.
- "The Language Model (LM) functions as an omni-capable planner: it transforms simple
  user queries into comprehensive song blueprints — scaling from short loops to
  10-minute compositions — while synthesizing metadata, lyrics, and captions via
  Chain-of-Thought to guide the Diffusion Transformer."
  **This is the vendor confirming §3.5: the planner is the structure engine.**
- "this alignment is achieved through intrinsic reinforcement learning relying solely on
  the model's internal mechanisms, thereby eliminating the biases inherent in external
  reward models or human preferences." Read alongside §2.2, this is why the PMI scorer
  ships in the repo — it is the model's own alignment signal, exposed as an API. It also
  implies that bolting an *external* reward model on top is working against the grain of
  how this model was aligned. Docs claim, not verified.
- The card also names "vocal-to-BGM conversion" among the editing capabilities, which
  does not appear in `TASK_TYPES` — likely the `extract` task under another name.
  Unconfirmed.

### 7.0b Two places the docs and this disk disagree — believe the disk

- The ACE-Step LoRA tutorial states **16 GB minimum VRAM, 20 GB+ recommended, ~17 GB
  typical** for LoRA training
  (`https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md`).
  **We trained successfully on a 10 GB 3080** — 100 epochs, 184 clips, 408 minutes,
  `vram_8gb` preset with `adamw8bit` + gradient checkpointing + encoder offload
  (`train/logs/train-20260820-170000.log`). The tutorial figure is for the default
  rank-64 config, not the small preset. **Do not let that number stop a run.**
  Independently corroborated: `https://github.com/Estylon/ace-lora-trainer` targets 1.5
  with a minimum of a 3060 8 GB and tiers **"RTX 3080/4070 | 10–12 GB | batch 1 | rank
  32"** using exactly the levers our preset uses — gradient checkpointing, encoder
  offload, 8-bit AdamW, two-pass preprocessing. **Rank 32 on this card is a third party's
  documented tier, not a gamble** (option 9).
- The tech report describes the LM planner as emitting a **`structure`** field in its
  chain-of-thought YAML. The shipped constrained decoder has **no such state**:
  `constrained_logits_processor.py:55-79` enumerates exactly `bpm, caption, duration,
  keyscale, language, timesignature` and nothing else. Two readings, both actionable:
  either structure lives inside the free-text `caption` field, or the FSM is a narrower
  schema than the model was trained to produce. **`use_constrained_decoding` is a
  parameter (`inference.py:207`, default `True`).** Turning it off is a five-minute
  experiment that would settle it — if the planner emits section timings when unmuzzled,
  that is the timing feature we are looking for, already trained, currently suppressed
  by a logits mask. Untested; flagged as the highest-value cheap experiment on this page.
- The tutorial's epoch guidance is "~100 songs → 500 epochs; 10-20 songs → 800 epochs".
  At 184 clips we are in the 500-epoch band and ran 100. Against that, our own loss
  curve was flat from epoch 30 (§4.3). **Trust the loss curve, but note the discrepancy
  is large enough to be worth one audible A/B** — turn on `--sample-every-n-epochs`
  (§2.10) and listen, rather than reasoning about it.

### 7.1 Emotion conditioning — the honest state of the art

**The single most important external finding, and it is bad news for prompt engineering.**

*AImoclips* (`https://arxiv.org/abs/2509.00813`) is the definitive measurement: 991
generated clips across 6 systems (AudioLDM 2, MusicGen, Mustango, Stable Audio Open,
Suno v4.5, Udio v1.5), 12 emotion words, **111 participants, 6,162 valence/arousal
ratings**. Findings:

- **All systems exhibit a centralizing tendency toward emotional neutrality** —
  generated music expresses valence/arousal *less clearly than the text asking for it*.
- Valence is conveyed best for **low-valence/high-arousal** targets (angry, anxious,
  scared). Arousal is conveyed best for happy/excited/energetic.
- **Valence differentiation collapses entirely when arousal is low** — "calm" and
  "gloomy" come out indistinguishable.

Corroborated independently by *LARA-Gen* (`https://arxiv.org/abs/2510.05875`,
Oct 2025), which measured text-only conditioning at **valence CCC 0.06, arousal CCC
0.23** — i.e. essentially no valence resolution from prompts at all.

**Conclusion, stated bluntly: arousal is controllable; valence is an open research
problem.** Chase energy and intensity first. Do not budget effort on "make it sadder"
as though it were an engineering task.

**What a dedicated emotion head costs.** LARA-Gen is the reference result for
conditioning-vector emotion control: a small MLP maps valence/arousal (1-9) to an
embedding concatenated with the text embeddings, plus a proxy network regressed onto a
frozen MERT-95M (loss = CE + 100·MSE). It moved arousal CCC 0.23 → **0.67** and valence
0.06 → **0.24**. It required **22,067 labelled 30-second clips ≈ 184 hours**. We have
**184 clips ≈ 10 hours** — three orders of magnitude short on labelled data for the
labelled axis. **This path is closed to us.**

**Emotion-labelled datasets that actually exist and are downloadable:**

| Dataset | Size | Labels | Audio downloadable | Licence |
|---|---|---|---|---|
| MTG-Jamendo mood/theme | **18,486 tracks**, 59 mood/theme tags | categorical folksonomy | **Yes** | Per-track CC; **non-commercial research only** (`https://mtg.github.io/mtg-jamendo-dataset/`) |
| MERGE (2024) | 3,554 audio / 30 s excerpts | Russell quadrants + continuous A/V | **Yes**, `https://zenodo.org/records/13939205` | CC BY-NC-SA 4.0 |
| DEAM / MediaEval | 1,802 songs | per-second continuous V/A | Yes | free/CC source audio |
| PMEmo | 794 chorus excerpts | static + dynamic V/A + EDA biosignals | Access request | copyright-constrained pop |
| EMOPIA | 1,087 clips | 4 quadrants | **MIDI only** — audio is YouTube IDs | CC BY-NC-SA 4.0 |
| Emotify | 400 one-minute excerpts | GEMS-9 *induced* emotion | Yes | — |
| AudioSet mood subset | ~13,713 clips, 7 classes | categorical | **No** — YouTube IDs, link rot | — |

The largest freely downloadable audio corpus with mood labels is MTG-Jamendo's, and its
labels are weak tags, not V/A — **and it is non-commercial research only, which matters
given ACE-Step's MIT/commercial framing (§7.0).** Continuous-V/A audio at scale
essentially does not exist publicly.

**Auto-labelling our own catalogue: this is the path that fits us.**

`music2emo` (`https://arxiv.org/abs/2502.03979`,
`https://github.com/AMAAI-Lab/Music2Emotion`, `https://huggingface.co/amaai-lab/music2emo`)
is the current tool. Its published numbers:

| Benchmark | Metric | Score |
|---|---|---|
| DEAM | R² valence / arousal | **0.5184 / 0.6228** |
| PMEmo | R² valence / arousal | **0.5473 / 0.7940** |
| EmoMusic | R² valence / arousal | **0.6512 / 0.7616** |
| MTG-Jamendo mood/theme | PR-AUC / ROC-AUC | **0.1543 / 0.7810** (near-SOTA) |

Read that honestly: **arousal auto-labelling is usable (R² 0.62–0.79); valence is
marginal (0.52–0.65); categorical mood tagging is weak in absolute terms (PR-AUC
0.154).** So auto-label into **coarse bins — four quadrants, or high/low arousal ×
positive/negative valence** — not fine-grained continuous targets we cannot verify.
That is exactly the granularity option 1b assumes.

**And training on auto-generated captions demonstrably works.** *Improving
Text-To-Audio Models with Synthetic Captions* (`https://arxiv.org/html/2406.15487v1`)
ran an audio-LLM over AudioSet to produce 696,079 synthetic captions over 331,421 clips,
CLAP-filtered, then pretrained on them. On MusicCaps: **FD 47.47 → 21.84, FAD 7.88 →
1.99, IS 1.85 → 2.21** — and the paper notes the **music-domain gains were by far the
largest**. SegTune (`https://arxiv.org/html/2606.02638v1`) labels its entire training
corpus this way as standard practice. **This is direct external support for option 8**
(recaptioning our catalogue), and a good reason to consider a stronger captioner than
ACE-Step's own `understand_audio_from_codes`: Music Flamingo
(`https://arxiv.org/pdf/2511.10289`) explicitly produces captions linking surface
attributes to "higher-level dimensions **including emotional trajectory**".

Caveat worth carrying: `https://arxiv.org/pdf/2511.05550` finds audio-LLMs near-human on
instrument identification but **chance-level on melody/rhythm relational tasks**. Trust
them for mood words and instrumentation; do not trust them for musical analysis.

**Training-free control on a frozen model — the one under-exploited idea.**

DITTO (`https://arxiv.org/abs/2401.12179`, ICML 2024 oral) optimises the **initial noise
latents** of a frozen text-to-music diffusion model against *any differentiable
feature-matching loss* — demonstrated for intensity, melody, **musical structure**,
inpainting and looping. DITTO-2 (`https://arxiv.org/abs/2405.20289`) is 10–20× faster.
MusicMagus / Loop Copilot (`https://arxiv.org/abs/2411.12641`) does zero-shot mood and
instrumentation editing by latent manipulation on a pretrained model.

The obvious composition — **DITTO with a frozen MERT/music2emo valence-arousal regressor
as the loss** — would give emotion control on a frozen ACE-Step with no training and no
labels. The subagent searched and **found no paper that has done this**; it is an
inference from two confirmed results, not a citation. It is also a fair amount of work
against a sampler we do not control.

**Preference optimisation — the floor is far above us.** See §5 for our side; the
external numbers:

| System | Method | Preference data required |
|---|---|---|
| MusicRL (`https://arxiv.org/abs/2402.04229`) | RLHF on MusicLM | **~300,000 pairwise human preferences** (proprietary) |
| **Tango 2** (`https://arxiv.org/abs/2404.09956`) | Diffusion-DPO | **~15,000 triplets** — and **semi-automatic, no human labelling**: multiple inferences per prompt, perturbed prompts, CLAP-score filtering |
| LeVo (`https://arxiv.org/html/2506.07520v1`) | multi-preference DPO | ~60,000 win/lose pairs for musicality |
| SegTune | 2 rounds DPO | ~20,000 pairs per round |
| Text2midi-InferAlign | inference-time alignment | **0** — +29.4% CLAP |

**Tango 2's ~15k is the smallest working preference set found, and it was
synthetically constructed rather than human-labelled.** The 2025 survey *Aligning
Generative Music AI with Human Preferences* (`https://arxiv.org/html/2511.15038v1`)
explicitly names **few-shot preference learning as an open, unsolved problem with no
concrete methods for small datasets**. This confirms §5: 215 play counts cannot support
DPO or a reward model. It does **not** close off Tango 2's trick — that recipe needs no
human labels at all, only many generations plus an automatic filter, and we already have
an automatic filter (§2.2). That is a real, if long, road.

### 7.2 Timing and structure — what to actually use

**Structure conditioning: ACE-Step's documented tag vocabulary is richer than our data
uses.** The upstream tutorial
(`https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md`) documents:

- Section tags: `[Intro] [Verse] [Verse 1] [Pre-Chorus] [Chorus] [Bridge] [Outro]
  [Build] [Drop] [Breakdown] [Instrumental] [Guitar Solo] [Piano Interlude] [Fade Out]
  [Silence]`
- **Modifier syntax: `[Chorus - anthemic]`** — officially documented, and exactly the
  form four of our samples already stumbled into (§2.8).
- Inline energy/emotion tags: `[high energy] [low energy] [building energy] [explosive]
  [melancholic] [euphoric] [dreamy] [aggressive]`
- Vocal modifiers: `[raspy vocal] [whispered] [falsetto] [powerful belting]
  [spoken word] [harmonies] [call and response] [ad-lib]`
- The `caption` field is documented as expecting "style, instruments, **emotion,
  atmosphere**, timbre, vocal gender, progression".

**This vindicates options 1b and 1c as officially-supported syntax rather than a hack.**
Our 184 samples use plain section tags and almost no modifiers. That is free control
sitting unused, and it is the cheapest emotion+timing lever on this page.

**LRC timestamps are output, not input** — confirmed externally, matching §2.2: ACE-Step
derives line timestamps *after* generation from DiT cross-attention. There is **no
timestamped-section conditioning input** in ACE-Step 1.5. Structure control = tags in the
lyrics field, plus whatever the LM plans.

**Tools to auto-label our catalogue's structure and timing** (this is the concrete
Phase-2 shopping list):

| Task | Tool | Why it, in 2026 |
|---|---|---|
| Section boundaries + functional labels | **SongFormer** (`https://github.com/ASLP-lab/SongFormer`, `https://arxiv.org/abs/2510.02797`) | New SOTA on strict boundary detection; **HR.5F 0.703, ACC 0.807** on SongFormBench-HarmonixSet, beating Gemini 2.5 Pro while staying cheap. HF one-click Space available |
| Beats + downbeats | **beat-this** (`https://github.com/CPJKU/beat_this`, ISMIR 2024) | Best published beat/downbeat F1, no DBN post-processing, `pip install beat-this`, actively maintained. **madmom is effectively superseded** |
| BPM + key + beats + sections in one call | allin1 (`https://github.com/mir-aidj/all-in-one`) | Convenient, but **brittle install** (NATTEN + madmom-from-git) and a known **20–40 ms MP3 decoder offset** against a 70 ms beat tolerance. Decode to WAV first. Community fix-forks exist |
| Sung-lyric word timings | **Demucs v4 → Parakeet-tdt-0.6b-v2 or Whisper-large-v3** | This is what JAM (`https://arxiv.org/html/2507.20880v1`) and SegTune actually do: separate the vocal stem, then run a speech ASR with word timestamps on it. Singing-specific aligners (SOFA, STARS) exist but **publish no head-to-head accuracy**, per the subagent — weakest-evidenced item in this report |

**Note what this buys directly:** SongFormer + beat-this on our 184 tracks produces
exactly the `bpm` / `timesignature` values that are currently `"N/A"` (option 1) **and**
the section boundaries needed for options 1c and 13, in one pass. That is one tool
purchase serving three options.

**Fine-tuning a small planner LM on structure is an established pattern.** The strongest
precedent: **SegTune fine-tunes Qwen3-4B on >100k LRC-format lyric files for the single
purpose of predicting section durations**, then broadcasts segment prompts to those
predicted boundaries (`https://arxiv.org/html/2606.02638v1`). Seed-Music
(`https://arxiv.org/html/2409.09214v2`) argues the same thing from first principles:
forcing one model to handle both long-range arrangement and dense token prediction is
why structure control fails, so insert a low-rate discrete sketch in between. That is
precisely ACE-Step's 5Hz LM. **Option 13 is not exotic — it is the industry pattern.**
What is missing is only that ACE-Step ships no tooling for it (§3.5).

**Timing conditioning is a solved, documented pattern if we ever fork.** Stable Audio's
`seconds_start` / `seconds_total` scalars are embedded and **concatenated along the
sequence dimension with text features into cross-attention**
(`https://github.com/Stability-AI/stable-audio-tools/blob/main/docs/conditioning.md`).
And `stable-audio-controlnet`
(`https://github.com/EmilianPostolache/stable-audio-controlnet`) is a working reference
implementation of a **DiT ControlNet on a frozen audio diffusion model** — the exact
shape option 20 would take. Music ControlNet (`https://arxiv.org/abs/2311.07069`) proves
melody/dynamics/**beat** control adaptors work on a frozen base with automatically
extracted training signals, and its controls can be **partially specified in time**.
None of this is on our path today; it is the map if the planner route fails.

### 7.3 The open-weights landscape, and where ACE-Step 1.5 sits

**There is no ACE-Step 2.0, and expressiveness is not on the roadmap.** Confirmed three
ways by the subagent: the org repo listing (`https://github.com/orgs/ace-step/repositories`)
shows no 2.x; `https://github.com/ace-step/awesome-ace-step` mentions only v1.0/v1.5;
no roadmap in the README. **The current release is ACE-Step 1.5 XL (4B DiT), 2026-04-02**
— four and a half months old. The XL checkpoints in `install/models.json` *are* the
newest thing there is.

**Two community issues describe our exact complaint, and both were closed "not planned"
with no maintainer reply.** This is the most decision-relevant external finding in the
report:

- `https://github.com/ace-step/ACE-Step-1.5/issues/1114` — "Technical Feedback on
  ACE-Step 1.5 XL (Music Quality & Expressiveness)": vocals "technically accurate but
  lack expressive nuance", "little to no continuous pitch modulation (pitch bending,
  scooping)", "vibrato is overly uniform", **"insufficient difference between verse and
  chorus intensity"**, timing **"overly grid-aligned… lacks human-like micro-timing
  variation"**. Its framing — "technically correct music generation vs. expressive
  musical performance" — is the owner's complaint in someone else's words.
- `https://github.com/ace-step/ACE-Step-1.5/issues/1063` — XL reported as *worse* than
  2B on melodic coherence; `https://github.com/ace-step/ACE-Step-1.5/issues/1203`
  reports "sudden tempo changes and beat chaos" on `xl-sft`.

**Do not wait for an upstream fix.** No 2.0, four months since the last release, and
these two issues closed unanswered. Whatever we want, we build or we switch.

**The models worth knowing about, August 2026:**

| Model | Params | VRAM | Licence | Weights | Lyrics/vocals | Why it matters to us |
|---|---|---|---|---|---|---|
| **DiffRhythm 2** (`https://huggingface.co/ASLP-lab/DiffRhythm2`) | ~1.1B | **~6–8 GB** | **Apache-2.0** | Yes | Yes | **Takes LRC lyrics with `[mm:ss.xx]` timestamps as its primary input.** The only open model with real per-line *timing* control. Fits our card today |
| **HeartMuLa-oss-3B** (`https://huggingface.co/HeartMuLa/HeartMuLaGen`) | 3B + 1.5B codec | 14–20 GB full; 10–14 GB with model swap; 4-bit ≈4.9 GB (validated for 16 GB) | **Apache-2.0** | Yes | Yes | **Mood is a first-class structured tag**, plus **per-section natural-language style prompts**. The closest thing in the field to what we are asking for |
| **SongBloom** (`https://huggingface.co/CypressYang/SongBloom`) | 2B | 8 GB fp32 / **6 GB bf16** | **Apache-2.0** | Yes | Yes | Cheapest second opinion on vocals. Section flags prepended per lyric paragraph |
| **MiniMax-Music3** | 11.1B | 24 GB std / **9.8 GiB Q4 GGUF measured** | Community (commercial OK under $20M revenue, must credit) | Yes | Yes | Already installed here. See below |
| LeVo 2 / SongGeneration 2 | 4B | 10–16 GB base | ❌ **NON-COMMERCIAL** — verified from Tencent's own card | Yes | Yes | Wins the academic evals; licence rules it out unless this stays personal |
| JAM-0.5 | 530M | 8 GB | ❌ commercial prohibited | Yes | Yes | Best timing control in the field (word/phoneme-level timestamps), wrong licence |
| Muse (Fudan) + **MuseCritic** | Qwen3 0.6–8B | unpublished | MIT code / Apache-2.0 weights | Yes | Yes | The only complete open pipeline, **and MuseCritic (`https://arxiv.org/abs/2608.11755`, Aug 2026) ships a reward model plus a `muse_grpo/` directory for RL-finetuning toward an aesthetic reward** — the only open route to optimise *expressiveness* directly. Community traction near zero; treat as a research artifact |
| Stable Audio 3.0 | 433M / 1.4B | 1.7–6.5 GB | Community, commercial under $1M | Yes | **No — "our models don't output intelligible vocals"** | Best instrumental model; LoRA is first-class. Not a song generator |
| MusicGen / MAGNeT | — | — | **CC-BY-NC** | Yes | **No** | No new model since Nov 2024. Dead end |
| Qwen-Music (`https://arxiv.org/abs/2607.11699`) | 33B | — | — | **No weights** | Yes | Native `[verse]`/`[chorus]` tokens, >5M training hours. Unreleased and would not fit anyway. **Note: "Qwen3-Music" does not exist** |
| Mureka, Seed-Music, WanSong | — | — | — | **No weights** | — | API or paper only |

**Where ACE-Step sits:** it is still the most-adopted open music model — 51,355
monthly downloads vs MiniMax's 16,644 and HeartMuLa's 924. On quality, *multiple
independent users* in
`https://huggingface.co/MiniMaxAI/MiniMax-Music3/discussions/4` say MiniMax output is
"cleaner than ace-step" — but note the ceiling they accept: the praise is "like Suno
**v3.5**". Nobody claims any open model matches current commercial output. The one
non-vendor academic head-to-head has LeVo 2 beating ACE-Step 1.5 on every subjective
axis, but that is Tencent scoring their own model. **No consensus ranking on vocal
expressiveness exists** and the subagent declined to invent one; ACE-Step is simply the
only model *explicitly criticised* for flat vocals, which may reflect its larger user
base as much as its quality.

**Caveat on that consensus, stated by the subagent:** Reddit was **completely
inaccessible** through every route tried. Q4 reflects the HF/GitHub builder population,
not r/LocalLLaMA — and on subjective vocal-expressiveness questions those populations
likely differ. Treat the community read as partial.

**MiniMax — the settled question, re-checked once and still settled.** The brief closed
this and I spent no time re-litigating it, but two facts arrived unbidden and both
*support* the existing conclusion, so they are worth one line. MiniMax-Music3 went
open-weights on 2026-08-13, and third-party LoRA training exists
(`https://github.com/filliptm/ComfyUI-FL-MiniMaxMusic3`, SimpleTuner backend). But in
`https://huggingface.co/MiniMaxAI/MiniMax-Music3/discussions/10` the community
disagrees about whether it is real: `ostris` and `RazzzHF` argue the **unreleased RVQ
encoder makes the conditioning improper**, and `bghira` — who has it running — warns
"not really in a useful manner just yet — don't waste your compute budgets." **That is
the same missing-encoder wall `train/README.md` already identified.** No change.
Separately, `FreeDiddy`, `TestingAitestingTesting` and `bghira` independently note
MiniMax has **no audio-reference input at all** for the same reason — which is why
`supports_cover=False` in our studio, exactly as documented.

### 7.4 The two moves this section actually recommends

**1. Quantised XL on the 10 GB card, today.**
`https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF` publishes ACE-Step 1.5 XL in
GGUF — **Q4_K_M at 2.8 GB**, through Q8/BF16 at 9.3 GB — and
`https://github.com/ace-step/acestep.cpp` runs both the 5Hz LM *and* the DiT, supports
LoRA (PEFT directories and ComfyUI safetensors), CUDA/Vulkan/CPU, with prebuilt Windows
binaries. **This puts the 4B XL quality tier on the current card without waiting for a
GPU** — which contradicts `install/models.json`'s 11.5–12 GB figure, because that figure
is for bf16.

Caveat flagged by the subagent and worth verifying before spending an evening: **the
acestep.cpp README does not list XL variants even though the GGUF repo publishes them.**
Also unknown whether a PEFT LoRA trained against bf16 `acestep-v15-turbo` loads correctly
against a quantised XL — almost certainly not, since it is a different base with
different dimensions. Treat GGUF XL as a *generation* path, not a LoRA path.

**2. Add DiffRhythm 2 as a second engine for timing-critical work.**
Apache-2.0, ~6–8 GB, fits today, and it takes timestamped LRC lyrics as its primary
input. The owner asked for "ideal timing/structure"; **ACE-Step structurally cannot take
a timed structure plan as input (§7.2), and DiffRhythm 2 can.** This is the one place
where the honest answer is "a different model already solves this", and the studio
already has a multi-engine architecture (ACE-Step + MiniMax) to slot it into.

*(See §8 for anything that could not be confirmed.)*

---

## 8. What I could not determine

Stated plainly, because guessing here would be worse than a gap.

1. **Whether any of this sounds better.** This was a read-only mission: no inference,
   no training, no listening. Every quality claim is architectural reasoning, not an
   A/B. In particular, options 3, 5, 6 and 12 are *plausible* emotional/structural
   levers because of what the code does, not because I heard them work.
2. **Exact token counts.** All caption/lyric token figures (the ~21% and ~22%) are
   estimated as characters ÷ 3.6. I did not load the Qwen3 tokenizer, because that
   would have meant touching the venv. The true figures could plausibly be ±20%
   relative. The *ordering* fact — that `# Metas` is last in `SFT_GEN_PROMPT` and so
   truncates first — is verified from `constants.py:160-168` and is not an estimate.
3. **VRAM for planner QLoRA on the 3080.** Estimated at 8–10 GB from parameter count.
   Not measured. If it does not fit, the fallback is a rented 24 GB card for an hour.
4. **What the 5Hz LM was trained on.** The checkpoint ships with no training
   description in `checkpoints/acestep-5Hz-lm-1.7B/`, and I did not confirm upstream
   whether the audio-code vocabulary is shared with, or fine-tunable alongside, the
   DiT's FSQ codebook. If the codebooks diverge, planner SFT gets harder.
5. **Whether `repaint`, `complete`, `lego` and `extract` work end-to-end here.** They
   are implemented and declared (§2.1). I traced the code paths; I did not run them.
   `lego` in particular has a second mode gated on `is_lego_sft`
   (`conditioning_text.py:73-77`) that expects an SFT-stems checkpoint we do not
   appear to have — so `lego` may be the weakest of the four on our checkpoints.
6. **Whether the existing LoRA actually degrades under cover/thinking.** §3.4 shows it
   was trained with silence context latents, zero timbre and no LM hints, so it *should*
   be off-distribution there. Untested.
7. **ACE-Step 1.5's own training cost.** v1 discloses enough to price (§6.2d); **1.5
   does not.** Its dataset is given only in *samples* (20M pairs / 27M corpus / 6M stems
   / 2M HQ SFT), never in hours, and GPU count is disclosed only for the 1D VAE with no
   wall-clock. Independently corroborated by
   `https://artintech.substack.com/p/ace-step-15-explained`: "the reinforcement learning
   components, parts of the training pipeline, and the training data are not publicly
   available." So every 1.5 cost figure in this report is extrapolated from v1.
8. **Wall-clock for YuE, JAM, LeVo, LeVo 2, DiffRhythm v1, SongGen, MusicLM.** GPU
   counts disclosed, durations not, so GPU-hours are not derivable. MusicGen never
   states its GPU *type* at all. **No foundation-model paper in this field states a
   dollar cost** — every USD figure in §6 is my conversion at §6.1b prices.
8b. **Vast.ai and GB200 primary pricing.** Vast injects prices client-side and returned
   blanks; no primary GB200/GB300 hourly price was found anywhere. The GB200 row in
   §6.1b is secondary-aggregate only.
9. **Whether the LM's `structure` field carries timestamps, durations, or anything at
   all.** The tech report names it; the shipped FSM has no state for it. This is the
   single most important open question in the report and option 0 answers it in five
   minutes.
10. **Reddit sentiment on which local model sounds best.** Reddit was completely
    inaccessible to the research — every route blocked. The community read in §7.3 is
    the HF/GitHub builder population only, and that is exactly the population *least*
    likely to discuss subjective vocal expressiveness. Treat it as partial.
11. **Whether MiniMax-Music3 fits 10 GB.** The only measurement found is a Q4/Q8 GGUF
    mix peaking at **9.8 GiB on a 5090**; a 3080 gives ~9.3 GiB usable. **Zero
    first-hand reports of it running on 10 or 12 GB exist**, and the vendor's "8 GB via
    group offloading" claim is uncorroborated. Irrelevant to the fine-tuning question,
    which stays closed.
12. **Whether GGUF XL actually works through `acestep.cpp`.** The GGUF repo publishes XL
    variants; the `acestep.cpp` README does not list them. Verify before committing an
    evening (§7.4).
13. **Head-to-head accuracy of singing forced-aligners** (SOFA vs MFA vs WhisperX vs
    STARS). No 2026 comparison found; SOFA publishes no headline metric. The subagent
    named this the weakest-evidenced item in its own report, and I am repeating that
    warning rather than laundering it.
14. **Whether ACE-Step's XL VRAM figure is 9 GB or 11.5–12 GB.** Upstream says ~9 GB
    bf16; `install/models.json` says 11.5–12 GB and claims to be "verified against
    ACE-Step's own `gpu_config.py` profiling, 2026-08-20". Note carefully: that is
    **derived from local profiling tables, not from a live XL run** — `install.ps1`
    hides what will not fit, so XL may never have executed on this box at all. I lean to
    our figure because it is at least locally derived, but the gap is large enough that
    XL may be closer to fitting in bf16 than we think. Neither number is measured.
15. **Whether any of the emotion/structure tag syntax in §7.2 measurably works.** It is
    officially documented syntax. No paper tests it. Our own data barely uses it. That
    combination is exactly why option 1c is cheap and worth trying, and exactly why I
    cannot promise it will work.


## 9. YuE2 (m-a-p, September 2026) — integrated 2026-09-11, unverified on hardware

**Status.** Wired in as the fourth engine (`studio/workers/yue2_worker.py`,
`supervisor.BACKENDS["yue2"]`, the `score` capability, the Score group in the
Create pane, `install/models.json` venv + weights gated at 24 GB). The
package installs and imports on Windows with the cu128 torch build, so the
"Linux" line in its README is a support statement, not a dependency. The whole
path was exercised against a stub pipeline with YuE2's signatures; the model
itself has not been loaded here (10 GB card). §9.1 below is the original
assessment, kept because its reasoning still holds — only "not going in now"
has changed to "in, waiting for the card".

Two things the source showed that the README does not say: `quantization="fp8"`
exists (compute capability 8.9+, so RTX 40/50 only) and `offload_ar=True`
parks the language model in RAM during synthesis. Together they may bring a
16 GB card into range; nothing below 24 GB has been measured. The worker exposes
both as environment knobs (`YUE2_QUANT`, `YUE2_OFFLOAD_AR`) and the studio's
gate can be lowered with `YUE2_VRAM_GB` to try it.

### 9.1 The assessment (2026-09-10)

Source: github.com/multimodal-art-projection/YuE and map-yue2.github.io, read
2026-09-10. Package `yue2` v0.1.6, weights `m-a-p/YuE2-3B`.

**What it is.** A 3.59B-parameter generator with a *symbolic* intermediate: the
model first writes a score plan in ABC notation (melody, chords, structure,
tempo), then generates semantic tokens against that plan, then synthesises
48 kHz stereo. The pipeline is `plan() -> generate_semantic() -> synthesize()
-> decode()`, with `cot` = `full` / `melody` / `off` choosing how much of the
plan is written before audio, an `abc` parameter for handing it a score, and a
`seed`.

**What translates into our setup.**

| YuE2 feature | what it gives us | studio surface it needs |
|---|---|---|
| ABC score plan (`cot=full`) | tempo, key, chords and section structure fixed *before* audio — the structure control ACE-Step's sliding window (§3) cannot give | a score panel: show the generated ABC, let the user edit and re-render from it |
| `abc=` input | zero-shot covers from a *score* rather than from audio; the same song in a new style with the melody held | a "cover from score" mode alongside the audio cover we have |
| SheetSage2 transcription | audio -> score, so any recording (including our own catalogue) becomes an editable plan | a transcribe-to-score action on a library row |
| agentic multi-turn editing | "make the chorus longer", "raise the key" as edits to the plan, not re-rolls | a chat-style edit box over the score |
| `cot=melody` | melody-only plan, cheaper, keeps the model freer on harmony | one more option in the thinking selector |

It is the only model in this document whose emotion/timing control is
*inspectable*: the plan is text you can read before the GPU spends a minute on
it. That is a different kind of control from anything ACE-Step or MiniMax
offer, and it is exactly what the "ideal timing" ask in §0 was about.

**Why it could not run here.**

1. **24 GB of VRAM in bf16.** No quantised path is *documented* (fp8 exists in
   the source; see the status note above). This box has 10 GB shared with
   ComfyUI; even alone it does not fit. It goes in the same column as ACE-Step
   XL and HeartMuLa: waits for the bigger card.
2. **Weights are CC BY-NC 4.0.** The code is Apache-2.0 but the model is
   non-commercial. If any Xenowiz release is ever sold, tracks rendered through
   YuE2 are a licence problem. ACE-Step and MiniMax do not have this restriction.
3. **No LoRA, no fine-tune recipe.** The catalogue-style work in §4 has nowhere
   to go on this model.
4. **A new worker.** It is a fourth venv, a fourth engine in `supervisor.py`,
   and a capability block for the score features — a real integration, not a
   checkbox.

**What was built (2026-09-11).** A `yue2` backend with the `score` capability
(plan modes full / melody / off, a pasted or planned ABC score, plan-only
requests that return the score without audio) and a Score group in the Create
pane that appears only for it; steps and duration became gated features and
leave with it. Every render keeps YuE2's own artefact folder
(`Music/studio/yue2/<stem>/`: score.abc, plan.json, latent.npy, result.json)
and the sidecar records the score it sang to, so Reuse restores it.
**Not built:** SheetSage2 transcription (separate venv, torch 2.8, FFmpeg 6.1
shared libs) — a transcribed score pastes into the box meanwhile; mid-request
cancel (YuE2 has a `cancelled` callback; the supervisor has no cancel path).
