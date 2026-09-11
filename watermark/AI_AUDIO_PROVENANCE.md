# AI Music: Watermarking, Tagging, and Detection

How AI-generated audio gets marked, and how it gets caught. Three independent
layers, with very different strengths — and different failure modes.

| Layer | Needs generator cooperation? | Survives processing? | Trivially removed? |
|---|---|---|---|
| **1. Passive artifacts** (fakeprints) | No | Partly | No — but degrades |
| **2. Active watermarks** (AudioSeal, SynthID) | Yes | Yes, well | Yes, if targeted |
| **3. Metadata** (C2PA, ID3, DDEX) | Yes | No | Yes, trivially |

The layers are complementary. Metadata says what the file *claims*; a watermark
says what the generator *stamped*; a fakeprint says what the audio *is*. Only
the third works on a generator that never cooperated.

---

## Layer 1 — Passive detection (spectral "fakeprints")

**The physics.** Nearly every neural audio generator ends in a vocoder that
upsamples a low-rate latent to waveform using **transposed convolutions**. That
operation is periodic by construction, and it stamps **evenly spaced peaks**
across the spectrum. The peaks are inaudible but survive in the average
spectrum. This is the basis of *"A Fourier Explanation of AI-Music Artifacts"*
(ISMIR 2025).

**The feature ("fakeprint")**, exactly as implemented in
`lofcz/ai-music-detector` and re-implemented in `ai_audio_forensics.py`:

1. Mono, resample to **16 kHz**, cap at 300 s.
2. STFT, `n_fft = 8192`, `hop = 4096`, power spectrum.
3. `10·log10`, clamped to `[1e-10, 1e6]`; average over time.
4. Keep bins **1000–8000 Hz** → 3585 features.
5. `hull = minimum_filter1d(spec, size=10)` — a local noise floor.
6. `residue = clip(spec − hull, 0, 5)`, then divide by its own max.
7. Logistic regression (3585 weights + bias) → sigmoid.

Step 5 is the trick: the minimum filter estimates the floor *between* peaks, so
subtracting it isolates the comb.

**Known signatures.** Suno is reported to show a 32 kHz sampling signature,
"digital haze" around 8–16 kHz, and characteristic HF roll-off. Different
generators produce different comb spacings — the artifact is a fingerprint of
the vocoder's upsampling factors, not of "AI" in the abstract.

**Reported accuracy:** 99.88 % over 17,866 samples (P 0.9985 / R 0.9998,
FPR 0.31 %) for Suno ≤ 5 and Udio ≤ 1.5. A 10k-parameter logistic regression
reportedly exceeds 99 % on DAC, EnCodec, Musika, Suno and Udio.

**Where it breaks.** Accuracy falls once output is re-encoded, compressed or
mixed in a DAW; real-world figures on professionally produced tracks are cited
around 85–93 %. It also only knows the generators it was trained on — a newer
model, or one with a different vocoder, is out of distribution.

### ⚠ The confound you must control for

Musical harmonic series **also** produce evenly spaced spectral peaks. A
sustained 199 Hz bass note puts energy at 199, 398, 597 … Hz — indistinguishable
from a comb by spacing alone.

The discriminator is **stability over time**: a vocoder artifact sits at a
*fixed* spacing for the whole file, while harmonics move with the chords. Test
it by computing the fakeprint over successive windows and watching the peak lag
(see "Worked example" below). Do not skip this — it is the difference between
evidence and a coincidence.

---

## Layer 2 — Active watermarking

Deliberately embedded, inaudible, survives processing. Only present if the
generator (or a later tool) put it there.

### AudioSeal (Meta) — installed here

- **MIT licensed**, code *and* weights. Commercial use allowed.
- Generator 14.7 M params, detector 8.65 M. Trained at **16 kHz**.
- **Localized**: detects per-sample, so it can mark *which segment* of a longer
  recording is generated, at 1/16000 s resolution.
- Carries an optional **16-bit message**.
- Strongest on speech/voice; weaker on music.

```python
from audioseal import AudioSeal
gen = AudioSeal.load_generator("audioseal_wm_16bits")
wm  = gen.get_watermark(wav, sample_rate=16000, message=bits)
marked = wav + wm                       # the mark is additive
det = AudioSeal.load_detector("audioseal_detector_16bits")
prob, msg = det.detect_watermark(marked, sample_rate=16000)
```

Two gotchas found the hard way, both handled in the scripts here:
- **AudioSeal 0.2+ does not resample internally.** Passing `sample_rate=` is a
  no-op — you must feed it 16 kHz yourself or scores are meaningless.
- It wraps SEANet blocks in `torch.compile`, which needs Triton and **fails on
  Windows**. Set `NO_TORCH_COMPILE=1` *before* importing `audioseal`.

### SynthID (Google DeepMind) — not runnable locally

- Covers Lyria (music), Veo, Imagen; >100 billion items marked by May 2026.
- Now cross-vendor: **OpenAI, ElevenLabs and Kakao** emit SynthID too.
- **SynthID *Text* is open-source** (in HF Transformers ≥ 4.46). **SynthID
  *Audio* is not** — no public weights, no public technical paper. Detection is
  only via Google's SynthID Detector web portal.
- Consequence: its robustness numbers are vendor claims, not independently
  benchmarked. You cannot verify a file locally.

### Others worth knowing

- **WavMark** — open source, message-carrying.
- **Timbre watermarking**, **StreamMark** (semi-fragile, for deepfake
  detection), **LambdaMark** (semantic, claims "radioactivity" — survives being
  trained on).

### The robustness caveat that matters

Published work (Yao, Huang & Wang, AAAI 2026) reports **overwriting attacks
drive AudioSeal, WavMark and peers to nearly 100 % attack success**. Watermarks
are robust to *incidental* processing, not to a *deliberate* adversary. Treat a
watermark hit as strong positive evidence and a watermark miss as almost no
evidence at all.

---

## Layer 3 — Metadata and provenance manifests

### C2PA / Content Credentials

An open standard (Coalition for Content Provenance and Authenticity). A
**Content Credential** is a cryptographically signed manifest recording origin,
edit history, and whether AI was used. Adopted by OpenAI among others.

Strength: signed, so it can't be forged. Weakness: it is **container metadata**
— stripping it is trivial and leaves no trace, and absence proves nothing. It's
a positive-evidence channel only.

Verify properly with `c2patool` (Rust, from the C2PA project). The byte-scan in
`ai_audio_forensics.py` only tells you a manifest is *present*, not valid.

### Container tags and DDEX

ID3 / Vorbis comments may name the generator outright (`Suno`, `Udio`,
`MusicGen`, …). **DDEX** disclosure fields are the music industry's route for
declaring AI involvement in distribution metadata. Both are trivially editable —
useful when present, worthless when absent.

---

## What's installed here

```
B:\AudioDev\watermark\
  .venv\                        Python 3.10, CPU torch (these models are small)
  models\ai_music_detector.onnx logistic-regression fakeprint classifier
  ai-music-detector\            lofcz/ai-music-detector source (reference impl)
  ai_audio_forensics.py         all three layers in one report
  make_controls.py              builds an AudioSeal-watermarked control
  robustness_test.py            does a watermark survive transcoding/SR?
  visualize_fakeprint.py        plots the residual + its self-similarity
```

AudioSeal weights download automatically from HuggingFace on first use.

```powershell
B:\AudioDev\watermark\.venv\Scripts\python.exe B:\AudioDev\watermark\ai_audio_forensics.py "track.wav"
```

---

## Worked example — why you must validate before believing a detector

Run against `Paycheck to Paycheck.wav` (48 kHz, 4:06, lossy-sourced):

| check | result |
|---|---|
| fakeprint classifier | **AI-GENERATED, p = 1.0000** (logit +26.2) |
| AudioSeal | none (p = 0.0375) |
| C2PA manifest | absent |
| container tags | none |

The classifier is maximally confident. Before accepting that, two controls:

**Control A — does the pipeline manufacture the verdict?** A known-non-AI file
scored p = 0.0000. Pushed through 64 kbps MP3 → **AudioSR regenerating
11–24 kHz from scratch via a HiFi-GAN vocoder** → still p = 0.0000. Apollo the
same. So heavy neural super-resolution does **not** create a false positive —
good news for this repo's restoration pipeline, and it means the verdict is a
property of the source material.

**Control B — is the comb fixed or musical?** Fakeprint peak lag over twelve
successive 20 s windows:

```
199, 199, 400, 400, 184, 199, 307, 600, 150, 148, 600, 201  Hz   (std 81 bins)
```

It **moves**, and it moves between musically related values (199 → 400 ≈ 2×,
600 ≈ 3×). That is the signature of harmonic content tracking the chords, **not**
of a fixed vocoder artifact.

**RESOLVED — the classifier was right.** The answer was in the file's RIFF
container the whole time:

```
LIST/INFO  ICMT: "made with suno; created=2026-08-06T22:27:55Z;
                  id=79062206-e92d-4c7c-b2c5-de37b8db738a"
           ISFT: Lavf60.16.100
```

The track is Suno-generated and says so in an embedded comment tag, with a
timestamp and a track UUID. `p = 1.0000` was a **true positive**, and the eight
failed attempts below to manufacture a false positive failed because the
detector is genuinely good — not because the controls were inadequate.

### The process failure worth remembering

`mutagen.File()` returns **no tags** for a plain PCM WAV carrying a `LIST/INFO`
chunk — it handles WAVE-with-ID3, not RIFF INFO. Layer 3 duly reported
"0 tags", that was taken at face value, and hours of spectral analysis followed
while a plain-text `made with suno` sat 164 bytes into the file.

**Read the container yourself before doing any signal analysis.** It is the
cheapest check and the most likely to be decisive. `wav_chunks.py` dumps the
chunk list; `ai_audio_forensics.py` now parses `LIST/INFO`, `bext`, `iXML` and
`_PMX` directly rather than trusting a tag library.

### Bit-perfect copy and container-strip test

| version | PCM sha256 (first 24) | tags | p(AI) |
|---|---|---|---|
| original | `5b9da7a29f29f29fdde20209` | ICMT = made with suno | 1.0000 |
| byte-for-byte copy | `5b9da7a29f29f29fdde20209` | ICMT = made with suno | 1.0000 |
| metadata stripped | `5b9da7a29f29f29fdde20209` | none | **1.0000** |

All three carry identical samples (`np.array_equal` → True). Stripping the
container removes the confession but **not** the flag — which is the whole point
of passive detection. Metadata is a courtesy; the artifact is in the waveform.

**The general lesson:** a saturated probability from a black-box classifier is
not evidence on its own. Always run (a) a known-negative control of *similar*
material, and (b) an interpretable check of the mechanism the model claims to
use.

### What actually drives the score (logit attribution)

The logistic regression is linear, so the logit decomposes exactly:
`logit = bias + Σ wᵢ·xᵢ`. Splitting that sum by frequency band:

```
bias (baseline pull toward "AI")            +4.98
weights: 1437 positive, 2148 negative       pos-sum +193.7, neg-sum -273.0
```

| band | contribution, flagged track | contribution, control |
|---|---|---|
| 1–2 kHz | +1.89 | −2.22 |
| 2–3 kHz | +0.22 | −5.63 |
| 3–4 kHz | +3.16 | −4.34 |
| 4–5 kHz | +4.86 | −1.12 |
| 5–6 kHz | +1.94 | −3.46 |
| 6–7 kHz | +4.39 | −1.03 |
| 7–8 kHz | +4.36 | −2.38 |

**No single band or comb frequency dominates** — the evidence is spread across
the whole 1–8 kHz range. So the trigger is not "a comb at frequency X".

The separating variable is how the residual **thins out toward high frequency**:

| file | p(AI) | fp @1–2 kHz | fp @7–8 kHz | HF/LF ratio |
|---|---|---|---|---|
| flagged track | 1.0000 | 0.195 | 0.109 | **0.559** |
| ″ as 96 k MP3 | 1.0000 | 0.258 | 0.180 | **0.697** |
| ″ + AudioSR | 1.0000 | 0.257 | 0.180 | **0.701** |
| control | 0.0000 | 0.237 | 0.248 | 1.045 |
| control 64 k MP3 | 0.0000 | 0.235 | 0.226 | 0.959 |
| control + AudioSR | 0.0000 | 0.230 | 0.220 | 0.957 |
| control + Apollo | 0.0000 | 0.242 | 0.222 | 0.921 |

Perfect separation, with a clear gap between 0.70 and 0.92. A **flat** residual
profile reads as real; a residual that **decays toward HF** reads as AI.

**Why that is a shaky discriminator.** A residual stays large only where the
spectrum is *rough* at a 10-bin scale. White noise is rough everywhere — which
is exactly why the synthetic control scores so "real", and why it is a poor
proxy for music. Dense, layered, heavily limited HF (cymbals, reverb tails,
saturation) is *smooth*, so the minimum filter tracks it and the residual
collapses. That means **densely produced, loudness-maximised, lossy-sourced
music can look "AI" to this feature for reasons that have nothing to do with a
vocoder.** It is also consistent with the reported Suno signature of "digital
haze" in the upper bands — the same measurement, two very different causes.

### Processing does not move the score

Every stage of this repo's pipeline, applied to the flagged track:

| variant | p(AI) |
|---|---|
| untouched | 1.0000 |
| after `band_splice` + AudioSR | 1.0000 |
| after `band_splice` + FlashSR | 1.0000 |
| after 96 kbps MP3 | 1.0000 |
| after full AudioSR restore | 1.0000 |
| after full Apollo restore | 1.0000 |

And in the other direction, a file that scores real stays real through 64 kbps
MP3 plus AudioSR regenerating 11–24 kHz from scratch. **Restoration, splicing
and transcoding neither create nor remove this flag.** The verdict is a property
of the source recording.

## SOLVED: the signature is a 200.00 Hz vocoder comb

The mechanism, confirmed quantitatively. **Suno's output carries a comb of
spectral peaks spaced exactly 200.00 Hz apart, and the classifier's weights are
concentrated on that same grid.**

Measured as on-grid vs off-grid mean residual across 1–8 kHz (±2 Hz tolerance):

| spacing | model \|weights\| | Suno residual | control residual |
|---|---|---|---|
| 100 Hz | 2.20× | 1.49× | 0.99× |
| **200 Hz** | **3.36×** | **1.84×** | 0.95× |
| 250 Hz (off-grid) | 1.37× | 1.05× | 0.81× |
| 333 Hz (off-grid) | 1.77× | 1.19× | 0.92× |
| **400 Hz** | **3.49×** | **1.96×** | 1.00× |
| **600 Hz** | 3.20× | **1.94×** | 0.61× |
| **800 Hz** | **3.91×** | **1.94×** | 0.93× |

Every multiple of 200 Hz shows a strong elevation; nothing off that grid does.
Swept at 0.05 Hz resolution over 190–210 Hz, the peak lands on **200.00 Hz
exactly** — a machine number, not a musical one:

| file | peak spacing | ratio at 200.00 Hz |
|---|---|---|
| Suno original | **200.00 Hz** | **2.995** |
| Suno + Apollo restoration | 200.00 Hz | 1.780 |
| Suno + DeepFilterNet | 200.00 Hz | 1.778 |
| synth control | 196.05 Hz (noise) | 0.947 |
| control via Encodec 3 kbps | 202.75 Hz (noise) | 0.929 |

**200 Hz = a 160-sample hop at 32 kHz**, which is Suno's reported generation
rate (equivalently 240 samples at 48 kHz). That is the vocoder's frame rate, and
transposed-convolution upsampling at that rate stamps the comb. The original
ISMIR "Fourier explanation" is exactly right.

Note the comb *survives* Apollo restoration and neural denoising at ~1.78×,
which is why neither cleared the flag.

Figure: `Music\vocoder_comb_200hz.png`.

### Two of my own hypotheses that this killed

**"There is no comb — disregard it."** An earlier pass tested 199.2 Hz spacing
in the 6–8 kHz band on the *raw spectrum* and found −5.25 dB, and I concluded the
comb was a coincidence. That measurement was wrong in three ways: too narrow a
band (few comb periods), too coarse a phase tolerance, and the raw spectrum
instead of the min-filter residual the detector actually uses. Measured properly
over the full 1–8 kHz on the residual, the comb is unambiguous at 3×.

**"It's a spectral-peakiness meter."** Mean residual separated all 14 test files
perfectly (flagged 0.129–0.214, clean 0.228–0.631), which looked like the
mechanism. It is only a correlate. Rescaling a control's spectral contrast until
its mean residual fell to 0.194 — inside the flagged range — left it at
p = 0.0000. And the decisive test: **randomly permuting the 3585 feature bins
takes the Suno track from 1.0000 to 0.0000** (20 trials, zero variance); so does
sorting, reversing, or even shuffling *within 50-bin windows*. The verdict
depends on precisely *which* frequencies carry residual, at ~2 Hz resolution —
a positional template, not a summary statistic.

## A generator-agnostic comb detector (`comb_detect.py`)

If the artefact is physical — peaks every F Hz, where F is the vocoder's frame
rate — then it can be found by search instead of training. No weights, no
per-generator retuning: average spectrum → minimum-filter residual → sweep
candidate spacings → score.

**Scoring matters more than the sweep.** A naive argmax over the ratio fails
twice: a comb at F also lands on multiples of 2F/3F/4F where the ratio is often
*higher* (fewer, stronger on-grid bins), so it reports a harmonic instead of the
fundamental; and the max of a noisy statistic over thousands of candidates is
inflated, so ordinary audio false-positives. First attempt reported 800 Hz for
Suno and flagged every control. Fixed by (a) scoring the **geometric mean of the
ratio at F, 2F and 3F** — real comb structure repeats, noise does not — (b)
preferring the smallest spacing within 85 % of the top score, and (c) capping the
search at 400 Hz, above which no plausible frame rate lives.

| file | spacing found | score | z | verdict |
|---|---|---|---|---|
| Suno original | **200.00 Hz** | 2.03 | 9.7 | comb → hop 160 @ 32k |
| **AudioSR output** | **100.00 Hz** | 2.47 | 17.4 | comb → hop 480 @ 48k |
| **FlashSR output** | **100.00 Hz** | 2.73 | 19.7 | comb → hop 480 @ 48k |
| Encodec 24 kbps | 299.95 Hz | 1.76 | 8.6 | comb |
| synth control (raw) | 73.25 Hz | 1.22 | 3.0 | none |
| synth control (mp3) | 146.50 Hz | 1.33 | 3.7 | none |
| noise+sweep control | 122.70 Hz | 1.27 | 5.9 | none |

**Predictions verified against source code.** The detector recovered 100.00 Hz
from AudioSR and FlashSR audio alone; both packages define
`hop_length = 480, sampling_rate = 48000` → 48000/480 = exactly 100 Hz. It
recovered 200.00 Hz from Suno, matching a 160-sample hop at its reported 32 kHz
generation rate. The method reads an architectural parameter straight out of the
waveform.

### ⚠ Correction: super-resolution DOES leave a vocoder fingerprint

Earlier in this document, "heavy super-resolution does not make real audio look
AI-generated" — measured against the Suno-tuned classifier, which reported
p = 0.0000 for AudioSR output. That conclusion was **specific to that detector,
and wrong in general.** AudioSR and FlashSR stamp a 100 Hz comb that is *more*
prominent than Suno's own 200 Hz comb (score 2.47/2.73 vs 2.03). The shipped
classifier simply cannot see it: its weights are pinned to the 200 Hz grid.

Practical consequence: audio processed through this repo's restoration tools
carries a detectable neural-vocoder signature. Today's Suno detector ignores it;
a generator-agnostic detector — like the one in this directory — does not.

### Stem separation does not remove it (`demucs`)

Demucs `htdemucs` is a full separate-and-recombine resynthesis — far more
invasive than the band-limited SR tools. Split the Suno excerpt into four stems,
sum them back:

| file | spacing | score | z |
|---|---|---|---|
| original | 200.00 Hz | 1.74 | 8.1 |
| **recombined stems** | **200.00 Hz** | **1.85** | 8.9 |
| **vocals stem** | **200.00 Hz** | **1.86** | 10.9 |
| drums stem | 100 Hz | 1.32 | none |
| other stem | 155.6 Hz | 1.41 | none |

The comb survives intact — slightly *stronger* after the round trip — and the
trained classifier still calls the recombination p = 1.0000.

Two things worth noting. The artefact concentrates in the **vocals** stem and is
absent from **drums**: a frame-rate comb needs sustained tonal content to sit on,
and broadband transients wash it out. And Demucs adds **no comb of its own**,
unlike AudioSR/FlashSR. That is architectural — Demucs predicts masks and
operates largely on the waveform, preserving input structure, whereas
AudioSR/FlashSR generate a mel latent and run it through a HiFi-GAN vocoder.
**Masking preserves; latent-to-vocoder generation stamps.**

### Time-domain confirmation: frame-locked waveform structure (`frame_fold.py`)

A 200 Hz comb implies a 5 ms period in the waveform. Test it without any STFT:
take the second difference (flattens smooth musical content, exposes
sample-scale joins), chop into blocks exactly one frame long, and average.
Anything not locked to the frame grid averages away as 1/N; a synthesis artefact
survives.

| file | period | gain | null | ratio |
|---|---|---|---|---|
| Suno track | 240 (200 Hz) | 63.4 | 1.12 | **56.9×** |
| AudioSR output | 240 | 387.8 | 2.80 | **138×** |
| FlashSR output | 480 (100 Hz) | 24.7 | 1.06 | **23.3×** |
| synth control | 240 | 0.98 | 0.98 | **1.00** |
| synth control | 480 | 0.93 | 1.09 | 0.86 |

The control sits at **exactly 1.00** — a perfect null — while generated audio
runs 23–138×. This is the cleanest discriminator found in this whole
investigation, and the simplest: raw waveform, no spectrum, no learned weights.
Figure: `Music\frame_lock_200hz.png`, where the frame-locked pattern is plainly
visible against a flat null.

Limitation: the frame rate must divide the sample rate exactly (200 Hz needs an
integer period; at 44.1 kHz that is 220.5, so the file must be resampled or the
fold done fractionally).

### Why spectral cleanup cannot remove it

This explains the denoising results. The artefact is **frame-locked,
phase-coherent structure in the waveform**, not additive spectral noise sitting
on top of the music. Spectral-domain processing — denoisers, spectral repair,
magnitude editing — modifies magnitudes and leaves the frame-synchronous
time-domain pattern intact. That is why every denoiser tested left the verdict
unchanged, and why the one setting that moved it did so by destroying the
*discriminative* information for both files rather than by removing an artefact.

### ⚠ A 0.1 % speed change defeats the trained classifier (`warp_probe.py`)

The single most consequential robustness result here. Resampling scales every
frequency by the speed ratio, so a 200 Hz comb *moves* to 200×r Hz — it is not
damaged, just relocated. A detector that searches for the spacing follows it. A
detector with weights pinned to the 200 Hz grid loses it completely.

| treatment | predicted comb | comb found | searching score | trained classifier |
|---|---|---|---|---|
| baseline | — | 200.00 Hz | 1.74 | **1.0000** |
| resample **+0.1 %** | 200.20 | 200.15 | 1.50 | **0.0006** |
| resample +0.5 % | 201.00 | **201.00** | 1.75 | **0.0000** |
| resample +2 % | 204.00 | **204.00** | 1.75 | **0.0000** |
| resample −3 % | 194.00 | **194.00** | 1.72 | **0.0000** |

**+0.1 % is about 1.7 cents — inaudible — and it takes the classifier from
p = 1.0000 to p = 0.0006.** The searching detector tracks the shift to within
0.15 Hz and, at ±0.5 % and beyond, to the exact predicted frequency.

This is a detection-side conclusion, not a recipe: it says any deployment of a
fixed-grid classifier should resample-sweep its input (test a range of ratios, or
search spacings) before trusting a negative, and that a clean result from such a
detector is worth very little. Note the metadata layer is untouched by any of
this — the `made with suno` tag survived every treatment in this table.

### What does and does not disturb the comb

| treatment | comb after | trained classifier | reading |
|---|---|---|---|
| time-stretch +0.5 % (`atempo`) | 200.00 Hz | 1.0000 | pitch preserved → comb stays put |
| time-stretch +3 % | 200.00 Hz | 0.9976 | same |
| time-stretch −5 % | lost (155.6) | 0.9908 | classifier still fires |
| **vibrato (continuous pitch warp)** | lost (174.5) | **0.0000** | modulation smears the grid |
| MP3 64 kbps | lost (249.3) | 0.9881 | classifier more robust than search |
| MP3 32 kbps | lost (115.4) | 0.0000 | both fail |
| Opus 48 kbps | lost (175.0) | 0.9976 | classifier survives |
| heavy compressor + exciter | 200.00 Hz | 1.0000 | dynamics don't touch it |

A prediction that failed: time-stretching was expected to smear the frame lock.
It does not — `atempo` preserves pitch, so the comb keeps its frequency, and the
classifier keeps firing. **Continuous pitch modulation (vibrato) is what breaks
it**, because the comb is smeared across frequency rather than moved coherently.

The two detectors are complementary in a useful way: the trained classifier
tolerates codecs and denoising but collapses under a trivial pitch shift, while
the searching detector tolerates pitch shifts but loses weak combs under heavy
lossy coding. Run both.

### Known false-positive mode: delay-based effects

The chorus/echo-smeared control flagged at 50.00 Hz (score 1.65, z = 18.3). That
is a genuine comb — but from delay lines, not a vocoder. **Any fixed delay
(chorus, flanger, echo, comb filter, or a room reflection) produces spectral
comb structure**, and nothing in the spectrum distinguishes its origin. A comb
alone is evidence of *periodic time-domain structure*, not of AI. Corroborate
with the frame rate decoding: 200 Hz and 100 Hz map onto standard hop/rate pairs;
an arbitrary 50 Hz from a 20 ms delay usually does not sit on a plausible one.

## The best detector found here: `frame_detect.py`

Detects every generator tested, needs no training, and returns cleanly on
controls — including the delay-effect case that false-positived the spectral
comb search.

| file | rate | gain | ratio | harmonic family | verdict |
|---|---|---|---|---|---|
| Suno | 200 Hz | 32.5 | **32.8×** | 100 / **200** / 400 | locked |
| ACE-Step generated | 250 Hz | 31.4 | **42.5×** | **25** / 50 / 125 / 250 | locked |
| ACE-Step VAE round-trip | 250 Hz | 52.3 | **75.3×** | **25** / 50 / 125 / 250 | locked |
| AudioSR | 200 Hz | 387.8 | **138.4×** | 50 / 66.7 / **100** / 200 | locked |
| FlashSR | 100 Hz | 24.7 | **23.3×** | — | locked |
| Encodec 24 kbps | 300 Hz | 69.6 | **64.5×** | — | locked |
| synth control (mp3) | — | 1.31 | **1.41×** | — | none |
| synth control + chorus/echo | — | 3.00 | **3.31×** | — | none |
| noise+sweep control | — | 1.08 | **1.08×** | — | none |

Separation is 10× wide: generated 23–138, controls 1.1–3.3. Every harmonic
family contains the architecturally-correct rate — 25 Hz for ACE-Step
(48000/1920, from its VAE config), 100 Hz for AudioSR (48000/480, from its
source), 200 Hz for Suno.

### Why a blind period search does not work

Two failed designs, both instructive.

**Epoch folding with coarse bins.** Folding needs about one phase bin per sample
to see a sample-scale join. Measured at the true period: Suno scores **63×**
with one bin per sample and **5.4** with 48 bins — the artefact is averaged away.
But at one bin per sample the trial period must be accurate to ~P/N (thousandths
of a sample over 20 s), and a blind scan at that resolution is hopeless.

**Uniform period grids are biased.** Step size must scale as P²/(N·bins), so a
uniform grid is either ruinously slow or silently misses the peak; and a grid
dense at short periods gives more trials there, so the argmax drifts to high
rates by chance. An early version reported 350 Hz for everything.

The fix is not a better search but a better candidate list: **vocoder frame rates
are not arbitrary.** They are sample_rate/hop for round hop sizes. Enumerating
those gives a few dozen candidates, each confirmable at full resolution in
milliseconds.

### Regenerating through a second model LAUNDERS attribution but not detection

ACE-Step's `cover` task takes an existing track and regenerates it in a new
style. Feeding it the Suno source asks a sharp question: does a second generator
overwrite the first's fingerprint, preserve it, or leave both?

Folding alone cannot answer it — 200 Hz is 8× 25 Hz, so a lock at either shows
at both. The discriminator is the **shape of the folded profile**: fold at 1920
samples and a true 25 Hz artefact leaves one feature per profile, while a 200 Hz
artefact leaves eight, spaced 240 apart. That shows up as energy concentrated on
multiples of 8 in the profile's own spectrum.

| file | on multiples of 8 | off | ratio | reading |
|---|---|---|---|---|
| **Suno source** | 1.19e2 | 8.9 | **13.39** | 200 Hz fundamental |
| **cover 1** | 5.11e1 | 3.74e1 | **1.36** | no 200 Hz structure |
| **cover 2** | 4.63e1 | 3.49e1 | **1.33** | no 200 Hz structure |
| ACE-Step text2music | 6.95e1 | 5.83e1 | 1.19 | 25 Hz fundamental |
| control | 3.7 | 2.9 | 1.27 | noise floor |

**Suno's fingerprint is gone** — 13.39 → 1.35, indistinguishable from native
ACE-Step output and from the control. It cannot survive: 200 Hz structure is
faster than ACE-Step's 25 Hz latent frame, so the encoder discards it and the
decoder stamps its own rate in its place.

But the audio is still obviously machine-made. `frame_confirm` catches both
covers at **24.5× and 24.8×**, on ACE-Step's grid (period 192 = 250 Hz, a
harmonic of 25 Hz).

**The erasure is complete at every fidelity setting.** Covers were generated
across a `cover_noise_strength` gradient — 0.40 (most reinterpreted) to 0.80
(closest to the source). Musical similarity to the original rises with the
setting; the Suno fingerprint does not survive at any of them:

| cover | chroma vs source | Suno 200 Hz fingerprint |
|---|---|---|
| SOURCE | 1.000 | **13.39** |
| noise 0.40 | 0.905 | 1.30 |
| noise 0.55 | 0.960 | 1.27 |
| noise 0.70 | 0.974 | 1.33 |
| noise 0.80 | **0.980** | **1.21** |
| ACE-Step native | — | 1.19 |
| control | — | 1.27 |

At 0.80 the cover follows the original's chords and rhythm almost exactly
(chroma 0.980, onset 0.958) and still carries *none* of Suno's signature. There
is no dose-response because every setting routes the audio through ACE-Step's
25 Hz latent bottleneck: the 200 Hz structure is finer than the latent's time
resolution and cannot pass through, whatever the noise schedule.

Across all three layers, for the covers:

| layer | source | covers |
|---|---|---|
| metadata | `made with suno` | **gone** (ACE-Step writes none) |
| trained Suno classifier | 1.0000 | **0.0000 – 0.0007** |
| Suno 200 Hz fingerprint | 13.39 | **1.21 – 1.33** |
| frame-lock detection | 27.9× | **10.7 – 13.4×** (ACE-Step grid) |

**The distinction that matters: detection survives regeneration, attribution does
not.** A track laundered through a second model still reads as generated, but now
points at the laundering model rather than the original. Any provenance claim of
the form "this came from generator X" is therefore only as good as the assumption
that nothing regenerated it since — and the container metadata, which is the one
layer that would have said `made with suno`, is destroyed by the round trip too
(ACE-Step writes no tags).

### Robustness: no single method survives everything

`frame_detect` against the same degradation set that broke the others:

| treatment | spectral comb | trained classifier | **frame fold** |
|---|---|---|---|
| baseline | ✓ 200 Hz | ✓ 1.0000 | ✓ **32.8×** |
| MP3 64 kbps | ✗ lost rate | ✓ 0.9881 | ✓ **12.2×** |
| Opus 48 kbps | ✗ lost rate | ✓ 0.9976 | ✓ **9.6×** |
| heavy compressor + exciter | ✓ 200 Hz | ✓ 1.0000 | ✓ **29.7×** |
| MP3 32 kbps | ✗ | ✗ 0.0000 | ✗ 2.4× |
| **resample +0.1 %** | ✓ follows | ✗ **0.0006** | ✗ **1.6×** |
| resample +0.5 – 2 % | ✓ exact | ✗ 0.0000 | ✗ 2.1–2.7× |
| time-stretch (`atempo`) | ✓ 200 Hz | ✓ 1.0000 | ✗ 2.0–3.9× |
| vibrato | ✗ | ✗ | ✗ 1.6× |

Folding beats the spectral search on codecs — it holds through 64 kbps MP3 and
Opus, where the comb search loses the spacing entirely. But it fails on anything
that changes the *timing*: a resample takes the period off the integers, and
`atempo`'s phase vocoder re-frames the audio on its own grid, destroying the
lock even though it preserves the spectral comb.

So the three methods are genuinely complementary, and only their union covers
the set. Nothing survives MP3 32 kbps or vibrato.

### Solved: `frame_confirm.py` — comb estimate → integerise → fold

The gaps above are complementary rather than fundamental, and chaining the two
methods closes them. The spectral comb search is the only thing that tracks a
speed change accurately; folding is the only thing sensitive enough to survive
lossy coding. So:

1. take rate hypotheses from **both** sources — the enumerated `sr/hop` list
   *and* the comb estimate (plus its harmonics, since the comb may land on a
   submultiple)
2. **resample the audio so that period becomes an exact integer** — a real
   polyphase resampler preserves the sample-scale artefact, unlike fold-time
   interpolation
3. fold at full sample resolution, and keep whichever hypothesis scores best

| file | rate found | gain | ratio | source | verdict |
|---|---|---|---|---|---|
| Suno baseline | 200.00 | 32.3 | **25.2×** | comb | locked |
| **resample +0.1 %** | 200.20 | 32.3 | **25.3×** | comb | locked |
| **resample +0.5 %** | 201.00 | 32.4 | **32.7×** | comb | locked |
| **MP3 64 kbps** | 200.00 | 8.0 | **9.9×** | list | locked |
| **Opus 48 kbps** | 200.00 | 6.8 | **7.0×** | list | locked |
| ACE-Step generated | 250.00 | 24.1 | **30.1×** | comb | locked |
| AudioSR | 200.00 | 387.8 | **138.4×** | comb | locked |
| synth control (mp3) | — | 1.4 | 1.38× | list | none |
| control + chorus/echo | — | 3.1 | 3.11× | comb | none |
| noise+sweep control | — | 1.2 | 1.16× | comb | none |

Every previously-failing case is recovered, and the `source` column shows the
two halves covering each other exactly as designed: **comb** wins on the
speed-shifted files, **list** wins on the codec-degraded ones. Resampled files
score the same as the undegraded original (32.3 vs 32.3 gain) — the speed change
is fully compensated rather than merely tolerated.

Remaining failures, unchanged: MP3 32 kbps and vibrato. Both destroy the
artefact rather than displace it.

### Negative result: interpolated folding cannot follow a resample

The obvious fix — fold at fractional periods — does not work. Linear
interpolation spreads each sample across two phase bins, low-passing the folded
profile, and the artefact is sample-scale. Measured on an undegraded file:
**32.8× under integer folding, 3.1× through the interpolated path**, i.e. the
method loses files it previously caught. `--drift` exists in the tool but is off
by default and documented as non-functional.

The right approach is inverted: sweep *resampling ratios*, resample the audio
with a real polyphase resampler (which preserves the artefact far better than
fold-time interpolation), and integer-fold each candidate. That is a 1-D sweep,
not a 2-D search, so it is affordable — it just is not built yet.

### An ambiguity that cannot be resolved by folding

The artefact behaves like an impulse train at the frame boundaries, and such a
train folds *equally well at every submultiple period* — each join still lands at
phase zero, with more blocks to average. Attempts to prefer the highest rate, or
the lowest, each mis-identified known cases. So the tool reports the **harmonic
family** rather than guessing a fundamental. Disambiguating needs the spectral
odd/even test (does energy sit on odd multiples?) or knowledge of the
architecture.

## Tested against a second generator: ACE-Step 1.5 (local, MIT, 2 B DiT)

Installed and generated locally on the RTX 3080 alongside ComfyUI. The result
is the most important one in this document, because it is a **complete
provenance blackout** for every method except one.

| layer | Suno track | ACE-Step generated | control |
|---|---|---|---|
| container metadata | `made with suno` ✓ | **none at all** ✗ | none |
| AudioSeal watermark | none | none | none |
| trained Suno classifier | 1.0000 ✓ | **0.0000 / 0.0003 ✗ MISS** | 0.0000 ✓ |
| spectral comb search | 200 Hz, 2.15 ✓ | 50 Hz, 1.28–1.46 (marginal) | 1.20 ✓ |
| **time-domain frame fold** | **11.1×** ✓ | **23.3×** ✓ | **1.00** ✓ |

ACE-Step writes **no metadata whatsoever** — its FLAC output carries not a single
tag, let alone a generator name. Suno's `made with suno` comment is a courtesy,
not an industry norm, and the courtesy is exactly what an open local model
removes.

The state-of-the-art passive classifier, 99.88 % accurate on Suno, calls real
ACE-Step output **"real" with maximum confidence**. That is not a marginal miss;
it is total.

### Correction: the 25 Hz prediction was right, measured in the wrong domain

The VAE-round-trip section below concluded the 25 Hz prediction had failed
because no spectral comb appeared. That conclusion was wrong — or rather, right
about the spectrum and wrong about the artefact.

Folding generated ACE-Step audio at **period 1920 samples = exactly 25 Hz**:

| file | gain | null | ratio |
|---|---|---|---|
| **ACE-Step generated** | **20.72** | 0.89 | **23.3×** |
| Suno | 11.16 | 1.01 | 11.1× |
| control | 1.00 | 0.97 | **1.04** |

The frame lock is there, precisely at the predicted rate, and it is *stronger*
than Suno's. It simply does not manifest as elevated energy on a spectral grid —
the Oobleck decoder's artefact is phase-structured rather than magnitude-
structured, so a magnitude-domain comb search is blind to it while a
time-domain fold sees it plainly.

**This makes the frame-fold statistic the single most valuable instrument found
here:** it is the only method that catches Suno, AudioSR, FlashSR *and*
ACE-Step, needs no training, and returns a clean 1.00 on controls.

### The spectral comb is NOT universal — newer VAEs hide it from the spectrum

A prediction registered in advance, and then falsified. ACE-Step 1.5 uses
diffusers' `AutoencoderOobleck` (the Stable Audio VAE):

```
downsampling_ratios = [2, 4, 4, 6, 10]   product = 1920
sampling_rate       = 48000              -> latent frame rate = 25.00 Hz
```

Confirmed empirically — encoding 30 s produced a latent of 750 frames, exactly
25.00 frames/second. So the prediction was a 25 Hz comb in the decoder's output.

**There is none.**

| file | spacing found | score | z | verdict |
|---|---|---|---|---|
| untouched control | 97.90 Hz | 1.20 | 3.6 | none |
| **ACE-Step VAE round-trip** | 100.00 Hz | **1.18** | 4.4 | **none** |
| Encodec 24 kbps | 120.00 Hz | 1.41 | 6.9 | weak comb |
| AudioSR / FlashSR | 100.00 Hz | 2.5–2.7 | 27–30 | strong comb |
| Suno | 200.00 Hz | 2.15 | 17.6 | strong comb |

The ACE-Step decoder is statistically **indistinguishable from untouched audio**.

This kills the tidy version of the theory. A transposed-convolution decoder does
not automatically stamp a frame-rate comb — it does so only if nothing in the
architecture prevents it. Oobleck and DAC-lineage designs specifically address
upsampling artefacts (snake activations, weight-normalised transposed convs,
anti-aliasing choices) and the artefact this whole method depends on simply is
not there. HiFi-GAN-lineage vocoders (AudioSR, FlashSR) and whatever Suno uses
still have it.

**Consequence:** comb detection — mine *and* the shipped classifier's — is a
generation-of-2023/24 technique. It catches Suno and older HiFi-GAN pipelines
and is blind to a current open-weights model running on a consumer GPU. A clean
comb result means "not one of the vocoders that leak", not "not generated".

Caveat on scope: this tested the VAE decoder round-tripping real audio. A latent
produced by the diffusion model, rather than by encoding real audio, could in
principle behave differently — though the decoder, which is what stamps the
artefact, is identical in both paths.

### Sensitivity limits

The physical detector is less sensitive than the trained model on degraded
material. Suno through 96 kbps MP3 (174.55 Hz, score 1.38), through Apollo
(100 Hz, 1.37) and through DeepFilterNet (correct 200.00 Hz but score 1.41) all
fall below threshold, while the trained classifier still called every one of them
at p ≈ 1.0. 3585 learned weights beat one physical statistic when the signal is
weak — the trade is generality for sensitivity.

### Why Encodec doesn't trigger it

Round-tripping the non-AI control through Encodec — a neural codec whose decoder
is a transposed-conv vocoder — leaves the verdict at 0.0000 at every bitrate
(24, 12, 6, 3 kbps). Its comb peak sits at 202.75 Hz, i.e. nowhere, and its
on-grid ratio at 200 Hz is 0.929.

So **this classifier is generator-specific, not vocoder-generic.** It is tuned to
one frame rate. A generator with a different hop — or one that dithers its frame
rate — would pass. Do not treat a clean result from it as evidence of human
origin.

### Attempts to manufacture a false positive — all failed

To test whether the detector simply flags ordinary produced music, non-AI
signals were built and pushed through a full production chain. `make_music_control.py`
synthesises a mix from scratch (harmonic saw chords, bass, kick, snare, hats,
convolution reverb) — no neural vocoder anywhere in its lineage.

| signal | p(AI) |
|---|---|
| sine sweep + white noise | 0.0000 |
| synth mix, raw | 0.0000 |
| + bus compression, limited to −14 LUFS | 0.0000 |
| + 320 kbps MP3 | 0.0000 |
| + chorus/echo spectral smearing | 0.0000 |
| + smearing then 320 kbps MP3 | 0.0000 |
| 64 kbps MP3 of the noise control | 0.0000 |
| after AudioSR regenerating 11–24 kHz | 0.0000 |

**Eight constructed non-AI signals, zero false positives.** Loudness
maximisation, lossy encoding, spectral smearing and neural bandwidth extension
all failed to trigger it. Two plausible hypotheses died along the way — "it
flags dense/limited production" (the synth control has a *lower* HF/LF ratio
than the flagged track and still scores clean) and "it flags spectral
smoothness" (smearing raised the residual rather than lowering it).

In hindsight these controls were never going to break it: the container later
confirmed the flagged track really was Suno-generated. The exercise is still
worth keeping — it establishes that ordinary mastering, limiting, lossy
encoding and neural bandwidth extension do **not** trigger this detector, which
is exactly what you need to know before trusting it on your own work.

### Where in the spectrum does the signature live? (`band_probe.py`, `band_swap.py`)

Using the now-confirmed Suno track as a known positive and the synthesised mix
as a known negative, both at 48 kHz stereo.

**Don't isolate bands into silence.** The obvious experiment — keep one band,
zero the rest — is confounded: an all-zero feature vector scores **0.993 "AI"**
from the +4.98 bias alone, so any narrow band collapses toward "AI" whatever it
contains. In that test 5–8 kHz read ~0.99 for *both* files, which looks like
"this band is decisive" and actually means "85 % of the vector is zeros".

**Ablation — no band is necessary.** Zero any single 1 kHz band and the verdict
is unchanged: the AI track stays 1.0000, the control stays 0.0000, for every
band from 0–1 k to 8–24 k.

**Swap — no band is sufficient.** Keeping both spectra fully populated and
exchanging one band between the files:

| band swapped | graft control ← AI | heal AI ← control |
|---|---|---|
| every 1 kHz band, 0–24 k | 0.0000 | 0.9998 – 1.0000 |

Not one 1 kHz slice moves the verdict in either direction.

**Cumulative swap — the boundary is ~1–4 kHz.** Exchanging a widening block:

| block swapped | graft control ← AI | heal AI ← control |
|---|---|---|
| 1–2 kHz | 0.0000 | 0.9998 |
| 1–3 kHz | 0.0010 | 0.9661 |
| **1–4 kHz** | **0.4986** | **0.0792** |
| 1–5 kHz | 0.9973 | 0.0002 |
| 1–6 kHz and wider | ≥0.9996 | 0.0000 |

Both directions cross at the same place. **The signature is distributed, not
localised**: roughly 3 kHz of contiguous spectrum has to change hands before the
verdict moves, and 1–5 kHz decides it outright. There is no single "tell"
frequency, no comb line, nothing that could be notched out — the evidence is a
broadband texture, redundantly present across the whole range.

That redundancy is precisely why the classifier is hard to fool incidentally,
and why the eight synthetic controls above never tripped it.

### Is the signature hiding in the noise floor? (`denoise_probe.py`)

If the artifact were an additive noise-like pattern, denoising should strip it.
Tested with FFT spectral subtraction, wavelet shrinkage, non-local means,
spectral gating, and a neural denoiser. SNR is measured after delay alignment
(FFT and network denoisers add latency — comparing sample-aligned makes a gentle
filter look like total destruction).

| denoiser | AI p | SNR dB | HF Δ dB | control p |
|---|---|---|---|---|
| afftdn nr=6 (light) | 1.0000 | 34.2 | −0.3 | 0.0000 |
| afftdn nr=20 | 1.0000 | 21.2 | −2.6 | 0.0000 |
| afftdn nr=40 (heavy) | 0.9999 | 14.2 | −13.5 | 0.0001 |
| **afftdn nr=97 (extreme)** | **0.7374** | 14.2 | −13.6 | **0.6075** |
| afwtdn wavelet | 0.9999 | 16.3 | −6.3 | 0.0000 |
| afwtdn sigma=0.2 | 1.0000 | 10.8 | −10.6 | 0.0000 |
| noisereduce 0.5 | 1.0000 | 19.0 | +0.2 | 0.0000 |
| noisereduce 1.0 | 0.9920 | 1.8 | +2.1 | 0.0000 |
| **DeepFilterNet 3 (neural)** | **1.0000** | **0.7** | +3.2 | 0.0076 |

**No denoiser removed the signature.** The most striking case is DeepFilterNet:
a neural denoiser that reduced correlation with the input to 0.7 dB SNR — the
output barely resembles the source — and the verdict stayed at *exactly* 1.0000.
(It applies complex filter coefficients to the STFT rather than resynthesising
through a vocoder, so the underlying spectral structure survives.)

**The one case that moved it moved BOTH files the same way.** At `afftdn nr=97`
the AI track fell to 0.737 while the control *rose* to 0.608 — both converging
toward the 0.993 empty-vector score. Extreme denoising does not selectively
strip an AI signature; it flattens the spectral residual for everything, the
feature vector approaches zero, and the +4.98 bias takes over.

**Practical consequence, and it is the opposite of the intuition:** aggressive
denoising causes **false positives on real audio before it causes false
negatives on generated audio**. Light-to-moderate noise reduction is safe to
apply to your own work — it does not move the verdict. Only settings extreme
enough to gut the audio (−13 dB of 4–8 kHz energy) shift anything, and they push
*real* material toward "AI".

This is consistent with the band-swap result: the signature is a broadband
texture spread redundantly across 1–5 kHz, not a discrete additive noise pattern
sitting on top of the music. There is nothing localized to subtract.

*(`anlmdn` segfaults on this ffmpeg build — exit `0xC0000005`, both stereo and
mono. A build defect, not a result.)*

### Implementation quirk: the resampler leaks ultrasonics into the decision

The detector nominally reads only 1–8 kHz of a 16 kHz downmix, so content above
8 kHz should be irrelevant. It isn't. Feeding it a signal masked to contain
*only* 8–24 kHz (verified: 2.6e-16 of its energy below 8 kHz):

- `torchaudio.transforms.Resample(48000, 16000)` at default settings
  (`lowpass_filter_width=6`) puts **99.6 % of its output energy into the
  1–8 kHz analysis band**, at about −16 dB relative to the input.

So ultrasonic content aliases straight into the features. This affects the
reference implementation too — it uses the same resampler — so it is a property
of the deployed detector, not of this re-implementation. Practical consequence:
a file's content above 8 kHz *can* influence its verdict, and swapping
resamplers (soxr, librosa, ffmpeg) will shift scores.

### Ordering: always check the container first

1. **`wav_chunks.py` / container tags** — seconds, and frequently decisive.
2. **Watermark detectors** — cheap, and a hit is strong positive evidence.
3. **Passive classifier** — the only layer that works on a stripped file.
4. **Signal forensics** — last, and only to explain a result you already have.

Running that order backwards costs hours. It did here.

### Forensic profile of the flagged file — no tampering found

| measure | value | reading |
|---|---|---|
| integrated loudness | −14.24 LUFS | exactly the streaming normalisation target |
| sample peak | −4.09 dBFS | ample headroom |
| crest factor | 12.25 dB | **not** heavily limited (<10 would be) |
| short-term RMS spread | 11.41 dB | good dynamics retained |
| samples at ceiling | 0 | no clipping whatsoever |
| deepest narrow notch | 6.0 dB (0 bins >9 dB) | no notch-carving watermark |
| L/R correlation | 0.844 | natural stereo, no HF collapse |
| codec cliff | 20010 Hz, −24.4 dB | one ordinary lossy encode |

An envelope periodicity at 2.34 Hz initially read as "pumping" — it is the
song's tempo (~140 BPM). Watch for that class of self-inflicted false alarm.

**Conclusion: nothing was done to degrade this audio.** It is a clean,
well-mastered, streaming-normalised file that has been through a single lossy
encode. The AI flag is not a damage signature — there is no damage.

### The overlooked explanation: neural tools elsewhere in the chain

AI involvement is not binary, and the flag does not mean "generated by Suno".
Any **neural vocoder anywhere in the production chain** can stamp resynthesis
artifacts:

- AI mastering services (LANDR and similar)
- stem separation / remixing (Demucs, Spleeter, MDX) — resynthesises audio
- vocal isolation or de-bleed
- neural noise reduction, de-reverb, de-clip (iZotope's ML modules)
- AI upscaling or "enhancement" applied by a distributor or plugin

Notably, this repo's own bandwidth-extension tools did **not** trigger the
detector — but they only regenerate a narrow high band. A full stem-separate →
resynthesise → remix pass rewrites the entire signal and is a far likelier
source of vocoder artifacts. If a flagged track is human-composed and
human-performed, this is the first place to look.

### If genuinely human-made work gets flagged

Do not try to tune the audio around the detector. It is a black box you cannot
validate, different detectors key on different features, and degrading a master
to satisfy one classifier trades real quality for an unverifiable gain. Add
provenance instead of subtracting signal:

- **Keep and archive the evidence** — project files, stems, dated exports,
  session history. This is what actually settles disputes.
- **Sign masters with C2PA Content Credentials.** Cryptographically bound, and
  it is the positive-evidence channel that multi-standard detection layers
  cross-reference.
- **Use DDEX disclosure fields** at distribution to declare AI involvement (or
  its absence) in the metadata distributors actually read.
- **Calibrate first.** Run a batch of known-human tracks through
  `ai_audio_forensics.py`. If they all flag, the detector is mis-calibrated for
  your material and that measurement is itself the rebuttal.

---

## Measured: does an AudioSeal watermark survive processing?

Embedded at 26.2 dB SNR, then transformed:

| transform | detector p | 16-bit message |
|---|---|---|
| none (reference) | 1.0000 | 2 bit errors |
| MP3 128 kbps round-trip | 1.0000 | 2 bit errors |
| MP3 64 kbps round-trip | 1.0000 | 1 bit error |
| resample 16 → 48 kHz | 1.0000 | 2 bit errors |
| lowpass 6 kHz | 1.0000 | **exact** |
| gain −6 dB | 1.0000 | **exact** |
| +40 dB SNR noise | 1.0000 | 2 bit errors |

**Detection is far more robust than the payload.** Every transform kept
detection pinned at p = 1.0000, while the 16-bit message picked up 1–3 bit
errors even on the pristine file. If you need the payload to survive, use error
correction; if you only need "is this watermarked", it holds up well.

Practical implication for this repo: the restoration pipeline (AudioSR / Apollo /
FlashSR) is *not* a provenance shredder for incidental processing — but any
pipeline that resamples and re-encodes should be assumed to weaken payload bits.

---

## Sources

- [AudioSeal](https://github.com/facebookresearch/audioseal) (Meta, MIT)
- [lofcz/ai-music-detector](https://github.com/lofcz/ai-music-detector) (MIT)
- [A Fourier Explanation of AI-Music Artifacts](https://transactions.ismir.net/articles/10.5334/tismir.254) (ISMIR / TISMIR)
- [SynthID](https://deepmind.google/models/synthid/) (Google DeepMind)
- [C2PA](https://c2pa.org/faqs/) · [C2PA Explainer](https://spec.c2pa.org/specifications/specifications/2.4/explainer/Explainer.html)
- [Watermarking for AI Content Detection: a review](https://arxiv.org/pdf/2504.03765)
- [SoK: Watermarking for AI-Generated Content](https://arxiv.org/pdf/2411.18479)
- [FlashSR / audio SR context](https://arxiv.org/pdf/2501.10807)
- [OpenAI on content provenance](https://openai.com/index/advancing-content-provenance/)
