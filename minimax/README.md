# MiniMax Music 3 on a single 10 GB GPU

Lyrics- and caption-conditioned music generation, up to ~5 minutes, **44.1 kHz
stereo**. Installed 2026-08-13, the day the weights went up.

```powershell
# is there room to run right now? (loads, reports footprints, generates nothing)
B:\AudioDev\minimax\.venv\Scripts\python.exe B:\AudioDev\minimax\gen.py --dry-run

# 30 s clip
B:\AudioDev\minimax\.venv\Scripts\python.exe B:\AudioDev\minimax\gen.py `
    --duration 30 --prompt-file caption.txt --lyrics-file song.txt -o out.wav
```

Measured on the RTX 3080 (10 GB, ~1 GB held by the desktop): **20 s of audio in
142 s — 7.1x realtime, peak 7.89 GB VRAM.**

> **Needs ~8 GB of VRAM free.** The LLM and the RVQ depth decoder must be
> resident together (7.49 GB) for the autoregressive stage. With ComfyUI's
> models loaded (~3.5 GB) there is not enough room — wait for it to idle-unload.
> `--dry-run` checks this and says so before anything is generated.

---

## The two things that make it fit

The upstream GitHub repo only ships an SGLang-Omni server and states plainly
that **inference requires two CUDA GPUs**. That path is unusable here. The
diffusers `MiniMaxMusic3ModularPipeline` is the one that works, with two
departures from the model card.

**1. Never `pipe.to("cuda")`.** The model card's example ends with it. It pushes
every component at the card at once and OOMs instantly. Use
`ComponentsManager.enable_auto_cpu_offload()` instead: weights live in system
RAM and each component is pulled onto the GPU only for its own forward pass.
It sizes decisions from `mem_get_info`, i.e. genuinely free VRAM, so a ComfyUI
instance holding a few GB is respected rather than fought over.

**2. The 8B LLM must be quantized.** It is 15.99 GB in bf16 and will not fit in
10 GB however much is evicted around it. `--quant 4bit` (NF4, double quant)
takes it to 6.29 GB. `load_components` routes a dict kwarg to one component, so
only the LLM is touched — the flow-matching transformer and the vocoder stay at
full bf16:

```python
load_components(dtype=torch.bfloat16,
                quantization_config={"language_model": bnb_4bit_config})
```

8-bit is not an option: ~8.9 GB for the LLM plus the 1.2 GB decoder it must sit
beside exceeds the card.

### Component footprints

| component | params | bf16 | 4-bit |
|---|---:|---:|---:|
| language_model (Qwen3-8B) | 8.58 B | 15.99 GB | **6.29 GB** |
| transformer (flow matching) | 2.43 B | 4.53 GB | 4.53 GB |
| rvq_depth_decoder | 0.65 B | 1.20 GB | 1.20 GB |
| vocoder | 54 M | 0.10 GB | 0.10 GB |
| condition_encoder | 25 M | 0.05 GB | 0.05 GB |
| **total** | | **21.87 GB** | **12.17 GB** |

---

## Gotchas hit during install

**Only ~27 GB of the 57 GB repo is needed.** `modular_model_index.json` names
exactly seven subfolders. `qwen_7B/` (18.5 GB), `flowmatching_vae.pth` (9.8 GB)
and `dav.pth` are the SGLang path. `download.ps1` pulls only what diffusers
resolves.

**Do not set `HF_HUB_OFFLINE=1`.** diffusers resolves a *sharded* checkpoint via
`_get_checkpoint_shard_files`, which calls the live `model_info()` API rather
than reading the local index. Offline mode therefore breaks the 9.3 GB
flow-matching transformer while every unsharded component loads fine — and
`load_components()` reports that failure as a *log warning*, leaving
`pipe.transformer = None` so the pipeline looks loaded and dies much later.
`gen.py` asserts the full component roster after loading for this reason.
Weights still come from cache; only the shard listing needs the network.

**The AR stage requires two models co-resident.**
`MiniMaxMusic3SemanticGenerationStep` drives the LLM and the RVQ depth decoder
on every frame and raises if they land on different devices. The stock
`AutoOffloadStrategy` knows nothing about that and evicts the 6.29 GB LLM to
place the 1.2 GB decoder. `gen.py` passes an `ARPairOffloadStrategy` that
refuses to evict one to place the other.

**bitsandbytes leaves ~1.9 GB of scratch in torch's caching allocator.** This is
what actually caused the eviction above: after moving the 4-bit LLM onto the
card, `mem_get_info` reported **0.63 GB free when ~2.5 GB was available**, so
every subsequent placement looked unaffordable. The offload strategy calls
`torch.cuda.empty_cache()` before measuring.

**`--reserve` defaults to 1 GB, not 2.** With a 2 GB margin the offloader
concluded the 1.2 GB decoder was unaffordable next to the LLM and evicted it.
`ARPairOffloadStrategy` now vetoes that eviction outright, so reserve mainly
governs placement of the non-AR components; 1 GB is the verified setting.

**PowerShell array splat silently truncates `hf download --include`.**
`--include @patterns` expands to `--include p1 p2 p3`, where `--include` takes
only `p1` and the rest become positional FILENAMES — dropping the first pattern
from the pull. Pass a separate `--include` per pattern.

---

## Prompting

The model was trained on long sectioned captions, not keyword lists. The repo's
own test script uses three headings, and short prompts give up most of the
arrangement control:

```
Global Metadata
Basic Attributes: bpm is 92. key is E, and scale is minor. Electric Blues.
Global Emotional Progression: ...
Sonics & Production Profile: ...
Vocal Details
Vocal Gender & Timbre: Singer A (Male). Deep, gravelly baritone.
Vocal Style: ...
Arrangement
Primary: ...
Secondary: ...
```

Lyrics take `[verse]` / `[chorus]` / `[bridge]` structure tags, each on its own
line. `audio_duration` is an upper bound — the LM may stop earlier. The AR stage
runs at 25 Hz and caps at 9000 frames (360 s).

---

## Corrections to the model card

- **Output is 44.1 kHz, not 32 kHz.** `vocoder/config.json` says
  `sampling_rate: 44100`, and the pipeline reads its rate from there.
- **"8 GB VRAM minimum with CPU offloading"** is only reachable with the LLM
  quantized. Nothing in the card or repo says so.

## Quality caveat

The 20 s smoke test is real full-band music (content to 22 kHz, no codec cliff),
but its balance is bass-heavy versus a finished master — 70.6% of energy below
500 Hz and 1.5% in 4–12 kHz, at 19 dB crest factor. Output is unmastered, which
explains much of that, but **the effect of 4-bit quantization on quality has not
been isolated**: an unquantized A/B needs more VRAM than this card has.

## Detection note

Relevant to `..\watermark\`: the smoke test shows **no spectral comb** (ratio
1.07, verdict none) but a strong **time-domain frame lock at 344.53 Hz**
(= 44100/128, fold ratio 6.42). That is the same split ACE-Step shows — flow
matching plus a Flow-VAE decoder does not stamp the spectral comb that
HiFi-GAN-style vocoders do, but frame-locked structure is still there. The
vocoder's `upsampling_ratios` multiply to 512, giving a nominal 86.13 Hz frame
rate; 344.53 Hz is 4x that, and the exact attribution is not pinned down.
