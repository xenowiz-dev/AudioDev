# Installing AudioDev Studio on a new box

```powershell
git clone git@github.com:xenowiz-dev/AudioDev.git   # anywhere; no B:\ needed
cd AudioDev
.\install.ps1 -WhatIf     # say what would happen, change nothing
.\install.ps1             # prereqs, clones, nine venvs, then the models
```

Everything is **resumable and idempotent** — an existing venv or a present
model is skipped, so a run that fails halfway is re-run, not unpicked.

| stage | what it does |
|---|---|
| `-Stage prereq` | GPU, both Pythons, ffmpeg/ffprobe/git, disk space |
| `-Stage clones` | clones the four upstream repos at the exact commits this was built on |
| `-Stage venvs` | creates the nine environments from frozen package lists |
| `-Stage models` | downloads weights, skipping anything the card cannot run |
| `-Stage verify` | imports torch in each venv and lists the DiT variants found |

## Where the tree lives is discovered, not assumed

Nothing in the code knows about `B:\AudioDev` any more. Every launcher
(`web\serve.ps1`, `web\restart_studio.ps1`, `studio\studio.ps1`,
`train\train_tonight.ps1`) sets `AUDIODEV_ROOT` from its own location before
starting Python, and every Python module falls back to its own `__file__` when
the variable is absent. The two agree by construction, so the repo runs from
any folder on any drive. Set `AUDIODEV_ROOT` yourself only to point a script at
a *different* checkout.

## What is NOT in the repo

The repository is the code, the frozen package lists, the manifest and the
docs — about 1.6 MB. Everything below is rebuilt by the installer or has to be
copied by hand.

| path | how it gets there | size |
|---|---|---|
| `*/.venv`, `train/.venv-label` | `-Stage venvs` | ~30 GB |
| `acestep/ACE-Step-1.5`, `msst/MSST`, `flashsr/FlashSR_Inference`, `watermark/ai-music-detector` | `-Stage clones` | ~0.5 GB |
| `acestep/checkpoints`, `minimax/hf`, `msst/checkpoints`, FlashSR `ModelWeights` | `-Stage models` | 74–128 GB |
| `Music/studio` — the library, sidecars, `settings.json` | **copy by hand** if you want the library on the new box | varies |
| `loras/` and `train/lora_out/` — every adapter trained here | **copy by hand**; 400+ GPU-minutes to recreate | ~1 GB |
| `train/dataset.json`, `train/preprocessed/` | `train\build_dataset.py` regenerates from the Suno folder | 0.9 GB |
| `train/labels.json` | committed — 40 CPU-minutes of bpm/key/meter labelling, keyed by filename | 86 KB |
| the Suno export folder (audio + `.txt` sidecars) | **copy by hand**, then set `SUNO_DIR` to it before running `label_audio.py` / `_audit_dedup.py`; `build_dataset.py` takes it as `--src` | ~1 GB |

GitHub access on the new box: `gh auth login` first (the clone URL above is
SSH), or clone over HTTPS with `https://github.com/xenowiz-dev/AudioDev.git`.

The Apollo *universal* checkpoint is fetched from the Hugging Face Space that
redistributes Lew's weights; it is the same size as the copy this box was built
with but its bytes were not compared. If a restore looks different on the new
machine, copy `msst\checkpoints\apollo_universal_sr.ckpt` across by hand.

## The two things that make this non-trivial

**Three different torch builds coexist, on purpose.**

| venv | Python | torch | why |
|---|---|---|---|
| studio | 3.12 | none | the web server; pure python so it starts instantly |
| minimax, acestep, lyrics | 3.12 | **cu128** | the generators and whisper |
| watermark | 3.12 | **cpu** | DSP only — a CUDA build here would just waste 3 GB |
| msst, audiosr, flashsr | **3.10** | **cu121** | the upscalers do not run on 3.12 |

The installer puts torch in **first**, from its own index URL. Installed as a
transitive dependency it silently resolves to the default CUDA build, and the
box then looks fine until a model loads.

**The package lists are frozen from a working machine** — `install/requirements/*.txt`,
captured with `pip freeze`, not hand-written. Regenerate them after changing an
environment:

```powershell
foreach ($v in "studio","minimax","acestep","lyrics","watermark","flashsr","msst","audiosr") {
  & ".\$v\.venv\Scripts\python.exe" -m pip freeze --local > ".\install\requirements\$v.txt"
}
& ".\train\.venv-label\Scripts\python.exe" -m pip freeze --local > ".\install\requirements\label.txt"
```

## Models follow the card

`install/models.json` records what each model needs to **run**, and the
installer hides anything that will not fit. Nothing to edit when the GPU
changes:

| card | models offered | download |
|---|---|---|
| 10 GB | 8 | ~88 GB |
| 16 GB+ | 10 (adds XL turbo and XL base) | ~128 GB |

`-All` includes the optional tier; `-SkipMiniMax` drops the largest single
download (54 GB) if you only want ACE-Step.

## Which DiT variant

Discovered from `acestep/checkpoints/` at runtime — dropping a checkpoint in is
all it takes to offer it.

| variant | size | VRAM | notes |
|---|---|---|---|
| `acestep-v15-turbo` | 4.5 GB | ~7.0 GB | 8 steps, no CFG. The default |
| `acestep-v15-base` | 4.5 GB | ~7.3 GB | **the only 2B variant that can continue a track** (`complete`), pull stems (`extract`), and honour `guidance_scale`. CFG plus 32–100 steps, so several times slower |
| `acestep-v15-xl-turbo` | 20 GB | ~11.5 GB | 4B decoder. Needs a 16 GB card |
| `acestep-v15-xl-base` | 20 GB | ~12.0 GB | 4B plus continuation. Slowest of the set |

VRAM figures are ACE-Step's own profiling (`gpu_config.py`), not estimates:
DiT weights + VAE 0.33 + text encoder 1.2 + CUDA context 0.5 + per-batch
activations.

**A LoRA only matches the variant it was trained on.** PEFT matches modules by
name, not by shape provenance, so a turbo-trained adapter will load against XL
and produce noise. The studio compares the adapter's recorded base against the
loaded variant and says so.

## After installing

```powershell
.\web\serve.ps1                 # http://127.0.0.1:7862/
.\web\serve.ps1 -Tailscale      # also publish it on the tailnet
```

`web\restart_studio.ps1` restarts the server and **refuses while a job is
running** — it kills the workers alongside uvicorn, which a naive `Stop-Process`
on uvicorn alone does not, leaving an orphan holding the GPU.
