"""ACE-Step 1.5 worker. Runs inside B:\\AudioDev\\acestep\\.venv.

Handles both tasks the installed DiT-only setup supports:

  text2music  caption (+ optional lyrics) -> new track
  cover       regenerate an existing track in a new style

The handler is initialised once and kept warm; only the first request pays it.

The three settings below are not defaults and are not guessable -- each one was
established the hard way, and `..\\..\\acestep\\cover.py` carries the same values:

  * `cover_noise_strength` defaults to **0.0 in the library, which means pure
    noise** -- the source is ignored entirely and you get an unrelated track
    with no resemblance to the input. ~0.75 is what makes a cover a cover.
  * `reference_audio` is a separate STYLE reference. Passing the source as both
    it and `src_audio` muddies the conditioning, so it is left unset.
  * empty lyrics let the model sing freely; "[Instrumental]" forces an
    instrumental. Defaulting to the latter silently produces vocal-less covers.
"""

import glob
import json
import os
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, "..", ".."))

os.environ.setdefault("ACESTEP_INIT_LLM", "false")   # nano-vllm has no Win wheel
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROTO = sys.stdout
sys.stdout = sys.stderr
PROTO.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.join(ROOT, "acestep")

# Which DiT checkpoint to load. `config_path=None` makes ACE-Step auto-select,
# which resolves to turbo; naming it explicitly makes the choice visible and
# switchable. The supervisor restarts this process when it changes, because the
# checkpoint is chosen at load and cannot be swapped afterwards.
#
#   acestep-v15-turbo              2B, 8 steps, no CFG      ~7.0 GB
#   acestep-v15-turbo-continuous   2B, 8 steps; continuous latents, NOT
#                                  song continuation despite the name
#   acestep-v15-base               2B, CFG + 32-100 steps   ~7.3 GB, and the
#                                  only 2B variant with `complete` (continue),
#                                  `extract` and guidance_scale
#   acestep-v15-xl-*               4B                       ~12 GB
VARIANT = os.environ.get("ACESTEP_VARIANT") or "acestep-v15-turbo"
OUTDIR = os.path.join(ROOT, "Music", "studio")


def emit(obj):
    PROTO.write(json.dumps(obj) + "\n")
    PROTO.flush()


# ------------------------------------------------------------------- LoRA
#
# The handler already implements all of this (add_lora / remove_lora /
# set_use_lora / set_lora_scale); the worker's job is only to hold the state
# machine so a request says what it wants and the adapter matches.
#
# Two things about that API decide the shape of the code below:
#
#   * every call returns a STRING beginning with a tick or a cross. There are
#     no exceptions to catch, so an unchecked call fails by generating with
#     the wrong model while the UI reports success -- the worst outcome this
#     feature has. Every return value is checked.
#   * `_default_adapter_name_from_path` names an adapter after its directory's
#     basename, so `train\lora_out\final` and `loras\x\final` would both be
#     "final". The caller passes an explicit name to make that impossible.

_LORA = {"path": None, "name": None, "scale": None}

# ---------------------------------------------------------------- thinking
#
# ACE-Step's 5Hz LM does chain-of-thought over the prompt before the DiT runs:
# it invents music metadata (bpm, key, time signature), can rewrite the caption
# and detect the vocal language, and it is the only thing that makes
# `lm_temperature` -- the closest either backend has to a real weirdness dial --
# mean anything.
#
# It never ran here for a reason that had nothing to do with the missing
# nano-vllm wheel: `generate_music(dit_handler, llm_handler, ...)` takes the LM
# as its SECOND argument and this worker passed None, so the gate
# (`llm_handler is not None and llm_handler.llm_initialized`) could never open
# whatever `thinking` was set to.
#
# Measured on this box: `backend="pt"` loads the 1.7B checkpoint in 23 s and
# costs ~6.8 GB of SYSTEM RAM, not VRAM, because offload_to_cpu keeps it off
# the card between uses. That is the right trade on 10 GB.
#
# Loaded lazily on the first request that asks for it: a worker that never
# thinks should never pay the 23 s or the RAM.
_LM = {"handler": None}
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B")


def lm_get(progress_fn=None):
    """The LLMHandler, loading it on first use. None if it will not load."""
    if _LM["handler"] is not None:
        return _LM["handler"]
    from acestep.llm_inference import LLMHandler
    if progress_fn:
        progress_fn(f"loading the 5Hz LM ({LM_MODEL}) for thinking mode "
                    f"— about 20 s, once")
    lm = LLMHandler()
    msg, ok = lm.initialize(
        checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
        lm_model_path=LM_MODEL,
        backend="pt",            # vllm has no Windows wheel; pt is supported
        device="cuda",
        offload_to_cpu=True)     # RAM between uses, card only while thinking
    if not ok:
        raise RuntimeError(f"thinking mode unavailable: {str(msg)[:200]}")
    _LM["handler"] = lm
    return lm


def lora_apply(handler, path, name, scale):
    """Make the loaded adapter match the request. Raises on failure.

    Returns a short line for the log, or None when nothing changed.
    """
    want = os.path.normcase(os.path.abspath(path)) if path else None
    have = os.path.normcase(os.path.abspath(_LORA["path"])) if _LORA["path"] else None

    if want != have:
        if have is not None:
            # Unreachable by design: the supervisor restarts this process for
            # any adapter change, because ACE-Step cannot take one back off.
            # `remove_lora` -> PEFT `delete_adapter` raises a bare
            # KeyError('<adapter>') for the adapter it was just asked about,
            # measured on peft 0.20.0, and a half-removed adapter would quietly
            # colour every later track. If this ever fires, the supervisor's
            # restart rule has regressed -- say so rather than guess.
            raise RuntimeError(
                "this worker already has a LoRA loaded and ACE-Step cannot "
                "unload one; the supervisor should have restarted it")
        if want is not None:
            msg = handler.add_lora(path, adapter_name=name)
            if not msg.startswith("✅"):
                raise RuntimeError(f"LoRA failed to load: {msg}")
            _LORA.update(path=path, name=name, scale=None)
            msg = handler.set_use_lora(True)
            if not msg.startswith("✅"):
                raise RuntimeError(f"LoRA loaded but not enabled: {msg}")
        changed = True
    else:
        changed = False

    if want is not None and scale is not None and scale != _LORA["scale"]:
        msg = handler.set_lora_scale(name, float(scale))
        if not msg.startswith("✅"):
            raise RuntimeError(f"could not set the LoRA scale: {msg}")
        _LORA["scale"] = float(scale)
        changed = True

    if not changed:
        return None
    if want is None:
        return "LoRA off (base model)"
    return f"LoRA {os.path.basename(path)} at scale {_LORA['scale']}"


def progress(msg, stage=None, frac=None):
    # ACE-Step gets coarse stage messages only, deliberately: its inference has
    # no tqdm or callback to hook, and a whole generation is ~10 s, so
    # instrumenting its internals could not repay the cost.
    m = {"type": "progress", "msg": msg}
    if stage:
        m["stage"] = stage
    if frac is not None:
        m["frac"] = frac
    emit(m)


def main():
    progress("importing ACE-Step")
    import torch
    from acestep.handler import AceStepHandler
    from acestep.inference import (GenerationParams, GenerationConfig,
                                   generate_music)

    progress(f"initialising handler ({VARIANT})")
    t0 = time.time()
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=PROJECT_ROOT, config_path=VARIANT, device="cuda",
        use_flash_attention=False, compile_model=False,
        offload_to_cpu=True,          # 10 GB card, shared with other work
        # gen.py/cover.py hardcode False, which is ACE-Step's own 12-16 GB
        # setting (gpu_config.py: "12-16GB can keep DiT on GPU"). For a 10 GB
        # tier its config defaults to True, and here it matters more: with the
        # DiT resident an *idle* warm worker sat on 4.6 GB, which is precisely
        # the VRAM this UI exists to not squat on.
        offload_dit_to_cpu=True, quantization=None)
    if not ok:
        emit({"type": "error", "msg": f"ACE-Step init failed: {status}"})
        return
    torch.cuda.empty_cache()
    progress(f"ready in {time.time() - t0:.0f}s")
    emit({"type": "ready", "model": "acestep", "variant": VARIANT})

    os.makedirs(OUTDIR, exist_ok=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("type") == "shutdown":
            break
        if req.get("type") != "generate":
            continue

        try:
            # Absent key == no LoRA == exactly today's behaviour, which is what
            # keeps the gradio app on :7861 working unchanged.
            line = lora_apply(handler, req.get("lora_path") or None,
                              req.get("lora_name") or "adapter",
                              req.get("lora_scale"))
            if line:
                progress(line)

            cover = bool(req.get("src_audio"))
            lyrics = req.get("lyrics") or ""
            instrumental = bool(req.get("instrumental"))
            if instrumental:
                lyrics = "[Instrumental]"

            thinking = bool(req.get("thinking"))
            kw = dict(
                caption=req["prompt"],
                lyrics=lyrics,
                instrumental=instrumental,
                inference_steps=int(req.get("steps", 8)),
                seed=int(req.get("seed", 42)),
                thinking=thinking,
            )
            if thinking:
                # Only meaningful with the LM attached; harmless otherwise, but
                # sending them unconditionally would imply a control that is
                # not connected.
                kw.update(lm_temperature=float(req.get("lm_temperature", 1.0)),
                          use_cot_metas=bool(req.get("cot_metas", True)),
                          use_cot_caption=bool(req.get("cot_caption", False)),
                          use_cot_language=bool(req.get("cot_language", True)))
            if cover:
                kw.update(
                    task_type="cover",
                    src_audio=req["src_audio"],
                    audio_cover_strength=float(req.get("cover_strength", 1.0)),
                    cover_noise_strength=float(req.get("noise_strength", 0.75)),
                )
                # duration comes from the source track; passing one is ignored
            else:
                kw.update(task_type="text2music",
                          duration=float(req.get("duration", 30.0)))

            params = GenerationParams(**kw)

            # Outputs are named by UUID, so take whatever is new in the folder.
            before = set(glob.glob(os.path.join(OUTDIR, "*.flac"))
                         + glob.glob(os.path.join(OUTDIR, "*.wav")))
            progress("generating (cover)" if cover else "generating")
            torch.cuda.reset_peak_memory_stats(0)
            t0 = time.time()
            # GenerationConfig() defaults are wrong for this app, twice over:
            #
            #   use_random_seed=True   `prepare_seeds` then IGNORES the seed
            #                          entirely and rolls a fresh random one per
            #                          item. Every render here used a random
            #                          seed no matter what the UI sent, which
            #                          also made ACE-Step look non-deterministic
            #                          when it was only ever un-seeded.
            #   batch_size=2           two takes are rendered per request and
            #                          this worker reports `files[-1]`; the
            #                          sibling was orphaned in the library with
            #                          no sidecar, one per generation forever.
            #
            # `seeds` is passed as well as `params.seed` because inference.py
            # prefers config.seeds and only falls back to the param.
            # Takes: 1 by default here, but the knob is real. ACE-Step gives
            # every extra take its own seed (prepare_seeds randomises items
            # after the first), so N takes are N genuinely different songs from
            # one prompt rather than N copies -- which is how Suno works and
            # how this library was built. Cost is close to linear in VRAM and
            # time, so it stays opt-in on a 10 GB card and becomes cheap on a
            # bigger one.
            takes = max(1, min(8, int(req.get("takes", 1) or 1)))
            cfg = GenerationConfig(batch_size=takes, use_random_seed=False,
                                   seeds=[int(req.get("seed", 42))])
            # The LM goes in the SECOND slot. Passing None here is what kept
            # thinking mode switched off no matter what `thinking` said.
            lm = lm_get(progress) if thinking else None
            result = generate_music(handler, lm, params, cfg,
                                    save_dir=OUTDIR)
            elapsed = time.time() - t0

            # `GenerationResult.audios` is the real field -- a list of dicts,
            # each carrying the path and the params that made it. The old code
            # looked for `audio_files`/`files`, never found either, and always
            # fell through to diffing the output folder.
            #
            # That fallback breaks the moment seeds are honoured: ACE-Step
            # names outputs with `generate_uuid_from_params`, documented as
            # "same parameters will always generate the same UUID", so a repeat
            # render overwrites its predecessor and the diff sees no new file.
            # It reported "produced no output file" for a generation that had
            # in fact succeeded.
            files = []
            for a in (getattr(result, "audios", None) or []):
                p = a.get("path") or a.get("audio_path") if isinstance(a, dict) else None
                if p and os.path.exists(p):
                    files.append(p)
            if not files:
                after = set(glob.glob(os.path.join(OUTDIR, "*.flac"))
                            + glob.glob(os.path.join(OUTDIR, "*.wav")))
                files = sorted(after - before, key=os.path.getmtime)
            if not files:
                if not getattr(result, "success", True):
                    raise RuntimeError(
                        f"ACE-Step failed: {getattr(result, 'error', '')}")
                raise RuntimeError("ACE-Step produced no output file")
            path = files[-1]

            import soundfile as sf
            info = sf.info(path)
            peak = torch.cuda.max_memory_allocated(0) / 1024 ** 3
            torch.cuda.empty_cache()
            # Read back from the HANDLER, not from the request: this is what
            # was actually in the forward pass, which is the only version worth
            # writing into a sidecar that outlives the session.
            st = {}
            try:
                st = handler.get_lora_status() or {}
            except Exception:
                pass
            emit({"type": "done", "path": path, "seconds": info.duration,
                  "sampling_rate": info.samplerate, "elapsed": elapsed,
                  "vram_peak": peak,
                  "lora_path": _LORA["path"], "lora_scale": _LORA["scale"],
                  "lora_used": bool(_LORA["path"]) and bool(
                      st.get("use_lora", True)),
                  # Every take, so the caller can surface the siblings instead
                  # of stranding them. `path` stays the primary one.
                  "takes": files, "takes_requested": takes,
                  "thinking": thinking,
                  "vram_idle": torch.cuda.memory_allocated(0) / 1024 ** 3})
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            emit({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
