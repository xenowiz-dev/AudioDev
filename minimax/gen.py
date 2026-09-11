"""MiniMax Music 3 generation on a single 10 GB GPU.

The upstream repo only ships an SGLang-Omni server that wants TWO CUDA GPUs, so
the usable path here is the diffusers `MiniMaxMusic3ModularPipeline`. The model
card's example ends with `pipe.to("cuda")`, which pushes ~27 GB of components at
a 10 GB card and OOMs immediately. Do not follow it.

What works instead:

  ComponentsManager.enable_auto_cpu_offload() -- everything lives in system RAM
  and each component is pulled onto the GPU only for its own forward pass, then
  evicted when the next one needs the room. It sizes every decision from
  `mem_get_info`, i.e. *actually free* VRAM, so a ComfyUI instance holding a few
  GB is respected automatically rather than fought over.

Even so the Qwen3-8B stage is the binding constraint: 16.4 GB in bf16 never fits
in 10 GB no matter how much is evicted around it. `--quant 4bit` takes it to
roughly 5.5 GB, which is what makes single-GPU generation possible at all. The
flow-matching transformer (~4.6 GB bf16) is the other big resident, and auto
offload keeps the two from being on the card at the same time.

`--dry-run` loads every component to CPU and reports its real memory footprint
without touching the GPU. Run that first -- it is free and it tells you whether
a real generation will fit before you start one.

Usage:
    python gen.py --dry-run
    python gen.py --dry-run --quant 4bit
    python gen.py --prompt "..." --lyrics-file song.txt --duration 30 -o out.wav
"""

import argparse
import os
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "minimax", "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Deliberately NOT setting HF_HUB_OFFLINE. diffusers resolves a *sharded*
# checkpoint through `_get_checkpoint_shard_files`, which calls the live
# `model_info()` API instead of reading the local index -- so offline mode makes
# the 9.3 GB flow-matching transformer fail while every unsharded component
# still loads. load_components() reports that failure as a warning and leaves
# `pipe.transformer = None`, so the pipeline looks loaded and dies later.
# Weights still come from the cache; only the shard listing needs the network.

REPO = "MiniMaxAI/MiniMax-Music3"

# The model was trained on long, sectioned captions, not keyword lists. The
# repo's own test script uses these three headings; short prompts work but give
# up most of the arrangement control.
DEFAULT_PROMPT = (
    "Global Metadata\n"
    "Basic Attributes: bpm is 96. key is C, and scale is major. Acoustic Pop.\n"
    "Sonics & Production Profile: warm, intimate, lightly compressed, natural room.\n"
    "Vocal Details\n"
    "Vocal Gender & Timbre: Singer A (Female). Soft, close, breathy head voice.\n"
    "Vocal Style: gentle and conversational in the verse, opening up in the chorus.\n"
    "Arrangement\n"
    "Primary: fingerpicked steel-string acoustic guitar throughout.\n"
    "Secondary: soft piano pads; brushed drums and upright bass enter at the chorus."
)

DEFAULT_LYRICS = """[verse]
Morning light filtering through the pine
Every quiet street is yours and mine
[chorus]
Softly the world begins to breathe"""


class ARPairOffloadStrategy:
    """Offload policy that respects the autoregressive stage's co-residency rule.

    `MiniMaxMusic3SemanticGenerationStep` drives the 8B LLM and the RVQ depth
    decoder together on *every* frame, so it refuses to run unless both sit on
    the same device. The stock `AutoOffloadStrategy` has no concept of that and
    evicts the LLM to make room for the 1.2 GB decoder, which fails the check.

    Two corrections, in order:

      1. `empty_cache()` before measuring. The strategy sizes decisions from
         `mem_get_info`, and moving a bitsandbytes 4-bit model onto the card
         leaves ~1.9 GB of scratch in torch's caching allocator -- so the card
         reports 0.63 GB free when 2.5 GB is genuinely available, and everything
         downstream looks unaffordable.
      2. Never evict one member of the AR pair to place the other. Any other
         resident model is still fair game, and sizing for non-AR components is
         left to the upstream strategy.
    """

    AR_PAIR = ("language_model", "rvq_depth_decoder")

    def __init__(self, memory_reserve_margin="1GB"):
        self.reserve = memory_reserve_margin
        self._base = self._make(memory_reserve_margin)

    @staticmethod
    def _make(memory_reserve_margin):
        from diffusers.modular_pipelines.components_manager import (
            AutoOffloadStrategy)
        return AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)

    def set_reserve(self, memory_reserve_margin):
        """Change the reserve without reloading the model.

        The margin is only read inside `__call__`, when a component is about to
        be placed -- `enable_auto_cpu_offload` passes it to the DEFAULT strategy
        and keeps no copy of its own. So replacing the delegate is enough to
        retune the reserve between requests, which is what keeps this knob out
        of the ~20 s reload that a precision change costs. Rebuilt rather than
        poked, so the "2GB" -> bytes parsing stays diffusers' own.
        """
        if memory_reserve_margin and memory_reserve_margin != self.reserve:
            self.reserve = memory_reserve_margin
            self._base = self._make(memory_reserve_margin)
        return self.reserve

    @classmethod
    def _is_ar(cls, model_id):
        # ids are "<component name>_<id>", e.g. "language_model_1196723399056"
        return any(model_id.startswith(p) for p in cls.AR_PAIR)

    def __call__(self, hooks, model_id, model, execution_device):
        import torch
        torch.cuda.empty_cache()
        chosen = self._base(hooks, model_id, model, execution_device)
        if self._is_ar(model_id):
            chosen = [h for h in chosen if not self._is_ar(h.model_id)]
        return chosen


def gb(n):
    return n / 1024 ** 3


def vram():
    """(free, total) GB on cuda:0, or (nan, nan) with no CUDA."""
    import torch
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    free, total = torch.cuda.mem_get_info(0)
    return gb(free), gb(total)


def build(args):
    import torch
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines import ComponentsManager

    load_kwargs = {"dtype": torch.bfloat16}

    if args.quant != "none":
        from transformers import BitsAndBytesConfig
        if args.quant == "4bit":
            qc = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            qc = BitsAndBytesConfig(load_in_8bit=True)
        # Keyed by component name: load_components() routes a dict value to the
        # matching component only, so this quantizes the 8B LLM and leaves the
        # flow-matching transformer and vocoder at full bf16 quality.
        load_kwargs["quantization_config"] = {"language_model": qc}

    if getattr(args, "quant_rvq", "none") != "none":
        # The RVQ depth decoder is the OTHER half of the co-resident AR pair --
        # 1.20 GB of the 7.49 GB floor. Shrinking it is the only lever left that
        # lowers the floor itself rather than the per-frame growth, and the
        # floor is what decides whether a long song can start at all.
        #
        # A DIFFERENT BitsAndBytesConfig class than the LLM's: this is a
        # diffusers ModelMixin, not a transformers model, and passing the
        # transformers config here is silently ignored.
        from diffusers import BitsAndBytesConfig as DiffusersBnB
        if args.quant_rvq == "4bit":
            rq = DiffusersBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
        else:
            rq = DiffusersBnB(load_in_8bit=True)
        load_kwargs.setdefault("quantization_config", {})
        load_kwargs["quantization_config"]["rvq_depth_decoder"] = rq

    cm = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(
        REPO, components_manager=cm, collection="minimax_music3")

    t0 = time.time()
    pipe.load_components(**load_kwargs)
    print(f"components loaded in {time.time() - t0:.0f}s", flush=True)

    # load_components() downgrades a per-component failure to a log warning and
    # carries on, so a missing 9 GB transformer surfaces only as a confusing
    # error deep in the denoise loop. Check the roster explicitly.
    missing = [n for n in pipe._component_specs if getattr(pipe, n, None) is None]
    if missing:
        raise RuntimeError(
            f"components failed to load: {', '.join(missing)}. Re-run with the "
            f"network reachable and HF_HUB_OFFLINE unset -- sharded components "
            f"need a live model_info() call even when fully cached.")

    # Critical with --quant: bitsandbytes quantizes on-device and leaves the
    # fp16 scratch in torch's caching allocator, so the card *looks* almost full
    # (measured: 0.59 GB reported free with only 6.29 GB of weights resident).
    # AutoOffloadStrategy sizes every decision from mem_get_info, so without this
    # it concludes nothing else fits and evicts the LLM to place the 1.2 GB RVQ
    # decoder -- which the AR stage rejects, since it needs both co-resident.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe, cm


def report(cm):
    import torch
    print()
    print(f"{'component':<22} {'params':>14} {'footprint GB':>13}  device")
    print("-" * 66)
    total = 0
    for name, c in cm.components.items():
        if not isinstance(c, torch.nn.Module):
            continue
        try:
            fp = c.get_memory_footprint()
        except AttributeError:
            fp = sum(p.numel() * p.element_size() for p in c.parameters())
        n = sum(p.numel() for p in c.parameters())
        dev = str(next(c.parameters()).device)
        total += fp
        print(f"{name:<22} {n:>14,} {gb(fp):>13.2f}  {dev}")
    print("-" * 66)
    print(f"{'TOTAL':<22} {'':>14} {gb(total):>13.2f}")
    return total


def to_numpy(a):
    import numpy as np
    if hasattr(a, "detach"):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a, dtype="float32")
    # soundfile wants (samples, channels); the pipeline emits (channels, samples)
    if a.ndim == 2 and a.shape[0] < a.shape[1]:
        a = a.T
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="music description; see DEFAULT_PROMPT for the "
                         "sectioned format the model was trained on")
    ap.add_argument("--prompt-file", dest="prompt_file", default=None)
    ap.add_argument("--lyrics", default=DEFAULT_LYRICS)
    ap.add_argument("--lyrics-file", dest="lyrics_file", default=None)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="upper bound in seconds; the LM may stop earlier")
    ap.add_argument("--steps", type=int, default=30,
                    help="flow-matching Euler steps per chunk")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("-o", "--out", default="out.wav")
    ap.add_argument("--quant", choices=("none", "4bit", "8bit"), default="4bit",
                    help="quantization for the 8B LLM only. bf16 needs 16.4 GB "
                         "and cannot fit a 10 GB card; 4bit needs ~5.5 GB.")
    ap.add_argument("--quant-rvq", dest="quant_rvq",
                    choices=("none", "8bit", "4bit"), default="none",
                    help="quantize the RVQ depth decoder too. It is 1.20 GB of "
                         "the 7.49 GB AR floor, and lowering the FLOOR is what "
                         "decides whether a long song can start at all.")
    ap.add_argument("--reserve", default="1GB",
                    help="VRAM the offloader keeps free for activations. 2GB "
                         "starves the AR pair on a 10 GB card; 1GB is the "
                         "verified setting.")
    ap.add_argument("--dry-run", action="store_true",
                    help="load and report footprints without generating. Note "
                         "that --quant still puts the LLM on the GPU: "
                         "bitsandbytes quantizes on-device at load time.")
    args = ap.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            args.prompt = fh.read().strip()
    if args.lyrics_file:
        with open(args.lyrics_file, encoding="utf-8") as fh:
            args.lyrics = fh.read().strip()

    import torch

    free, total = vram()
    print(f"GPU: {gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB "
          f"total, {free:.1f} GB free "
          f"({total - free:.1f} GB held by other processes)"
          if torch.cuda.is_available() else "GPU: none")
    print(f"quant={args.quant}  reserve={args.reserve}", flush=True)

    pipe, cm = build(args)
    footprint = report(cm)

    if args.dry_run:
        print()
        where = ("all on CPU" if args.quant == "none" else
                 "LLM already on the GPU -- bitsandbytes quantizes on-device")
        print(f"dry run: no generation. Weights total {gb(footprint):.2f} GB "
              f"({where}).")
        # The floor is NOT the largest single component. The autoregressive
        # stage drives the LLM and the RVQ depth decoder on every frame and
        # refuses to run unless both are on the device at once, so their sum is
        # what has to fit -- everything else can be evicted around them.
        pair = 0.0
        for want in ARPairOffloadStrategy.AR_PAIR:
            for name, c in cm.components.items():
                if name.startswith(want) and isinstance(c, torch.nn.Module):
                    pair += c.get_memory_footprint()
                    break
        print(f"AR pair (language_model + rvq_depth_decoder) {gb(pair):.2f} GB "
              f"-- the floor: both must be resident together.")
        if torch.cuda.is_available():
            # The floor is only the start. The KV cache and the per-frame
            # hidden states grow at ~7.3 MiB per SECOND of music, so what
            # actually fits is a question about duration, not about the model.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import ar_cache
            print()
            print(f"  {'song':>6}  {'cache':>8}  {'total':>8}   verdict")
            print("  " + "-" * 44)
            for s in (30, 60, 120, 180, 240, 300):
                need, fits, head = ar_cache.budget(s, free, base_gb=gb(pair))
                mark = f"fits ({head:+.1f} GB spare)" if fits else \
                    f"SPILLS ({-head:.1f} GB short)"
                print(f"  {s:>5}s  {ar_cache.projected_gb(s):>7.2f}G  "
                      f"{need:>7.2f}G   {mark}")
            print()
            print(f"  measured: spilling costs ~3x -- 8.9 GB free ran 7x "
                  f"realtime, 7.0 GB free ran 28x.")
        return 0

    cm.enable_auto_cpu_offload(
        device="cuda", memory_reserve_margin=args.reserve,
        offload_strategy=ARPairOffloadStrategy(args.reserve))
    torch.cuda.empty_cache()
    print(f"auto CPU offload enabled; {vram()[0]:.2f} GB free before generation",
          flush=True)

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    audio = pipe(
        prompt=args.prompt,
        lyrics=args.lyrics,
        audio_duration=args.duration,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        output="audios",
    )[0]
    dt = time.time() - t0

    import soundfile as sf
    a = to_numpy(audio)
    sf.write(args.out, a, pipe.sampling_rate)
    secs = a.shape[0] / pipe.sampling_rate
    print(f"wrote {args.out}  {secs:.1f}s @ {pipe.sampling_rate} Hz "
          f"{'stereo' if a.ndim > 1 and a.shape[1] == 2 else 'mono'}")
    print(f"generated in {dt:.0f}s ({dt / max(secs, 1e-9):.2f}x realtime)")
    print(f"peak VRAM allocated {gb(torch.cuda.max_memory_allocated(0)):.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
