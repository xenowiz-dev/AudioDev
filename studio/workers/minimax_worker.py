"""MiniMax Music 3 worker. Runs inside B:\\AudioDev\\minimax\\.venv.

Loads the pipeline once and stays warm, so only the first generation pays the
~20 s load and 4-bit quantization cost.

Everything model-specific is imported from `minimax\\gen.py` rather than copied,
so the offload fixes that make this fit in 10 GB (the AR-pair co-residency rule
and the bitsandbytes allocator-scratch workaround) cannot drift out of sync.

Between jobs the worker parks every component back on the CPU, so an idle warm
worker holds ~0 GB rather than sitting on the 4.5 GB flow-matching transformer.
"""

import json
import os
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "minimax"))
import ar_cache

# The protocol owns the real stdout. Anything that prints -- hf warnings,
# "components loaded in Xs", stray library chatter -- must land on stderr or it
# corrupts the JSON-lines stream.
PROTO = sys.stdout
sys.stdout = sys.stderr
PROTO.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

OUTDIR = os.path.join(ROOT, "Music", "studio")


def emit(obj):
    PROTO.write(json.dumps(obj) + "\n")
    PROTO.flush()


def progress(msg, stage=None, frac=None):
    m = {"type": "progress", "msg": msg}
    if stage:
        m["stage"] = stage
    if frac is not None:
        m["frac"] = round(max(0.0, min(1.0, frac)), 4)
    emit(m)


class StageMeter:
    """Per-stage progress from torch forward hooks.

    Not tqdm parsing: tqdm redraws with '\\r' and never emits a newline until it
    closes, and both the supervisor and the log readers iterate `for line in
    pipe`, which only yields on '\\n' -- a tqdm bar would surface once, at 100%.

    `register_forward_hook` is the supported nn.Module API. It lives in
    `_forward_hooks`, so it does not collide with accelerate's `_hf_hook`
    offload wrappers or with park()/unpark.
    """

    MIN_INTERVAL = 0.5      # seconds; 25 AR frames/s would otherwise be 25 msg/s
    MIN_DELTA = 0.02        # or 2% of the stage

    def __init__(self):
        self.n = 0
        self.total = 1
        self.stage = None
        self.t0 = time.time()
        self._last_emit = 0.0
        self._last_frac = -1.0

    def begin(self, stage, total, label):
        self.n, self.total, self.stage = 0, max(1, int(total)), stage
        self.label = label
        self.t0 = time.time()
        self._last_emit, self._last_frac = 0.0, -1.0
        progress(f"{label} 0%", stage=stage, frac=0.0)

    def tick(self, k=1):
        if self.stage is None:
            return
        self.n += k
        frac = min(1.0, self.n / self.total)
        now = time.time()
        if (now - self._last_emit < self.MIN_INTERVAL
                and frac - self._last_frac < self.MIN_DELTA):
            return
        self._last_emit, self._last_frac = now, frac
        el = now - self.t0
        eta = (el / frac - el) if frac > 0.02 else None
        tail = f" · ~{eta:.0f}s left" if eta and eta > 1 else ""
        progress(f"{self.label} {frac*100:.0f}%{tail}", stage=self.stage,
                 frac=frac)

    def end(self):
        self.stage = None


# Two vocabularies meet here. Everything above this worker (the API, the UI)
# says 4bit / 8bit / bf16, because that is what a user is choosing between.
# `gen.build` says 4bit / 8bit / none, where "none" means "do not quantize",
# i.e. bf16. Translate once, HERE: `build` treats any quant_rvq that is not
# "none" and not "4bit" as 8-bit, so letting the string "bf16" reach it would
# silently give 8-bit weights -- the exact quality change the user declined.
_QUANT = {"bf16": "none", "none": "none", "8bit": "8bit", "4bit": "4bit"}
_UNQUANT = {"none": "bf16", "8bit": "8bit", "4bit": "4bit"}


def _quant_env(var, default):
    """A load-time precision from the environment, normalised for gen.build."""
    raw = (os.environ.get(var) or default).strip().lower()
    if raw not in _QUANT:
        progress(f"warning: {var}={raw!r} is not one of "
                 f"{', '.join(sorted(_QUANT))} — using {default}")
        raw = default
    return _QUANT[raw]


def main():
    from types import SimpleNamespace

    progress("importing torch and diffusers")
    import torch
    import soundfile as sf
    from gen import build, to_numpy, ARPairOffloadStrategy, gb

    reserve = os.environ.get("OFFLOAD_RESERVE", "1GB")
    # Precision is baked into the weights AS THEY LOAD, so both of these are
    # read once, here, and can only be changed by starting a new worker. The
    # supervisor is what restarts us; see Supervisor.ensure(quality=...).
    #
    # QUANT_RVQ shrinks the OTHER half of the co-resident AR pair. The LLM is
    # 4-bit by default; the RVQ depth decoder is the remaining 1.20 GB of the
    # 7.49 GB floor, and the floor is what decides whether a long song can
    # start at all. Off by default because it touches acoustic detail --
    # set QUANT_RVQ=8bit (or 4bit) to trade some of that for length.
    quant_llm = _quant_env("QUANT_LLM", "4bit")
    quant_rvq = _quant_env("QUANT_RVQ", "none")
    progress(f"loading components (LLM {_UNQUANT[quant_llm]}, "
             f"RVQ {_UNQUANT[quant_rvq]})")
    t0 = time.time()
    pipe, cm = build(SimpleNamespace(quant=quant_llm, quant_rvq=quant_rvq))
    if quant_rvq != "none":
        progress(f"RVQ depth decoder quantized to {quant_rvq}")
    # Kept in a variable: the reserve is the one offload knob that can be
    # retuned per request (set_reserve), unlike the two precisions above.
    strategy = ARPairOffloadStrategy(reserve)
    cm.enable_auto_cpu_offload(device="cuda", memory_reserve_margin=reserve,
                              offload_strategy=strategy)
    torch.cuda.empty_cache()

    # The MEASURED floor for the precisions we just loaded: the AR stage holds
    # the LLM and the RVQ depth decoder together, so their sum is what every
    # budget question starts from. Measured rather than taken from a table,
    # because the table cannot know which precisions this worker was started
    # with -- and the auto KV choice below is only as honest as this number.
    ar_floor = 0.0
    for want in ARPairOffloadStrategy.AR_PAIR:
        for cname, comp in cm.components.items():
            if cname.startswith(want) and isinstance(comp, torch.nn.Module):
                try:
                    ar_floor += gb(comp.get_memory_footprint())
                except AttributeError:
                    ar_floor += gb(sum(p.numel() * p.element_size()
                                       for p in comp.parameters()))
                break
    progress(f"AR floor {ar_floor:.2f} GB "
             f"(LLM {_UNQUANT[quant_llm]} + RVQ {_UNQUANT[quant_rvq]})")
    progress(f"ready in {time.time() - t0:.0f}s")

    def park():
        """Push every component back to CPU, keeping the offload hooks live."""
        for h in (cm.model_hooks or []):
            try:
                h.offload()
            except Exception:
                pass
        torch.cuda.empty_cache()

    # Progress hooks, registered once. Counting is derived from the source,
    # not guessed:
    #   AR      encoders.py calls `language_model.model(...)` once for the
    #           prompt and once per frame. `_generate_depth_codes` only touches
    #           `language_model.model.embed_tokens`, a submodule, so it does
    #           not trip this hook. -> calls ~= frames + 2.
    #   DENOISE denoise.py calls `components.transformer(...)` once per guider
    #           branch per step, over `chunks x steps`. The branch count is the
    #           guider's `num_conditions` (2 with CFG on), which can vary by
    #           step range -- so the total is an estimate and the meter clamps.
    meter = StageMeter()

    def ar_hook(*a, **k):
        if meter.stage == "ar":
            meter.tick()

    def denoise_hook(*a, **k):
        # The first transformer call IS the end of the AR stage -- no separate
        # signal exists, and this cannot fire early: denoise only runs once the
        # semantic block has returned all its frames.
        if meter.stage != "denoise":
            meter.begin("denoise", meter.denoise_total, "flow matching")
        meter.tick()

    pipe.language_model.model.register_forward_hook(ar_hook)
    pipe.transformer.register_forward_hook(denoise_hook)

    park()
    # `quality` is what we ACTUALLY loaded, in the UI's vocabulary. The
    # supervisor treats this as ground truth when deciding whether a request
    # needs a different worker, so it must describe the weights in memory --
    # never the request that asked for them.
    emit({"type": "ready", "model": "minimax",
          "sampling_rate": pipe.sampling_rate,
          "quality": {"llm": _UNQUANT[quant_llm],
                      "rvq": _UNQUANT[quant_rvq],
                      "reserve": strategy.reserve,
                      "base_gb": round(ar_floor, 2)}})

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
            # MiniMax has no instrumental mode: the text-encoder block declares
            # lyrics required and rejects an empty string outright. Fail with
            # something the user can act on rather than passing a guessed
            # "[instrumental]" token the checkpoint may never have been trained on.
            if not (req.get("lyrics") or "").strip():
                raise ValueError(
                    "MiniMax Music 3 requires lyrics -- it has no instrumental "
                    "mode. Use ACE-Step for instrumentals.")

            torch.cuda.reset_peak_memory_stats(0)
            t0 = time.time()

            # Per-request, no reload: the offloader reads the margin when it
            # places a component, not when offloading was enabled.
            if req.get("reserve"):
                got = strategy.set_reserve(str(req["reserve"]))
                progress(f"offload reserve: {got}")

            duration = float(req.get("duration", 30.0))
            steps = int(req.get("steps", 30))
            # Same bounds the pipeline uses: 25 Hz frames, capped at 9000, and
            # 200-frame denoising windows (MiniMaxMusic3ChunkLoopWrapper).
            max_frames = min(int(duration * pipe.frame_rate), 9000)
            chunks = max(1, -(-max_frames // 200))
            branches = getattr(pipe.guider, "num_conditions", 2) or 2
            meter.denoise_total = chunks * steps * branches

            # Pick the KV cache for THIS request. The default DynamicCache
            # rebuilds itself with torch.cat every frame across all 36 layers,
            # which both costs O(N^2) copying and creeps the peak upward until
            # a long song spills to system memory mid-run. Choose by what
            # actually fits in the VRAM that is free right now.
            free_now = gb(torch.cuda.mem_get_info(0)[0])
            want = (req.get("kv_cache") or "auto").lower()
            if want not in ("auto", "static", "quantized", "dynamic"):
                # ar_cache.install() would read an unknown mode as "static" and
                # say nothing -- a typo must not quietly hand back the mode with
                # the biggest footprint.
                progress(f"warning: unknown kv_cache {want!r} — using auto")
                want = "auto"
            if want == "auto":
                fits16 = ar_cache.budget(duration, free_now, base_gb=ar_floor,
                                         kv_bits=16)[1]
                fits4 = ar_cache.budget(duration, free_now, base_gb=ar_floor,
                                        kv_bits=4)[1]
                want = "static" if fits16 else "quantized"
                if not fits16 and not fits4:
                    progress("warning: this length does not fit even with a "
                             "4-bit cache — it will spill and run slowly")
            cache_h = ar_cache.install(
                pipe, mode=want, max_frames=max_frames,
                verbose=lambda m: progress(m))
            progress(f"KV cache: {want} · {free_now:.1f} GB free")

            meter.begin("ar", max_frames + 2, "singing (autoregressive)")

            audio = pipe(
                prompt=req["prompt"],
                lyrics=req["lyrics"],
                audio_duration=float(req.get("duration", 30.0)),
                num_inference_steps=int(req.get("steps", 30)),
                generator=torch.Generator("cuda").manual_seed(
                    int(req.get("seed", 7))),
                output="audios",
            )[0]
            elapsed = time.time() - t0
            meter.end()
            # Drop the wrapper and the cache before the vocoder stage: a warm
            # worker serves many requests and must not carry one song's KV
            # into the next.
            try:
                cache_h.reset()
                cache_h.uninstall()
            except Exception:
                pass
            torch.cuda.empty_cache()
            progress("decoding to audio", stage="vocoder", frac=1.0)

            a = to_numpy(audio)
            path = os.path.join(OUTDIR, req.get("out_name", "minimax.wav"))
            sf.write(path, a, pipe.sampling_rate)
            peak = gb(torch.cuda.max_memory_allocated(0))
            park()
            emit({"type": "done", "path": path,
                  "seconds": a.shape[0] / pipe.sampling_rate,
                  "sampling_rate": pipe.sampling_rate,
                  "elapsed": elapsed, "vram_peak": peak,
                  "vram_idle": gb(torch.cuda.memory_allocated(0))})
        except Exception as e:
            import traceback
            traceback.print_exc()
            meter.end()
            try:
                cache_h.reset()
                cache_h.uninstall()
            except Exception:
                pass
            try:
                park()
            except Exception:
                pass
            emit({"type": "error", "msg": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
