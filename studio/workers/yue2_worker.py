"""YuE2 worker. Runs inside <root>\\yue2\\.venv and speaks the studio's stdio
JSON protocol, like the ACE-Step and MiniMax workers.

YuE2 is the one engine here with a SYMBOLIC stage: it writes an ABC score
(melody, chords, structure, tempo) before any audio, and that score is text
the user can read, edit and hand back. So two requests exist:

  generate    style + lyrics (+ optional score) -> plan -> sing -> audio
  plan_only   stop after the score. No audio, nothing in the library; the
              score comes back in the reply for the editor.

Every artefact of a render is kept under <OUTDIR>\\yue2\\<stem>\\ exactly as
YuE2's own save_artifacts writes it (score.abc, plan.json, latent.npy,
result.json ...) so a song can be re-decoded or re-planned later; the audio
is copied up to <OUTDIR>\\<stem>.flac where the library expects it.

Load-time settings, read from the environment because the supervisor starts
this process and cannot pass arguments:

  YUE2_MODEL / YUE2_VAE      hub ids; default m-a-p/YuE2-3B and m-a-p/YuE2-Vae
  YUE2_MEMORY_BUDGET_GIB     YuE2's per-process VRAM cap (its default is 24)
  YUE2_QUANT                 none | fp8. fp8 needs compute capability 8.9+
                             (RTX 40/50 series) and is the one lever YuE2
                             offers below its documented 24 GB floor.
  YUE2_OFFLOAD_AR            1 to park the language model in RAM while the
                             acoustic stage runs
  HF_HOME                    defaults to <root>\\minimax\\hf like everything else
  STUDIO_OUTDIR              where audio lands; the server sets it

NOT YET RUN ON HARDWARE (2026-09-11): this box has a 10 GB card and the model
needs 24. Everything here was exercised end to end against a stub pipeline
carrying the same signatures (studio\\testing\\yue2_stub); the first real
render on the bigger machine is the test, and web\\srv_err.txt is where a
failure will show.
"""

import json
import os
import shutil
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, "..", ".."))

# The protocol channel is stdout. Everything else -- YuE2's own progress,
# transformers' warnings, stray prints -- goes to stderr, or it corrupts JSON.
PROTO = sys.stdout
sys.stdout = sys.stderr
PROTO.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "minimax", "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUTDIR = os.environ.get("STUDIO_OUTDIR") or os.path.join(ROOT, "Music", "studio")
MODEL = os.environ.get("YUE2_MODEL", "m-a-p/YuE2-3B")
VAE = os.environ.get("YUE2_VAE", "m-a-p/YuE2-Vae")
QUANT = os.environ.get("YUE2_QUANT", "none").strip().lower() or "none"


def emit(obj):
    PROTO.write(json.dumps(obj) + "\n")
    PROTO.flush()


def progress(msg, stage=None, frac=None):
    m = {"type": "progress", "msg": msg}
    if stage:
        m["stage"] = stage
    if frac is not None:
        m["frac"] = frac
    emit(m)


class TokenMeter:
    """Progress from YuE2's on_token callback, rate-limited.

    The pipeline reports one call per emitted token with the phase name
    ("abc" while planning, "semantic" while singing) and no total -- its own
    Progress class is explicit that a generation limit is not a target. So
    this reports a count every couple of seconds, never a fraction.
    """

    def __init__(self):
        self.n = {}
        self.last = 0.0

    def __call__(self, phase, _token):
        self.n[phase] = self.n.get(phase, 0) + 1
        now = time.time()
        if now - self.last < 2.0:
            return
        self.last = now
        if phase == "abc":
            progress(f"planning score: {self.n[phase]} tokens", stage="plan")
        else:
            progress(f"singing: {self.n[phase]} tokens", stage="semantic")


def stem_of(out_name):
    # free_out_name() hands out "<stamp>_<model>.wav"; the stem is the song id
    # YuE2 records in its request, and the name the audio and its folder get.
    base = os.path.basename(out_name or "yue2.wav")
    return os.path.splitext(base)[0] or "yue2"


def gb(n):
    return round(n / 1024 ** 3, 2)


class _Cuda:
    """torch.cuda memory calls, or zeros when there is no CUDA device.

    Lets the protocol be exercised against the stub with CUDA_VISIBLE_DEVICES
    empty, so a test never touches the shared card; on the real machine every
    call goes straight through.
    """

    def __init__(self, torch):
        self.t = torch
        self.on = torch.cuda.is_available()
        if self.on:
            # torch 2.10: the allocator's stats calls raise "Invalid device
            # argument" until the context exists. Measured here; the other
            # workers never hit it because a model load comes first.
            try:
                torch.cuda.init()
            except Exception:
                self.on = False

    def reset_peak(self):
        if self.on:
            try:
                self.t.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass                # bookkeeping must never fail a render

    def peak(self):
        return gb(self.t.cuda.max_memory_allocated(0)) if self.on else 0.0

    def idle(self):
        return gb(self.t.cuda.memory_allocated(0)) if self.on else 0.0

    def empty(self):
        if self.on:
            self.t.cuda.empty_cache()


def main():
    progress("importing YuE2")
    import torch
    from yue2 import YuE2Pipeline

    cuda = _Cuda(torch)
    kw = dict(vae=VAE, device="cuda" if cuda.on else "auto", progress=False)
    budget = os.environ.get("YUE2_MEMORY_BUDGET_GIB", "").strip()
    if budget:
        kw["memory_budget_gib"] = float(budget)
    if QUANT != "none":
        kw["quantization"] = QUANT
    if os.environ.get("YUE2_OFFLOAD_AR") == "1":
        kw["offload_ar"] = True

    progress(f"loading {MODEL} (bf16{', ' + QUANT if QUANT != 'none' else ''})",
             stage="load")
    t0 = time.time()
    try:
        pipe = YuE2Pipeline.from_pretrained(MODEL, **kw)
    except Exception as e:
        import traceback
        traceback.print_exc()
        emit({"type": "error", "msg": f"YuE2 init failed: {type(e).__name__}: {e}"})
        return
    progress(f"ready in {time.time() - t0:.0f}s")
    # `quality` is what the supervisor compares against a requested load-time
    # setting; only the quantisation is load-time here.
    emit({"type": "ready", "model": "yue2", "model_id": MODEL, "vae": VAE,
          "quality": {"quant": QUANT}})

    os.makedirs(OUTDIR, exist_ok=True)

    try:
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
                stem = stem_of(req.get("out_name"))
                cot = (req.get("cot") or "full").strip().lower()
                request = dict(style=req["prompt"], lyrics=req.get("lyrics") or "",
                               cot=cot, seed=int(req.get("seed", 831001)), id=stem)
                abc = (req.get("abc") or "").strip()
                if abc and cot != "off":
                    # A supplied score: YuE2 tokenises it and sings to it
                    # instead of planning one. protocol.SongRequest refuses a
                    # score with cot="off", and api.py refuses it first.
                    request["abc"] = abc
                if req.get("cfg_scale") is not None:
                    request["cfg_scale"] = float(req["cfg_scale"])

                plan_dir = os.path.join(OUTDIR, "yue2", stem)
                meter = TokenMeter()
                cuda.reset_peak()
                t0 = time.time()

                if req.get("plan_only"):
                    progress("planning score", stage="plan")
                    plan = pipe.plan(on_token=meter, **request)
                    plan.save(plan_dir)
                    elapsed = time.time() - t0
                    cuda.empty()
                    emit({"type": "done", "score": plan.abc or "",
                          "plan_dir": plan_dir, "cot": cot,
                          "truncated": {"abc": bool(plan.truncated)},
                          "elapsed": elapsed,
                          "vram_peak": cuda.peak(), "vram_idle": cuda.idle()})
                    continue

                progress("using the supplied score" if "abc" in request
                         else ("planning score" if cot != "off" else "generating"),
                         stage="plan")
                song = pipe(on_token=meter, **request)
                progress("saving", stage="save", frac=1.0)
                song.save_artifacts(plan_dir)
                path = os.path.join(OUTDIR, stem + ".flac")
                shutil.copyfile(os.path.join(plan_dir, "audio.flac"), path)
                elapsed = time.time() - t0
                seconds = len(song.audio) / float(song.sample_rate)
                peak = cuda.peak()
                cuda.empty()
                emit({"type": "done", "path": path, "seconds": seconds,
                      "sampling_rate": int(song.sample_rate), "elapsed": elapsed,
                      "vram_peak": peak,
                      # The score that was in the forward pass -- planned or
                      # supplied -- so the sidecar records what was sung to.
                      "score": song.abc or "", "plan_dir": plan_dir, "cot": cot,
                      "truncated": {k: bool(v) for k, v in
                                    (song.truncated or {}).items()},
                      "vram_idle": cuda.idle()})
            except Exception as e:
                import traceback
                traceback.print_exc()
                cuda.empty()
                emit({"type": "error", "msg": f"{type(e).__name__}: {e}"})
    finally:
        try:
            pipe.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
