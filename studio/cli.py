"""Drive the workers without Gradio, to keep protocol bugs separable from UI bugs.

    python cli.py --model minimax --duration 15
    python cli.py --model acestep --duration 15
    python cli.py --model acestep --src "B:\\AudioDev\\Music\\track.wav"
    python cli.py --both                      # smoke-test the model switch
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supervisor import Supervisor, BACKENDS, gpu_memory

PROMPTS = {
    "minimax": (
        "Global Metadata\n"
        "Basic Attributes: bpm is 100. key is A, and scale is minor. Indie folk.\n"
        "Vocal Details\n"
        "Vocal Gender & Timbre: Singer A (Female). Warm, close, breathy.\n"
        "Arrangement\n"
        "Primary: fingerpicked acoustic guitar. Secondary: soft upright bass."),
    "acestep": "indie folk, fingerpicked acoustic guitar, warm analog production",
}
LYRICS = "[verse]\nWalking home through amber light\n[chorus]\nHold the quiet close tonight"


def run(sup, model, args):
    print(f"\n=== {BACKENDS[model]['label']} ===")
    used, total = gpu_memory()
    if used is not None:
        print(f"GPU {used:.1f}/{total:.1f} GB used before load")
    # MiniMax rejects empty lyrics -- it has no instrumental mode -- so
    # --instrumental only applies to ACE-Step.
    instrumental = args.instrumental and model != "minimax"
    t0 = time.time()
    params = dict(
        prompt=args.prompt or PROMPTS[model],
        lyrics="" if instrumental else LYRICS,
        duration=args.duration, steps=args.steps, seed=args.seed,
        instrumental=instrumental,
        out_name=f"cli_{model}.wav",
    )
    if args.src:
        params["src_audio"] = args.src
    res = sup.generate(model, params,
                       on_progress=lambda m: print("  ..", m.get("msg", m)))
    print(f"  -> {res['path']}")
    print(f"     {res['seconds']:.1f}s @ {res['sampling_rate']} Hz | "
          f"gen {res['elapsed']:.0f}s | peak {res.get('vram_peak', 0):.2f} GB | "
          f"idle {res.get('vram_idle', 0):.2f} GB | wall {time.time()-t0:.0f}s")
    used, total = gpu_memory()
    if used is not None:
        print(f"     GPU now {used:.1f}/{total:.1f} GB used")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(BACKENDS), default="minimax")
    ap.add_argument("--both", action="store_true",
                    help="run both back to back; exercises the worker switch")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--src", default=None, help="source audio -> ACE-Step cover")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--instrumental", action="store_true")
    args = ap.parse_args()

    sup = Supervisor()
    try:
        models = list(BACKENDS) if args.both else [args.model]
        for m in models:
            a = argparse.Namespace(**vars(args))
            if a.steps is None:
                a.steps = 30 if m == "minimax" else 8
            run(sup, m, a)
    finally:
        sup.unload()
        used, total = gpu_memory()
        if used is not None:
            print(f"\nafter unload: GPU {used:.1f}/{total:.1f} GB used")


if __name__ == "__main__":
    sys.exit(main())
