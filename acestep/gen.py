"""
Minimal ACE-Step 1.5 generation, DiT only (no lyric LLM).

The LLM path needs nano-vllm, which has no Windows wheel, so `thinking` is off
and only the DiT + VAE are loaded. That is all the frame-rate experiment needs:
the comb, if any, is stamped by the VAE decoder.

Usage:
    python gen.py --caption "..." --seconds 30 --out out.wav
"""

import argparse
import os
import sys

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

os.environ.setdefault("ACESTEP_INIT_LLM", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.join(ROOT, "acestep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", default="upbeat indie pop, guitar, drums, "
                                         "warm analog production")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from acestep.handler import AceStepHandler
    from acestep.inference import (GenerationParams, GenerationConfig,
                                   generate_music)

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path=None,
        device=args.device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=True,      # 10 GB card, shared with other work
        offload_dit_to_cpu=False,
        quantization=None,
    )
    print("init:", status, "ok:", ok, "device:", handler.device)
    if not ok:
        sys.exit(1)

    params = GenerationParams(
        caption=args.caption,
        lyrics="[Instrumental]",
        instrumental=True,
        duration=args.seconds,
        inference_steps=args.steps,
        seed=args.seed,
        task_type="text2music",
        thinking=False,
    )
    config = GenerationConfig()

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    result = generate_music(handler, None, params, config, save_dir=outdir)

    files = getattr(result, "audio_files", None) or getattr(result, "files", None)
    print("result files:", files)
    if files:
        import shutil
        shutil.copy(files[0], args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
