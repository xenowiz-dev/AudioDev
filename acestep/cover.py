"""
ACE-Step 1.5 "cover" task: regenerate an existing track in a new style.

Also an experiment. The source here is Suno-generated and carries Suno's 200 Hz
vocoder comb; ACE-Step decodes through a VAE whose frame rate is 25 Hz. So the
cover output answers a question worth knowing: does passing audio through a
second generator OVERWRITE the first one's fingerprint, preserve it, or leave
both?

Cover skips the lyric LLM, so it runs DiT-only -- which is what is installed.
Duration is locked to the source audio.

Usage:
    python cover.py --src in.wav --caption "..." --out out.wav [--strength 0.6]
"""

import argparse
import glob
import os
import shutil
import sys
import time

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

os.environ.setdefault("ACESTEP_INIT_LLM", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.join(ROOT, "acestep")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source audio to cover")
    ap.add_argument("--caption", required=True, help="target style prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="audio_cover_strength 0-1 (default 1.0); lower = more "
                         "style transfer, further from the source")
    ap.add_argument("--noise-strength", dest="noise_strength", type=float,
                    default=0.75,
                    help="cover_noise_strength: 0 = pure noise (IGNORES the "
                         "source entirely), 1 = closest to source. The library "
                         "default is 0.0, which produces an unrelated track -- "
                         "this is the parameter that actually makes it a cover.")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lyrics", default="",
                    help="lyrics text; leave empty to let the model sing freely. "
                         "'[Instrumental]' forces an instrumental.")
    ap.add_argument("--lyrics-file", dest="lyrics_file", default=None,
                    help="read lyrics from a file")
    ap.add_argument("--instrumental", action="store_true", default=False)
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
        offload_to_cpu=True,
        offload_dit_to_cpu=False,
        quantization=None,
    )
    print("init ok:", ok, "device:", handler.device, flush=True)
    if not ok:
        print(status)
        sys.exit(1)

    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    before = set(glob.glob(os.path.join(outdir, "*.flac"))
                 + glob.glob(os.path.join(outdir, "*.wav")))

    lyrics = args.lyrics
    if args.lyrics_file:
        with open(args.lyrics_file, encoding="utf-8") as fh:
            lyrics = fh.read().strip()

    params = GenerationParams(
        caption=args.caption,
        lyrics=lyrics,
        instrumental=args.instrumental,
        task_type="cover",
        # src_audio is the track being covered. reference_audio is a separate
        # STYLE reference -- passing the source as both muddies the
        # conditioning, so it is deliberately left unset.
        src_audio=args.src,
        audio_cover_strength=args.strength,
        cover_noise_strength=args.noise_strength,
        inference_steps=args.steps,
        seed=args.seed,
        thinking=False,
    )
    print(f"cover: noise_strength={args.noise_strength} "
          f"cover_strength={args.strength} "
          f"lyrics={'yes (%d chars)' % len(lyrics) if lyrics else 'free'}",
          flush=True)
    t0 = time.time()
    generate_music(handler, None, params, GenerationConfig(), save_dir=outdir)
    print(f"generated in {time.time()-t0:.0f}s", flush=True)

    # The pipeline names outputs by UUID; pick up whatever is new.
    after = set(glob.glob(os.path.join(outdir, "*.flac"))
                + glob.glob(os.path.join(outdir, "*.wav")))
    new = sorted(after - before, key=os.path.getmtime)
    if not new:
        print("no new output found")
        sys.exit(1)
    shutil.copy(new[-1], args.out)
    print("wrote", args.out)
    for extra in new[:-1]:
        print("  (also produced", os.path.basename(extra) + ")")


if __name__ == "__main__":
    main()
