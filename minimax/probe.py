"""Introspect the MiniMax-Music3 pipeline without loading any weights.

`ModularPipeline.from_pretrained` reads modular_model_index.json and builds the
block graph but does NOT pull tensors -- weights only arrive on
`load_components()`. So this reports the real call signature and the component
inventory for free, with zero VRAM.

Usage:
    python probe.py
"""

import os

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "minimax", "hf"))

REPO = "MiniMaxAI/MiniMax-Music3"


def main():
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(REPO)
    print("pipeline :", type(pipe).__name__)
    print("blocks   :", type(pipe.blocks).__name__)
    print()

    print("=== components declared ===")
    for name, spec in pipe._component_specs.items():
        # type_hint is resolved to a class for some components and left as a
        # ("library", "ClassName") pair for others.
        th = spec.type_hint
        if isinstance(th, (tuple, list)):
            th = th[-1]
        print(f"  {name:<20} {getattr(th, '__name__', th)}")
    print()

    print("=== accepted call inputs ===")
    try:
        for p in pipe.blocks.inputs:
            d = (p.description or "").strip().replace("\n", " ")
            print(f"  {p.name:<24} default={p.default!r:<12} {d[:88]}")
    except Exception as e:
        print("  (blocks.inputs unavailable:", e, ")")
    print()

    print("=== outputs ===")
    try:
        for p in pipe.blocks.outputs:
            print(f"  {p.name}")
    except Exception as e:
        print("  (unavailable:", e, ")")
    print()

    # These come from component configs, so they are authoritative -- and they
    # disagree with the model card, which claims 32 kHz output.
    print("=== rates (from configs, no weights) ===")
    for attr in ("sampling_rate", "frame_rate", "latent_hop_length",
                 "num_codebooks", "num_channels_latents"):
        try:
            print(f"  {attr:<20} {getattr(pipe, attr)}")
        except Exception as e:
            print(f"  {attr:<20} <needs weights: {type(e).__name__}>")


if __name__ == "__main__":
    main()
