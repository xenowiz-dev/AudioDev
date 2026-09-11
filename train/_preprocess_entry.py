"""Call ACE-Step's preprocessor with named arguments. Run by run_lora.py.

A file rather than an inline `python -c`: the arguments are Windows paths with
spaces and backslashes, and building that command as a string is how you get a
quote collision that only shows up the first time someone runs it for real, at
which point the failure looks like a bug in the preprocessor. This is also
syntax-checkable before the GPU is ever free.
"""
import argparse
import sys
import os

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

ACE = os.path.join(ROOT, "acestep", "ACE-Step-1.5")
sys.path.insert(0, ACE)

from acestep.training_v2.preprocess import (            # noqa: E402
    preprocess_audio_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--variant", default="turbo")
    ap.add_argument("--max-duration", type=float, default=240.0)
    a = ap.parse_args()

    done = 0

    def progress(*args, **kw):
        nonlocal done
        done += 1
        if done % 10 == 0:
            print(f"  ... {done} samples", flush=True)

    preprocess_audio_files(
        audio_dir=None,
        output_dir=a.output_dir,
        checkpoint_dir=a.checkpoint_dir,
        variant=a.variant,
        max_duration=a.max_duration,
        dataset_json=a.dataset_json,
        progress_callback=progress,
    )
    print("preprocess finished")


if __name__ == "__main__":
    main()
