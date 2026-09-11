"""Preprocess and train an ACE-Step LoRA, with a hard gate on the GPU.

The card is shared three ways -- this studio, ComfyUI, and now training -- and
training is the only one of the three that runs for hours and cannot yield. So
nothing here starts until the GPU is provably free, and "provably" means all
of these, not one:

  * the studio's job lane is idle           (GET /api/jobs?active=1)
  * no model is resident in the studio      (GET /api/vram -> loaded)
  * ComfyUI is not running
  * nvidia-smi reports enough free memory   (measured, not assumed)

Any one of them failing stops the run with a message saying which, because a
training run that starts against a busy card does not fail cleanly -- it either
OOMs an hour in or quietly thrashes to shared system memory and takes ten times
as long, which is the same trap the MiniMax work hit.

Two stages, resumable independently:

    python run_lora.py preprocess      # audio -> .pt tensors, ~minutes, GPU
    python run_lora.py train           # the LoRA itself, hours, GPU
    python run_lora.py check           # just the gate, changes nothing

Preprocess writes tensors once; re-running `train` with different
hyperparameters reuses them.
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
ACE = os.path.join(ROOT, "acestep", "ACE-Step-1.5")
ACE_PY = os.path.join(ROOT, "acestep", ".venv", "Scripts", "python.exe")
CKPT = os.path.join(ROOT, "acestep", "checkpoints")
STUDIO = "http://127.0.0.1:7862"

DATASET = os.path.join(HERE, "dataset.json")
TENSORS = os.path.join(HERE, "preprocessed")
OUTPUT = os.path.join(HERE, "lora_out")

# These come from ACE-Step's own constants, not from me guessing.
# `gpu_utils.py` documents, for the 2B decoder:
#     _WEIGHT_OFFSET_2B        = 4096 MiB   decoder weights
#     _BYTES_PER_BATCH_2B_BF16 = 1200 MiB   forward+backward, ONE sample, and
#                                           explicitly "very conservative", for
#                                           FULL finetuning where every
#                                           parameter carries a gradient
# A LoRA at batch_size=1 with gradient checkpointing is well under that batch
# figure -- rank-16 adapters and their optimizer state are a few tens of MiB --
# so 4.0 + 1.2 = 5.2 GiB is already an upper bound, and 0.8 safety puts the
# floor at ~6.5 GB. My first pass hardcoded 8.0, which would have refused runs
# that actually fit.
WEIGHTS_GB = 4096 / 1024.0
PER_BATCH_GB = 1200 / 1024.0
SAFETY = 0.8

NEED_GB = {
    # Preprocess runs the VAE and text encoder over the audio; no optimizer,
    # no gradients, and the DiT is never loaded.
    "preprocess": round(WEIGHTS_GB * 0.75 / SAFETY, 1),
    "train": round((WEIGHTS_GB + PER_BATCH_GB) / SAFETY, 1),
}


def gpu_free_gb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        total, used = (int(x) for x in r.stdout.strip().split("\n")[0].split(","))
        return (total - used) / 1024.0, total / 1024.0
    except Exception:
        return None, None


def studio_state():
    """(active_jobs, resident_model) or (None, None) if the studio is down.

    A studio that is not running is fine -- it just cannot be asked. It is
    only a blocker when it answers AND says it is busy.
    """
    try:
        with urllib.request.urlopen(f"{STUDIO}/api/jobs?active=1", timeout=5) as r:
            jobs = json.loads(r.read()).get("jobs", [])
        with urllib.request.urlopen(f"{STUDIO}/api/vram", timeout=5) as r:
            loaded = json.loads(r.read()).get("loaded")
        return jobs, loaded
    except Exception:
        return None, None


def comfy_running():
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                            "/FO", "CSV"], capture_output=True, text=True,
                           timeout=30)
        # tasklist cannot show command lines, so ask WMI only if needed.
        if "python.exe" not in (r.stdout or ""):
            return False
        q = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*ComfyUI*' } | "
             "Measure-Object).Count"],
            capture_output=True, text=True, timeout=60)
        return int((q.stdout or "0").strip() or 0) > 0
    except Exception:
        return False


DESKTOP_HOGS = ("Unity", "chrome", "brave", "msedge", "Discord", "firefox",
                "ComfyUI", "parsecd", "obs64", "Photoshop", "AfterFX")


def desktop_users():
    """Desktop apps currently on the GPU, biggest first.

    On this box the studio is rarely what is holding the card -- Unity, two
    browsers and Discord sit on ~3.7 GB of a 10 GB card between them, which is
    the difference between the 8 GB preset fitting and not. "Only 6.3 GB free"
    is not an actionable message; "close Unity" is.
    """
    try:
        q = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process | Where-Object { $_.Name -match '"
             + "|".join(DESKTOP_HOGS)
             + "' } | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=60)
        # Unity alone spawns ~8 helper processes; report the app, not its
        # process tree, or the advice becomes a wall of noise.
        seen = {}
        for line in (q.stdout or "").splitlines():
            name = line.strip()
            if not name:
                continue
            app = next((h for h in DESKTOP_HOGS
                        if name.lower().startswith(h.lower())), name)
            seen[app] = seen.get(app, 0) + 1
        return [f"{k} x{v}" for k, v in sorted(seen.items(),
                                               key=lambda kv: -kv[1])]
    except Exception:
        return []


def gate(stage, force=False):
    """Print the verdict; return True when it is safe to run."""
    need = NEED_GB.get(stage, 8.0)
    free, total = gpu_free_gb()
    jobs, loaded = studio_state()
    comfy = comfy_running()

    print("GPU gate")
    print(f"  card            : {free:.2f} GB free of {total:.0f} GB"
          if free is not None else "  card            : nvidia-smi unavailable")
    if jobs is None:
        print("  studio (:7862)  : not answering (fine - nothing to wait for)")
    else:
        print(f"  studio jobs     : {len(jobs)} active"
              + (f" - {', '.join(j['kind'] for j in jobs)}" if jobs else ""))
        print(f"  studio resident : {loaded or 'nothing'}")
    print(f"  ComfyUI         : {'RUNNING' if comfy else 'not running'}")
    apps = desktop_users()
    print(f"  desktop on GPU  : {', '.join(apps) if apps else 'none'}")

    blocks = []
    if jobs:
        blocks.append(f"{len(jobs)} studio job(s) running")
    if loaded:
        blocks.append(f"{loaded} is resident in the studio "
                      f"(unload it from the GPU sheet)")
    if comfy:
        blocks.append("ComfyUI is running")
    if free is None:
        blocks.append("nvidia-smi did not answer")
    elif free < need:
        blocks.append(f"only {free:.2f} GB free, {stage} wants {need:.1f} GB")
        if apps:
            blocks.append("closing these would free most of it: "
                          + ", ".join(a.split(" x")[0] for a in apps))

    if not blocks:
        print(f"\n  CLEAR — {stage} can run.\n")
        return True
    print("\n  BLOCKED:")
    for b in blocks:
        print(f"    - {b}")
    if force:
        print("\n  --force given; running anyway.\n")
        return True
    print("\n  Nothing was started. Clear the blockers, or pass --force.\n")
    return False


def run(cmd, cwd=ACE):
    print("+ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n",
          flush=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.call(cmd, cwd=cwd, env=env)


def do_preprocess(a):
    if not os.path.exists(DATASET):
        sys.exit(f"no dataset at {DATASET} — run build_dataset.py first")
    n = len(json.load(open(DATASET, encoding="utf-8"))["samples"])
    print(f"dataset: {n} samples -> {TENSORS}\n")
    os.makedirs(TENSORS, exist_ok=True)
    return run([ACE_PY, os.path.join(HERE, "_preprocess_entry.py"),
                "--dataset-json", DATASET,
                "--output-dir", TENSORS,
                "--checkpoint-dir", CKPT,
                "--variant", a.variant,
                "--max-duration", str(a.max_duration)], cwd=HERE)


# `train.py fixed` has NO --preset flag; the preset JSONs under
# acestep/training_v2/presets/ are data the UI expands. Every key does have a
# matching CLI flag, so expand them here rather than pass a name the parser
# will reject -- which it does instantly, after the GPU window has opened.
PRESET_FLAGS = {
    "rank": "--rank", "alpha": "--alpha", "dropout": "--dropout",
    "learning_rate": "--lr", "batch_size": "--batch-size",
    "gradient_accumulation": "--gradient-accumulation",
    "epochs": "--epochs", "warmup_steps": "--warmup-steps",
    "weight_decay": "--weight-decay", "max_grad_norm": "--max-grad-norm",
    "seed": "--seed", "shift": "--shift",
    "num_inference_steps": "--num-inference-steps",
    "optimizer_type": "--optimizer-type",
    "scheduler_type": "--scheduler-type", "cfg_ratio": "--cfg-ratio",
    "save_every": "--save-every", "log_every": "--log-every",
    "log_heavy_every": "--log-heavy-every",
    "sample_every_n_epochs": "--sample-every-n-epochs",
    "bias": "--bias", "attention_type": "--attention-type",
}
# BooleanOptionalAction: present enables, --no-X disables. Never a value.
PRESET_BOOLS = {"gradient_checkpointing": "--gradient-checkpointing",
                "offload_encoder": "--offload-encoder"}
PRESET_DIR = os.path.join(ACE, "acestep", "training_v2", "presets")


# Card size -> preset. The 10 GB card is the FLOOR this was built on, not the
# target: rank scales with the memory you have, and rank is most of what
# decides how much of your library a LoRA can actually absorb. `auto` reads the
# card at launch, so the same scheduled command does the right thing after a
# GPU upgrade with nothing to edit.
PRESET_BY_TOTAL_GB = [
    (11.0, "vram_8gb"),        # 10 GB and below: rank 16, 8-bit optim, offload
    (14.0, "vram_12gb"),       # 12 GB: rank 32
    (20.0, "vram_16gb"),       # 16 GB: rank 64, nothing offloaded
    (float("inf"), "vram_24gb_plus"),   # 24 GB+: rank 128, batch accum halved
]


def auto_preset(verbose=True):
    _free, total = gpu_free_gb()
    if not total:
        if verbose:
            print("  card size unknown - falling back to vram_8gb")
        return "vram_8gb"
    name = next(p for cap, p in PRESET_BY_TOTAL_GB if total < cap)
    if verbose:
        print(f"  auto preset: {total:.0f} GB card -> {name}")
    return name


def preset_args(name):
    if name == "auto":
        name = auto_preset()
    path = os.path.join(PRESET_DIR, f"{name}.json")
    if not os.path.isfile(path):
        have = [f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json")]
        sys.exit(f"no preset {name!r}. Available: {', '.join(sorted(have))}")
    with open(path, encoding="utf-8") as fh:
        p = json.load(fh)
    out = []
    for key, flag in PRESET_FLAGS.items():
        if p.get(key) is not None:
            out += [flag, str(p[key])]
    for key, flag in PRESET_BOOLS.items():
        if key in p:
            out.append(flag if p[key] else "--no-" + flag[2:])
    if p.get("target_modules_str"):
        out += ["--target-modules"] + p["target_modules_str"].split()
    return out, p


def do_train(a):
    if not os.path.isdir(TENSORS) or not os.listdir(TENSORS):
        sys.exit(f"no tensors in {TENSORS} - run `preprocess` first")
    os.makedirs(OUTPUT, exist_ok=True)
    flags, p = preset_args(a.preset)
    print(f"preset {a.preset}: {p.get('description', '')}")
    print(f"  rank {p.get('rank')} alpha {p.get('alpha')} "
          f"lr {p.get('learning_rate')} epochs {p.get('epochs')} "
          f"bs {p.get('batch_size')}x{p.get('gradient_accumulation')}\n")
    # `--yes` and `--plain` belong to the TOP-LEVEL parser, before the
    # subcommand: `train.py [--plain] [--yes] {vanilla,fixed,estimate} ...`.
    # Putting them after `fixed` is an "unrecognized arguments" exit.
    cmd = [ACE_PY, os.path.join(ACE, "train.py"), "--yes", "--plain", "fixed",
           "--checkpoint-dir", CKPT,
           "--model-variant", a.variant,
           "--dataset-dir", TENSORS,
           "--output-dir", OUTPUT] + flags
    # ACE-Step sandboxes filesystem access: `path_safety._SAFE_ROOT` is set to
    # os.getcwd() at import, and `data_module` runs the tensor dir through it,
    # so a dataset outside the cwd is rejected as "Path escapes safe root".
    # Running from B:\AudioDev makes one root that covers the tensors, the
    # output and the checkpoints. train.py is passed absolutely, and Python
    # puts its directory on sys.path, so `import acestep` still resolves.
    return run(cmd, cwd=ROOT)
    if a.epochs:
        # An explicit --epochs must win over the preset's; argparse takes the
        # last occurrence, so append rather than replace.
        cmd += ["--epochs", str(a.epochs)]
    return run(cmd)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("check", "preprocess", "train"))
    ap.add_argument("--preset", default="auto",
                    help="auto (by card size) | vram_8gb | vram_12gb | vram_16gb | vram_24gb_plus | quick_test")
    ap.add_argument("--variant", default="turbo")
    ap.add_argument("--epochs", type=int, default=0, help="0 = the preset's")
    ap.add_argument("--max-duration", type=float, default=240.0)
    ap.add_argument("--force", action="store_true",
                    help="run even though the GPU is not free (don't)")
    a = ap.parse_args()

    if not gate(a.stage if a.stage != "check" else "train", a.force):
        return 1 if a.stage != "check" else 0
    if a.stage == "check":
        return 0
    return do_preprocess(a) if a.stage == "preprocess" else do_train(a)


if __name__ == "__main__":
    sys.exit(main())
