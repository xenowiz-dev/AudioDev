"""Find the LoRA adapters on this box and describe them.

Two roots, both scanned recursively:

  B:\\AudioDev\\loras        the drop folder -- put anything downloaded here
  B:\\AudioDev\\train\\lora_out   whatever was trained here, including the
                            per-epoch checkpoints, because A/B-ing epoch 5
                            against epoch 10 is a real thing you want to do

An adapter is a DIRECTORY holding both `adapter_config.json` and
`adapter_model.safetensors`. That pair is the whole test: PEFT writes them
together, and a directory with only one of them is a half-finished save.

No torch, no peft, no safetensors import -- the studio venv has none of them
and this is what the UI calls to build its dropdown. Everything here is
stdlib reading of a small JSON file.
"""

import json
import os
import re

# Root discovery: AUDIODEV_ROOT if the launcher set it, else relative to
# this file. Every path below derives from it, so the tree moves as one.
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("AUDIODEV_ROOT") or os.path.normpath(os.path.join(_HERE, ".."))

ROOTS = [
    ("drop", os.path.join(ROOT, "loras")),
    ("trained", os.path.join(ROOT, "train", "lora_out")),
]

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"
MAX_DEPTH = 4          # deep enough for lora_out/checkpoints/epoch_10_.../

_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(s):
    return _SLUG.sub("-", (s or "").lower()).strip("-")


def is_adapter(path):
    return (os.path.isfile(os.path.join(path, CONFIG))
            and os.path.isfile(os.path.join(path, WEIGHTS)))


def _describe(root_key, root, path):
    """One entry, or None when the config will not parse."""
    try:
        with open(os.path.join(path, CONFIG), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    rel = os.path.relpath(path, root).replace("\\", "/")
    # The id has to survive a rename of nothing and stay stable across scans,
    # so it is derived from the path rather than from a counter or mtime.
    ident = f"{root_key}:{rel}"

    base = cfg.get("base_model_name_or_path") or ""
    rank = cfg.get("r")
    alpha = cfg.get("lora_alpha")
    targets = cfg.get("target_modules") or []
    if isinstance(targets, (set, tuple)):
        targets = list(targets)

    # `lora_out/final` on its own says nothing; the parent does.
    parts = [p for p in rel.split("/") if p]
    name = parts[-1] if parts else os.path.basename(root)
    if name in ("final", "adapter", "checkpoint") and len(parts) > 1:
        name = f"{parts[-2]} ({name})"
    elif len(parts) > 1:
        name = "/".join(parts[-2:])

    try:
        size = os.path.getsize(os.path.join(path, WEIGHTS))
        mtime = os.path.getmtime(os.path.join(path, WEIGHTS))
    except OSError:
        size, mtime = 0, 0

    bits = []
    if rank:
        bits.append(f"rank {rank}")
    if alpha:
        bits.append(f"alpha {alpha}")
    if size:
        bits.append(f"{size / 1048576:.0f} MB")

    return {
        "id": ident,
        "name": name,
        "label": f"{name} — {', '.join(bits)}" if bits else name,
        "path": path,
        "source": root_key,
        "rank": rank,
        "alpha": alpha,
        "peft_type": cfg.get("peft_type") or "",
        "target_modules": sorted(targets) if targets else [],
        "base_model": base,
        "base_model_name": os.path.basename(base.rstrip("\\/")) if base else "",
        "size_bytes": size,
        "mtime": mtime,
    }


def scan(roots=None):
    """Every adapter found, newest first. Never raises on a bad directory."""
    out = []
    for root_key, root in (roots or ROOTS):
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip("\\/").count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count(os.sep) - base_depth > MAX_DEPTH:
                dirnames[:] = []
                continue
            if is_adapter(dirpath):
                e = _describe(root_key, root, dirpath)
                if e:
                    out.append(e)
                # An adapter directory has no adapters inside it.
                dirnames[:] = []
    out.sort(key=lambda e: -e["mtime"])
    return out


def by_id(ident, roots=None):
    """Look one up. Returns None rather than raising -- a stale id from a
    remembered UI setting is a normal thing to receive, not an error."""
    if not ident:
        return None
    return next((e for e in scan(roots) if e["id"] == ident), None)


def adapter_name(ident):
    """A collision-proof name to load the adapter under.

    ACE-Step's own `_default_adapter_name_from_path` takes the directory's
    basename, so `train\\lora_out\\final` and `loras\\anything\\final` would
    both register as "final" and the second would silently not be what you
    asked for. Deriving from the full id cannot collide.
    """
    return _slug(ident) or "adapter"


def ensure_roots():
    """Create the drop folder so there is somewhere obvious to put files."""
    made = []
    for _key, root in ROOTS:
        if not os.path.isdir(root):
            try:
                os.makedirs(root, exist_ok=True)
                made.append(root)
            except OSError:
                pass
    return made


if __name__ == "__main__":
    ensure_roots()
    found = scan()
    print(f"{len(found)} adapter(s)\n")
    for e in found:
        print(f"  {e['id']}")
        print(f"    {e['label']}")
        print(f"    base={e['base_model_name'] or '(unrecorded)'} "
              f"type={e['peft_type']} targets={','.join(e['target_modules'])}")
        print(f"    {e['path']}")
        print()
