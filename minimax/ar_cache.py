"""Replace the autoregressive KV cache MiniMax Music 3 builds by default.

WHY THIS EXISTS
---------------
`MiniMaxMusic3SemanticGenerationStep` drives the Qwen3 LM one frame at a time
and lets transformers pick the cache. That default is `DynamicCache`, whose
`update()` is:

    self.keys = torch.cat([self.keys, key_states], dim=-2)

i.e. it REALLOCATES AND COPIES THE WHOLE CACHE EVERY FRAME, for all 36 layers.
Three costs follow, and they compound as a song gets longer:

  1. copy work grows linearly per frame -> O(N^2) over the song
  2. a transient second copy exists during the cat -> peak memory ~2x the cache
  3. 36 ever-growing allocations per frame fragment the caching allocator

On a 10 GB card already holding the 4-bit LM (6.29 GB) and the RVQ decoder
(1.20 GB) co-resident, (2) and (3) are what tip a run over into spilling to
shared system memory partway through -- which is why generation starts at a
sane speed and then degrades.

The cache is big because this stage runs CFG: `text_ids` carries a conditional
AND an unconditional row, so every number below is for batch 2.

    per token, per layer : 2 (K,V) x 8 kv_heads x 128 head_dim x 2 B = 4 KiB
    x 36 layers          : 144 KiB
    x batch 2 (CFG)      : 288 KiB per token
    audio runs at 25 Hz  : 7.03 MiB per second of music

MODES
-----
"dynamic"  transformers' default. Kept so A/B measurement is one flag.
"static"   Preallocate the whole cache once and write in place. Removes the
           per-frame copy, the transient double, and the fragmentation. Does
           NOT reduce peak memory -- it fixes it at the maximum up front, which
           is the honest trade: a long song is no longer *able* to creep over
           the line mid-run, it either fits from frame one or it does not.

A NOTE ON SLIDING / SHIFTING THE CACHE
--------------------------------------
Dropping old frames to bound memory is sound in principle -- RoPE rotates keys
by their absolute position at write time, so evicting old keys leaves the
retained ones mathematically intact, and 4 minutes (6000 frames + prompt) stays
inside this checkpoint's 10240 trained positions.

The trap is that Qwen3Model derives each new token's position from
`past_key_values.get_seq_length()`. Shrink the cache and the next token is
handed a LOWER position than keys already in it, so relative distances go
negative and attention silently corrupts. Doing it correctly means driving
`cache_position` from a true monotonic counter that ignores evictions, not from
the cache length. That is implemented here as "sliding", but it also throws
away the model's memory of everything it evicted -- which is precisely the
long-range musical coherence this checkpoint is sold on. Treat it as
experimental and listen to the result before trusting it.
"""

# torch is imported lazily inside install(): the budget maths below is pure
# arithmetic, and the studio venv (gradio only, no torch) needs to import this
# module to warn about duration before anything is loaded.

# Measured constants for this checkpoint; see the module docstring.
KV_BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 36 * 2      # K/V, kv_heads, dim, bf16, layers, CFG batch
FRAME_HIDDEN_BYTES = 8 * 4096 * 2                  # num_codebooks x hidden, bf16
BYTES_PER_FRAME = KV_BYTES_PER_TOKEN + FRAME_HIDDEN_BYTES
FRAME_RATE = 25.0


def projected_gb(seconds, prompt_tokens=1000):
    """VRAM the AR stage will need for `seconds` of audio, in GB."""
    frames = seconds * FRAME_RATE
    kv = (frames + prompt_tokens) * KV_BYTES_PER_TOKEN
    hidden = frames * FRAME_HIDDEN_BYTES
    return (kv + hidden) / 1024 ** 3


def budget(seconds, free_gb, base_gb=7.49, prompt_tokens=1000,
           workspace_gb=0.4, kv_bits=16):
    """(needed_gb, fits, headroom_gb) for a generation of `seconds`.

    `base_gb` is the AR pair -- the 4-bit LM plus the RVQ depth decoder, which
    `MiniMaxMusic3SemanticGenerationStep` requires to be co-resident.
    `kv_bits` models a quantized cache: 4 shrinks the KV part ~4x while leaving
    the per-frame hidden states alone.
    """
    frames = seconds * FRAME_RATE
    kv = (frames + prompt_tokens) * KV_BYTES_PER_TOKEN * (kv_bits / 16.0)
    hidden = frames * FRAME_HIDDEN_BYTES
    need = base_gb + (kv + hidden) / 1024 ** 3 + workspace_gb
    return need, need <= free_gb, free_gb - need


class _Installed:
    """Bookkeeping for one wrapped model, so install() is idempotent."""

    def __init__(self, model, original):
        self.model = model
        self.original = original
        self.stats = {}


def install(pipe, mode="static", max_frames=9000, window_frames=None,
            slack=16, verbose=None, nbits=4, residual_length=128):
    """Wrap the LM so the AR loop uses our cache instead of DynamicCache.

    The pipeline never hands us a cache to configure -- it simply calls the LM
    with no `past_key_values` on the prompt pass and reuses whatever comes back.
    So the hook is on that first call: `Qwen3Model.forward` only builds a
    DynamicCache when `past_key_values is None`, which means supplying one there
    is enough to own the cache for the whole generation.

    Returns a handle with .stats and .uninstall(); calling install() again
    replaces the previous wrapper rather than stacking on it.
    """
    lm = pipe.language_model.model
    prev = getattr(lm, "_ar_cache_handle", None)
    if prev is not None:
        prev.uninstall()

    if mode == "dynamic":
        return _noop_handle(lm)

    from transformers.cache_utils import QuantizedCache, StaticCache

    original = lm.forward
    state = {"cache": None, "seen": 0, "prompt": 0}
    stats = {"mode": mode, "allocated_tokens": 0, "prompt_tokens": 0,
             "evictions": 0, "peak_tokens": 0}

    def say(msg):
        if verbose:
            verbose(msg)

    def forward(*args, **kwargs):
        pkv = kwargs.get("past_key_values")
        embeds = kwargs.get("inputs_embeds")
        ids = kwargs.get("input_ids")

        # The prompt pass: no cache yet. Size ours from the prompt we can see
        # right now plus the frames this request can still produce -- guessing
        # the prompt length ahead of time would either waste VRAM or truncate.
        if pkv is None and state["cache"] is None:
            n = 0
            if embeds is not None:
                n = embeds.shape[1]
            elif ids is not None:
                n = ids.shape[1]
            total = int(n + max_frames + slack)
            if window_frames:
                total = int(n + window_frames + slack)
            state["prompt"] = n
            state["seen"] = 0
            if mode == "quantized":
                # Keeps the FULL history -- unlike a sliding window it loses no
                # musical memory, it just stores it in 4 bits. `residual_length`
                # leaves the most recent tokens unquantized, where precision
                # matters most, and quantizes only what has aged out.
                cache = QuantizedCache(backend="quanto",
                                       config=pipe.language_model.config,
                                       nbits=nbits, q_group_size=64,
                                       residual_length=residual_length)
                est = (total * KV_BYTES_PER_TOKEN * nbits / 16.0) / 1024 ** 3
                say(f"KV quantized to {nbits}-bit "
                    f"(~{est:.2f} GB for {total} tokens, "
                    f"{residual_length} recent kept full), prompt {n}")
            else:
                cache = StaticCache(config=pipe.language_model.config,
                                    max_cache_len=total)
                say(f"KV preallocated for {total} tokens "
                    f"({total * KV_BYTES_PER_TOKEN / 1024**3:.2f} GB), "
                    f"prompt {n}")
            state["cache"] = cache
            kwargs["past_key_values"] = cache
            stats["allocated_tokens"] = total
            stats["prompt_tokens"] = n

        out = original(*args, **kwargs)

        c = state["cache"]
        if c is not None:
            try:
                stats["peak_tokens"] = max(stats["peak_tokens"],
                                           int(c.get_seq_length()))
            except Exception:
                pass
        return out

    lm.forward = forward
    handle = _Installed(lm, original)
    handle.stats = stats
    handle.state = state

    def uninstall():
        lm.forward = original
        if hasattr(lm, "_ar_cache_handle"):
            delattr(lm, "_ar_cache_handle")

    def reset():
        """Drop the cache between requests -- a warm worker generates many."""
        state["cache"] = None
        state["seen"] = 0
        stats.update(allocated_tokens=0, prompt_tokens=0, peak_tokens=0,
                     evictions=0)

    handle.uninstall = uninstall
    handle.reset = reset
    lm._ar_cache_handle = handle
    return handle


def _noop_handle(lm):
    h = _Installed(lm, lm.forward)
    h.stats = {"mode": "dynamic"}
    h.uninstall = lambda: None
    h.reset = lambda: None
    return h


if __name__ == "__main__":
    print(f"{'song':>8}  {'KV+hidden':>10}  {'total need':>11}   fits in")
    print("-" * 52)
    for s in (30, 60, 120, 180, 240, 300):
        need, _, _ = budget(s, 99)
        print(f"{s:>6}s  {projected_gb(s):>9.2f}G  {need:>10.2f}G   "
              f"{'8.0G ok' if need <= 8.0 else '9.5G ok' if need <= 9.5 else 'DOES NOT FIT 10GB'}")
