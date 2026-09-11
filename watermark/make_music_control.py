"""
Synthesise a music-like control with KNOWN human (well, algorithmic) origin.

Why this exists: the only non-AI control available was a sine sweep plus white
noise. That is spectrally nothing like music -- it is rough at every frequency,
which is exactly the property the fakeprint detector reads as "real". Comparing
a produced track against it is meaningless.

This builds something with the features that actually matter: harmonic series
with realistic roll-off, percussive transients, dense noise-like top end from
hats and cymbals, and reverb tails. No neural vocoder is involved at any point,
so anything the AI detector says about it is by definition a false positive.

Emits the chain stage by stage, so you can see which processing step (if any)
moves the detector:

    raw mix -> compressed/limited to streaming loudness -> 320 kbps MP3

Usage:
    python make_music_control.py -o outdir --seconds 40
"""

import argparse
import os
import subprocess

import numpy as np
import soundfile as sf

SR = 48000


def adsr(n, a, d, s, r, sr=SR):
    a, d, r = int(a * sr), int(d * sr), int(r * sr)
    s_n = max(0, n - a - d - r)
    return np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, s, d, endpoint=False),
        np.full(s_n, s),
        np.linspace(s, 0, r),
    ])[:n]


def saw(f, n, sr=SR, partials=40):
    t = np.arange(n) / sr
    out = np.zeros(n)
    for k in range(1, partials + 1):
        if f * k > sr / 2 * 0.95:
            break
        out += np.sin(2 * np.pi * f * k * t) / k
    return out


def onepole_lp(x, cutoff, sr=SR):
    a = np.exp(-2 * np.pi * cutoff / sr)
    y = np.zeros_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def noise_burst(n, decay, hp=None, rng=None, sr=SR):
    rng = rng or np.random.RandomState(0)
    x = rng.randn(n) * np.exp(-np.arange(n) / (decay * sr))
    if hp:  # crude high-pass: subtract a smoothed copy
        k = max(1, int(sr / hp))
        x = x - np.convolve(x, np.ones(k) / k, mode="same")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--bpm", type=float, default=120.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.RandomState(7)
    n = int(args.seconds * SR)
    beat = 60.0 / args.bpm
    L = np.zeros(n)
    R = np.zeros(n)

    # --- chords: detuned saws, filtered. Gives dense harmonic structure.
    prog = [[110.0, 138.6, 164.8], [98.0, 123.5, 146.8],
            [87.3, 110.0, 130.8], [98.0, 123.5, 155.6]]
    bar = int(beat * 4 * SR)
    for bi in range(0, n, bar):
        chord = prog[(bi // bar) % len(prog)]
        ln = min(bar, n - bi)
        env = adsr(ln, 0.01, 0.25, 0.65, 0.35)
        for f in chord:
            for det, pan in ((1.0, 0.0), (1.004, -0.7), (0.996, 0.7)):
                v = saw(f * det, ln) * env * 0.05
                v = onepole_lp(v, 3200)
                L[bi:bi + ln] += v * (1 - max(0.0, pan))
                R[bi:bi + ln] += v * (1 + min(0.0, pan))

    # --- bass
    for bi in range(0, n, int(beat * SR)):
        ln = min(int(beat * SR), n - bi)
        f = prog[(bi // bar) % len(prog)][0] / 2
        v = saw(f, ln, partials=18) * adsr(ln, 0.005, 0.15, 0.5, 0.2) * 0.16
        v = onepole_lp(v, 900)
        L[bi:bi + ln] += v
        R[bi:bi + ln] += v

    # --- kick
    for bi in range(0, n, int(beat * SR)):
        ln = min(int(0.28 * SR), n - bi)
        t = np.arange(ln) / SR
        f = 110 * np.exp(-t * 28) + 44
        v = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11) * 0.5
        L[bi:bi + ln] += v
        R[bi:bi + ln] += v

    # --- snare on 2 and 4
    for bi in range(int(beat * SR), n, int(beat * 2 * SR)):
        ln = min(int(0.25 * SR), n - bi)
        v = noise_burst(ln, 0.09, hp=900, rng=rng) * 0.30
        v += np.sin(2 * np.pi * 190 * np.arange(ln) / SR) * np.exp(
            -np.arange(ln) / (0.05 * SR)) * 0.10
        L[bi:bi + ln] += v * 0.95
        R[bi:bi + ln] += v * 1.05

    # --- hats: the dense, noise-like top end real mixes have
    for bi in range(0, n, int(beat * SR / 2)):
        ln = min(int(0.09 * SR), n - bi)
        v = noise_burst(ln, 0.022, hp=5500, rng=rng) * 0.16
        pan = 0.25 if (bi // int(beat * SR / 2)) % 2 else -0.25
        L[bi:bi + ln] += v * (1 - pan)
        R[bi:bi + ln] += v * (1 + pan)

    # --- reverb: exponentially decaying noise IR, adds dense diffuse HF
    ir_n = int(1.5 * SR)
    ir = rng.randn(ir_n) * np.exp(-np.arange(ir_n) / (0.32 * SR))
    ir[:int(0.012 * SR)] = 0
    ir /= np.abs(ir).sum() / 6
    for ch in (0, 1):
        src = L if ch == 0 else R
        wet = np.convolve(src, ir * (1.0 if ch == 0 else 0.93), mode="full")[:n]
        if ch == 0:
            L = src + 0.16 * wet
        else:
            R = src + 0.16 * wet

    mix = np.stack([L, R], axis=1)
    mix /= np.abs(mix).max() / 0.6

    raw = os.path.join(args.outdir, "musiccontrol_1_raw.wav")
    sf.write(raw, mix, SR, subtype="PCM_24")

    # --- stage 2: bus compression + limiting to streaming loudness
    x = mix.copy()
    env = np.abs(x).max(axis=1)
    k = int(0.03 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same") + 1e-9
    thr = np.percentile(env, 65)
    ratio = 3.5
    gain = np.where(env > thr, (thr + (env - thr) / ratio) / env, 1.0)
    x = x * gain[:, None]
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(SR)
        cur = meter.integrated_loudness(x)
        x = x * 10 ** ((-14.0 - cur) / 20)
    except Exception:
        pass
    peak = np.abs(x).max()
    if peak > 0.89:
        x = x / peak * 0.89
    mastered = os.path.join(args.outdir, "musiccontrol_2_mastered.wav")
    sf.write(mastered, x, SR, subtype="PCM_24")

    # --- stage 3: 320 kbps MP3 round trip, matching a distributed master
    mp3 = os.path.join(args.outdir, "_tmp.mp3")
    enc = os.path.join(args.outdir, "musiccontrol_3_mp3_320.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mastered,
                    "-c:a", "libmp3lame", "-b:a", "320k", mp3], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
                    "-ar", "48000", "-c:a", "pcm_s24le", enc], check=True)
    os.remove(mp3)

    for p in (raw, mastered, enc):
        print("wrote", p)


if __name__ == "__main__":
    main()
