"""
What has been done to this audio? A processing-history profile.

Reports the things that leave measurable traces: loudness/limiting, clipping,
codec cutoff, encoder frame periodicity, stereo manipulation, spectral gating,
and any regular amplitude modulation of the sort a crude watermark would add.

The point is to separate "ordinary mastering and lossy delivery" from "something
was deliberately injected", which look nothing alike once measured.

Usage:
    python audio_forensics.py track.wav [more.wav ...]
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


def db(x):
    return 20 * np.log10(np.abs(x) + 1e-12)


def loudness_profile(x, sr):
    out = {}
    mono = x.mean(axis=1)
    peak = np.abs(x).max()
    rms = np.sqrt((mono ** 2).mean())
    out["sample_peak_dbfs"] = float(db(peak))
    out["rms_dbfs"] = float(db(rms))
    out["crest_db"] = float(db(peak) - db(rms))

    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        out["lufs_integrated"] = float(meter.integrated_loudness(x))
    except Exception:
        out["lufs_integrated"] = None

    # Short-term RMS spread = how much dynamic range survives. Heavy limiting
    # collapses this.
    win = int(0.4 * sr)
    n = len(mono) // win
    if n > 4:
        st = np.array([np.sqrt((mono[i*win:(i+1)*win] ** 2).mean())
                       for i in range(n)])
        st = db(st[st > 0])
        out["shortterm_rms_p10_p90_db"] = float(np.percentile(st, 90)
                                                - np.percentile(st, 10))
    else:
        out["shortterm_rms_p10_p90_db"] = None

    # Clipping: runs of consecutive samples pinned at/near full scale.
    thr = 0.9989
    flat = np.abs(x).max(axis=1)
    at_ceiling = flat >= thr
    runs, cur = [], 0
    for v in at_ceiling:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    out["samples_at_ceiling"] = int(at_ceiling.sum())
    out["clipped_runs_ge3"] = int(sum(1 for r in runs if r >= 3))
    return out


def spectral_profile(x, sr):
    out = {}
    mono = x.mean(axis=1)
    n = 8192
    win = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    cnt = 0
    for s in range(0, len(mono) - n, max(n, (len(mono) - n) // 400)):
        acc += np.abs(np.fft.rfft(mono[s:s + n] * win))
        cnt += 1
    acc /= max(cnt, 1)
    fr = np.fft.rfftfreq(n, 1 / sr)
    d = 20 * np.log10(acc / acc.max() + 1e-12)

    sm = np.convolve(d, np.ones(9) / 9, mode="same")
    grad = np.diff(sm)
    band = fr[:-1] > 6000
    i = int(np.argmin(np.where(band, grad, 0)))
    out["cliff_hz"] = float(fr[i])
    lo = d[(fr >= fr[i] - 1500) & (fr < fr[i])]
    hi = d[(fr > fr[i]) & (fr <= fr[i] + 1500)]
    out["cliff_drop_db"] = float(lo.mean() - hi.mean()) if len(lo) and len(hi) else 0.0

    # Narrow notches: a crude watermark often carves fixed narrow bands.
    med = np.convolve(d, np.ones(31) / 31, mode="same")
    dip = med - d
    inband = (fr > 500) & (fr < min(sr / 2 - 500, 20000))
    out["deepest_notch_db"] = float(dip[inband].max())
    out["notch_freq_hz"] = float(fr[inband][np.argmax(dip[inband])])
    out["n_notches_over_9db"] = int((dip[inband] > 9).sum())
    return out


def stereo_profile(x, sr):
    out = {}
    if x.shape[1] < 2:
        out["mode"] = "mono"
        return out
    L, R = x[:, 0], x[:, 1]
    out["lr_correlation"] = float(np.corrcoef(L, R)[0, 1])
    mid, side = (L + R) / 2, (L - R) / 2
    out["side_to_mid_db"] = float(db(np.sqrt((side ** 2).mean()))
                                  - db(np.sqrt((mid ** 2).mean())))
    # Fully mono above some frequency is a codec (joint-stereo) tell.
    n = 8192
    win = np.hanning(n)
    fr = np.fft.rfftfreq(n, 1 / sr)
    sides, mids = np.zeros(n // 2 + 1), np.zeros(n // 2 + 1)
    cnt = 0
    for s in range(0, len(mid) - n, max(n, (len(mid) - n) // 300)):
        sides += np.abs(np.fft.rfft(side[s:s + n] * win))
        mids += np.abs(np.fft.rfft(mid[s:s + n] * win))
        cnt += 1
    ratio = 20 * np.log10((sides / max(cnt, 1)) / ((mids / max(cnt, 1)) + 1e-12) + 1e-12)
    collapsed = fr[(ratio < -40) & (fr > 1000)]
    out["stereo_collapse_above_hz"] = float(collapsed.min()) if len(collapsed) else None
    return out


def modulation_profile(x, sr):
    """Regular amplitude modulation, and periodicity at codec frame sizes.

    A crude watermark that ducks or pulses the signal shows up as a spectral
    line in the amplitude envelope. Lossy codecs impose block structure at
    known frame lengths (MP3 1152, AAC 1024 samples).
    """
    out = {}
    mono = x.mean(axis=1)
    env = np.abs(mono)
    k = max(1, int(sr * 0.002))
    env = np.convolve(env, np.ones(k) / k, mode="same")[::k]
    env = env - env.mean()
    if len(env) < 1024:
        return out
    E = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    ef = np.fft.rfftfreq(len(env), k / sr)
    sel = (ef > 0.3) & (ef < 60)
    Es, fs = E[sel], ef[sel]
    med = np.median(Es)
    out["env_peak_ratio"] = float(Es.max() / (med + 1e-12))
    out["env_peak_hz"] = float(fs[np.argmax(Es)])

    for name, frame in (("mp3_1152", 1152), ("aac_1024", 1024)):
        m = mono[: (len(mono) // frame) * frame].reshape(-1, frame)
        if len(m) < 8:
            continue
        prof = np.abs(m).mean(axis=0)
        out[f"{name}_frame_ripple_db"] = float(db(prof.max()) - db(prof.min()))
    return out


def analyse(path):
    x, sr = sf.read(path, always_2d=True, dtype="float64")
    r = {"file": os.path.basename(path), "sr": sr, "ch": x.shape[1],
         "dur_s": round(len(x) / sr, 1)}
    r.update(loudness_profile(x, sr))
    r.update(spectral_profile(x, sr))
    r.update(stereo_profile(x, sr))
    r.update(modulation_profile(x, sr))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        r = analyse(f)
        print(f"\n=== {r['file']}  ({r['sr']} Hz, {r['ch']} ch, {r['dur_s']}s)")
        print("  LOUDNESS / DYNAMICS")
        print(f"    integrated LUFS      : {r['lufs_integrated']}")
        print(f"    sample peak          : {r['sample_peak_dbfs']:.2f} dBFS")
        print(f"    crest factor         : {r['crest_db']:.2f} dB "
              f"(<10 = heavily limited)")
        print(f"    short-term RMS spread: {r['shortterm_rms_p10_p90_db']} dB "
              f"(<6 = very compressed)")
        print(f"    samples at ceiling   : {r['samples_at_ceiling']} "
              f"({r['clipped_runs_ge3']} runs >=3 -> hard clipping)")
        print("  SPECTRUM")
        print(f"    codec cliff          : {r['cliff_hz']:.0f} Hz "
              f"(drop {r['cliff_drop_db']:.1f} dB)")
        print(f"    deepest notch        : {r['deepest_notch_db']:.1f} dB "
              f"at {r['notch_freq_hz']:.0f} Hz "
              f"({r['n_notches_over_9db']} bins >9 dB)")
        print("  STEREO")
        for k in ("lr_correlation", "side_to_mid_db", "stereo_collapse_above_hz"):
            if k in r:
                print(f"    {k:<21}: {r[k]}")
        print("  MODULATION / FRAMING")
        print(f"    envelope line        : {r.get('env_peak_ratio', float('nan')):.1f}x "
              f"median at {r.get('env_peak_hz', float('nan')):.2f} Hz "
              f"(>8x = strong periodic pumping)")
        for k in ("mp3_1152_frame_ripple_db", "aac_1024_frame_ripple_db"):
            if k in r:
                print(f"    {k:<21}: {r[k]:.2f} dB")
    print()


if __name__ == "__main__":
    sys.exit(main())
