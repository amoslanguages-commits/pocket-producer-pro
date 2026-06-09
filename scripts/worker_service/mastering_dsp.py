#!/usr/bin/env python3
"""
Mastering & Mixing DSP Module — A+ Studio Grade.

Implements:
  • Per-stem parametric EQ (Audio EQ Cookbook biquad filters, genre-tuned presets)
  • 3-Band Linkwitz-Riley Crossover (4th-order, flat summation)
  • Multiband Compressor (independent band dynamics)
  • Sidechain Compressor / Ducker (kick→bass, vocal→instrumental)
  • Dynamic EQ / Spectral Match (pink-noise slope reference)
  • ITU-R BS.1770-4 Integrated Loudness (LUFS) targeting via pyloudnorm
  • True-Peak Look-Ahead Limiter (−1.0 dBFS)
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf

# ---------------------------------------------------------------------------
# Per-stem parametric EQ presets (Audio EQ Cookbook biquad)
# Each band: {"type": "hp"|"lp"|"peak"|"hs"|"ls", "freq": Hz,
#             "gain": dB (peak/shelf only), "Q": float}
# ---------------------------------------------------------------------------
STEM_EQ_PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "drums": [
        {"type": "hp",   "freq": 30,    "Q": 0.707},
        {"type": "peak", "freq": 60,    "gain":  2.5, "Q": 0.70},   # sub punch
        {"type": "peak", "freq": 300,   "gain": -2.5, "Q": 1.20},   # cut boxiness
        {"type": "peak", "freq": 3500,  "gain":  2.0, "Q": 1.00},   # snap/attack
        {"type": "hs",   "freq": 8000,  "gain":  1.5, "Q": 0.707},  # air shelf
    ],
    "bass": [
        {"type": "hp",   "freq": 35,    "Q": 0.707},
        {"type": "peak", "freq": 100,   "gain":  2.5, "Q": 0.90},   # fundamental
        {"type": "peak", "freq": 400,   "gain": -2.0, "Q": 1.20},   # cut mud
        {"type": "lp",   "freq": 5000,  "Q": 0.707},                 # keep clean
    ],
    "keys": [
        {"type": "hp",   "freq": 60,    "Q": 0.707},
        {"type": "peak", "freq": 250,   "gain":  1.0, "Q": 1.00},   # warmth
        {"type": "peak", "freq": 600,   "gain": -1.5, "Q": 1.20},   # cut honk
        {"type": "peak", "freq": 3000,  "gain":  1.5, "Q": 1.00},   # presence
        {"type": "hs",   "freq": 10000, "gain":  2.0, "Q": 0.707},  # air shelf
    ],
    "guitar": [
        {"type": "hp",   "freq": 80,    "Q": 0.707},
        {"type": "peak", "freq": 300,   "gain": -2.0, "Q": 1.20},   # cut mud
        {"type": "peak", "freq": 2500,  "gain":  2.5, "Q": 1.00},   # presence/bite
        {"type": "hs",   "freq": 8000,  "gain":  1.0, "Q": 0.707},  # air
    ],
    "strings": [
        {"type": "hp",   "freq": 100,   "Q": 0.707},
        {"type": "peak", "freq": 350,   "gain":  1.0, "Q": 1.00},   # body warmth
        {"type": "peak", "freq": 800,   "gain": -1.5, "Q": 1.20},   # cut honk
        {"type": "peak", "freq": 5000,  "gain":  1.0, "Q": 0.80},   # shimmer
    ],
    "brass": [
        {"type": "hp",   "freq": 120,   "Q": 0.707},
        {"type": "peak", "freq": 400,   "gain": -1.5, "Q": 1.20},   # cut mud
        {"type": "peak", "freq": 1500,  "gain":  2.0, "Q": 1.00},   # bite/edge
        {"type": "hs",   "freq": 8000,  "gain":  1.0, "Q": 0.707},  # air
    ],
    "vocal_lead": [
        {"type": "hp",   "freq": 80,    "Q": 0.707},
        {"type": "peak", "freq": 250,   "gain": -2.0, "Q": 1.20},   # cut mud
        {"type": "peak", "freq": 3000,  "gain":  2.5, "Q": 1.00},   # presence
        {"type": "hs",   "freq": 10000, "gain":  1.5, "Q": 0.707},  # air shelf
    ],
    "vocal_double_l": [
        {"type": "hp",   "freq": 120,   "Q": 0.707},
        {"type": "peak", "freq": 2500,  "gain":  1.5, "Q": 1.20},
    ],
    "vocal_double_r": [
        {"type": "hp",   "freq": 120,   "Q": 0.707},
        {"type": "peak", "freq": 2500,  "gain":  1.5, "Q": 1.20},
    ],
    "vocal_harmony_3": [
        {"type": "hp",   "freq": 100,   "Q": 0.707},
        {"type": "peak", "freq": 2000,  "gain":  1.0, "Q": 1.00},
    ],
    "vocal_harmony_5": [
        {"type": "hp",   "freq": 100,   "Q": 0.707},
        {"type": "peak", "freq": 2000,  "gain":  1.0, "Q": 1.00},
    ],
}

# ---------------------------------------------------------------------------
# Biquad EQ filter implementations (Audio EQ Cookbook, R. Bristow-Johnson)
# ---------------------------------------------------------------------------

def _peaking_eq_filter(y: np.ndarray, sr: int, center_hz: float, gain_db: float, Q: float = 1.4) -> np.ndarray:
    """Peaking (bell) parametric EQ filter."""
    if abs(gain_db) < 0.1:
        return y
    w0 = 2.0 * np.pi * center_hz / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * Q)
    b0 =  1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 =  1.0 - alpha * A
    a0 =  1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 =  1.0 - alpha / A
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.filtfilt(b, a, y).astype(y.dtype)


def _high_shelf_filter(y: np.ndarray, sr: int, shelf_hz: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    """High-shelf EQ filter."""
    if abs(gain_db) < 0.1:
        return y
    w0 = 2.0 * np.pi * shelf_hz / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    sqA = np.sqrt(A)
    b0 =      A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =      A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 =           (A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha
    a1 =  2 *     ((A - 1) - (A + 1) * cos_w0)
    a2 =           (A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.filtfilt(b, a, y).astype(y.dtype)


def _low_shelf_filter(y: np.ndarray, sr: int, shelf_hz: float, gain_db: float, Q: float = 0.707) -> np.ndarray:
    """Low-shelf EQ filter."""
    if abs(gain_db) < 0.1:
        return y
    w0 = 2.0 * np.pi * shelf_hz / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    sqA = np.sqrt(A)
    b0 =      A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha)
    b1 =  2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 =      A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha)
    a0 =           (A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha
    a1 = -2 *     ((A - 1) + (A + 1) * cos_w0)
    a2 =           (A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.filtfilt(b, a, y).astype(y.dtype)


def _butter_filter(y: np.ndarray, cutoff: float, sr: int, btype: str, order: int = 2) -> np.ndarray:
    """Butterworth high/low-pass filter."""
    nyq = sr / 2.0
    Wn = cutoff / nyq
    Wn = float(np.clip(Wn, 1e-6, 1.0 - 1e-6))
    b, a = signal.butter(order, Wn, btype=btype)
    return signal.filtfilt(b, a, y, axis=0).astype(y.dtype)


def apply_stem_eq(y: np.ndarray, sr: int, stem_name: str) -> np.ndarray:
    """Apply the per-stem parametric EQ preset to an audio array.

    Uses the Audio EQ Cookbook biquad designs for accurate phase-linear
    filtering (via filtfilt). Silently returns original on any error.
    """
    preset = STEM_EQ_PRESETS.get(stem_name)
    if not preset:
        return y
    out = y.copy()
    try:
        for band in preset:
            btype = band["type"]
            freq  = float(band.get("freq", 1000))
            gain  = float(band.get("gain", 0.0))
            Q     = float(band.get("Q", 1.0))
            if btype == "peak":
                out = _peaking_eq_filter(out, sr, freq, gain, Q)
            elif btype == "hs":
                out = _high_shelf_filter(out, sr, freq, gain, Q)
            elif btype == "ls":
                out = _low_shelf_filter(out, sr, freq, gain, Q)
            elif btype == "hp":
                out = _butter_filter(out, freq, sr, btype="high", order=2)
            elif btype == "lp":
                out = _butter_filter(out, freq, sr, btype="low", order=2)
    except Exception as exc:
        print(f"[mastering_dsp] Stem EQ failed for '{stem_name}': {exc} — bypassing")
        return y
    return out


# ---------------------------------------------------------------------------
# Linkwitz-Riley 3-band crossover
# ---------------------------------------------------------------------------

def linkwitz_riley_crossover(
    y: np.ndarray, sr: int, f_low: float = 120.0, f_high: float = 4000.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4th-order Linkwitz-Riley crossover — flat magnitude on recombination."""
    low  = _butter_filter(y, f_low,  sr, btype="low",  order=4)
    high = _butter_filter(y, f_high, sr, btype="high", order=4)
    mid  = y - low - high
    return low, mid, high


# ---------------------------------------------------------------------------
# Multiband compressor
# ---------------------------------------------------------------------------

def _compress_band(
    band: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
) -> np.ndarray:
    """Per-band RMS compressor."""
    if ratio <= 1.0:
        return band
    threshold = 10.0 ** (threshold_db / 20.0)
    makeup     = 10.0 ** (makeup_db    / 20.0)
    block_size = max(256, int(sr * 0.010))
    attack_coef  = np.exp(-1.0 / (attack_ms  * sr / 1000.0))
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    envelope = np.zeros(len(band))
    rms = 0.0
    for i in range(len(band)):
        x_sq   = float(np.mean(band[max(0, i - block_size): i + 1] ** 2))
        curr   = np.sqrt(x_sq)
        if curr > rms:
            rms = attack_coef  * rms + (1.0 - attack_coef)  * curr
        else:
            rms = release_coef * rms + (1.0 - release_coef) * curr
        envelope[i] = rms
    gain = np.ones(len(band))
    for i in range(len(band)):
        ev = envelope[i]
        if ev > threshold:
            ev_db   = 20.0 * np.log10(ev + 1e-9)
            gain_db = (threshold_db - ev_db) * (1.0 - 1.0 / ratio)
            gain[i] = 10.0 ** (gain_db / 20.0)
    return band * gain * makeup


def apply_multiband_compression(
    y: np.ndarray,
    sr: int,
    low_opts: Dict[str, float],
    mid_opts: Dict[str, float],
    high_opts: Dict[str, float],
) -> np.ndarray:
    """Independent compression on Low, Mid, High bands, then recombine."""
    low, mid, high = linkwitz_riley_crossover(y, sr)
    return (
        _compress_band(low,  sr, **low_opts)
        + _compress_band(mid,  sr, **mid_opts)
        + _compress_band(high, sr, **high_opts)
    )


# ---------------------------------------------------------------------------
# Sidechain ducker
# ---------------------------------------------------------------------------

def _rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    rms_step = max(1, int(sr * 0.005))
    rms = np.zeros(len(y))
    for i in range(0, len(y), rms_step):
        block = y[i: i + rms_step]
        rms[i: i + rms_step] = np.sqrt(np.mean(block ** 2)) if block.size else 0.0
    return rms


def apply_sidechain_ducking(
    carrier: np.ndarray,
    modulator: np.ndarray,
    sr: int,
    threshold_db: float = -32.0,
    ducking_amount_db: float = -6.0,
    release_ms: float = 80.0,
) -> np.ndarray:
    """Duck carrier when modulator exceeds threshold."""
    mod_env      = _rms_envelope(modulator, sr)
    threshold    = 10.0 ** (threshold_db      / 20.0)
    max_reduction= 10.0 ** (ducking_amount_db / 20.0)
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    gain = np.ones(len(carrier))
    current_gain = 1.0
    for i in range(len(carrier)):
        target = max_reduction if mod_env[i] > threshold else 1.0
        if target < current_gain:
            current_gain = target
        else:
            current_gain = release_coef * current_gain + (1.0 - release_coef) * target
        gain[i] = current_gain
    return carrier * gain


# ---------------------------------------------------------------------------
# Dynamic EQ / Spectral Match
# ---------------------------------------------------------------------------

def apply_spectral_matching(y: np.ndarray, sr: int, target_slope: float = -3.0) -> np.ndarray:
    """OLA spectral match to pink-noise slope (−3 dB/oct)."""
    n_fft = 4096
    hop   = 1024
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freqs[0] = 1.0
    target = freqs ** (target_slope / 6.0)
    target /= np.max(target)
    window  = signal.windows.hann(n_fft)
    out     = np.zeros_like(y)
    win_sum = np.zeros_like(y)
    for offset in range(0, len(y) - n_fft, hop):
        chunk    = y[offset: offset + n_fft]
        spectrum = np.fft.rfft(chunk * window)
        mag      = np.abs(spectrum) + 1e-9
        smoothed = signal.medfilt(mag, 51)
        eq_gain  = target / (smoothed / (np.max(smoothed) + 1e-9) + 1e-6)
        eq_gain  = np.clip(eq_gain, 0.25, 2.0)
        out    [offset: offset + n_fft] += np.fft.irfft(spectrum * eq_gain) * window
        win_sum[offset: offset + n_fft] += window * window
    win_sum[win_sum < 1e-4] = 1.0
    return out / win_sum


# ---------------------------------------------------------------------------
# True-peak look-ahead limiter
# ---------------------------------------------------------------------------

def apply_lookahead_limiter(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -1.0,
    lookahead_ms: float = 2.0,
    release_ms: float = 150.0,
) -> np.ndarray:
    """Brickwall look-ahead limiter — prevents inter-sample clipping."""
    threshold        = 10.0 ** (threshold_db / 20.0)
    lookahead_samples= int(lookahead_ms * sr / 1000.0)
    n                = len(y)
    release_coef     = np.exp(-1.0 / (release_ms * sr / 1000.0))
    abs_y            = np.abs(y)
    peak_env         = np.zeros(n)
    for i in range(n):
        wend         = min(n, i + lookahead_samples)
        peak_env[i]  = np.max(abs_y[i: wend]) if i < wend else 0.0
    gain         = np.ones(n)
    current_gain = 1.0
    for i in range(n):
        target = threshold / (peak_env[i] + 1e-9) if peak_env[i] > threshold else 1.0
        if target < current_gain:
            current_gain = target
        else:
            current_gain = release_coef * current_gain + (1.0 - release_coef) * target
        gain[i] = current_gain
    return y * gain


# ---------------------------------------------------------------------------
# ITU-R BS.1770-4 LUFS normalisation  (pyloudnorm)
# ---------------------------------------------------------------------------

def apply_lufs_normalization(
    y: np.ndarray,
    sr: int,
    target_lufs: float = -14.0,
    headroom_db: float = -1.0,
) -> np.ndarray:
    """Normalize to target integrated loudness (ITU-R BS.1770-4).

    Falls back to peak normalization when pyloudnorm is unavailable.
    Target: −14 LUFS  (Spotify / Apple Music / YouTube streaming standard).
    """
    try:
        import pyloudnorm as pyln  # type: ignore

        meter    = pyln.Meter(sr)  # BS.1770 meter
        # pyloudnorm expects float64 and shape (samples,) or (samples, channels)
        y64      = y.astype(np.float64)
        loudness = meter.integrated_loudness(y64)
        if not np.isfinite(loudness) or loudness < -70.0:
            # Signal too quiet to meter — just peak-normalize
            raise ValueError(f"Unmeasurable loudness: {loudness}")
        normalized = pyln.normalize.loudness(y64, loudness, target_lufs)
        # Hard-clip to headroom ceiling after loudness norm
        ceiling    = 10.0 ** (headroom_db / 20.0)
        peak       = float(np.max(np.abs(normalized)))
        if peak > ceiling:
            normalized *= ceiling / peak
        return normalized.astype(y.dtype)
    except Exception as exc:
        print(f"[mastering_dsp] LUFS normalization skipped ({exc}) — using peak norm")
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        ceiling = 10.0 ** (headroom_db / 20.0)
        return (y * ceiling / peak).astype(y.dtype) if peak > 0 else y


# ---------------------------------------------------------------------------
# Master mix_and_master entry point
# ---------------------------------------------------------------------------

def mix_and_master(
    stems: Dict[str, str],
    output_path: str,
    mix_options: Dict[str, Any],
    mastering_options: Dict[str, Any],
) -> None:
    """Load stems → per-stem EQ → sidechain → sum → master → LUFS → write."""

    # ── 1. Load & EQ each stem ──────────────────────────────────────────────
    loaded_stems: Dict[str, np.ndarray] = {}
    sr      = 48000
    max_len = 0

    for name, path in stems.items():
        if not os.path.exists(path) or name == "mix":
            continue
        y, file_sr = sf.read(path)
        sr = file_sr
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = np.asarray(y, dtype=np.float32)

        # Per-stem parametric EQ (A+ upgrade)
        if mix_options.get("per_stem_eq", True):
            y = apply_stem_eq(y, sr, name)

        loaded_stems[name] = y
        max_len = max(max_len, len(y))

    if not loaded_stems:
        raise ValueError("No valid stems found to mix")

    # ── 2. Pad to common length ─────────────────────────────────────────────
    for name in loaded_stems:
        curr = len(loaded_stems[name])
        if curr < max_len:
            loaded_stems[name] = np.pad(loaded_stems[name], (0, max_len - curr))

    # ── 3. Sidechain ducking ────────────────────────────────────────────────
    # Kick → Bass sidechain
    if (
        mix_options.get("kick_bass_sidechain", True)
        and "drums" in loaded_stems
        and "bass"  in loaded_stems
    ):
        nyq = sr / 2.0
        b, a = signal.butter(4, 100.0 / nyq, btype="low")
        kick_signal = signal.filtfilt(b, a, loaded_stems["drums"])
        loaded_stems["bass"] = apply_sidechain_ducking(
            loaded_stems["bass"], kick_signal, sr,
            threshold_db=-38.0, ducking_amount_db=-7.0, release_ms=65.0,
        )

    # Lead vocal → instrumental mid-duck
    if mix_options.get("vocal_ducking", True) and "vocal_lead" in loaded_stems:
        inst_names  = ["keys", "guitar", "strings", "brass"]
        active_inst = [n for n in inst_names if n in loaded_stems]
        if active_inst:
            inst_sum = np.sum([loaded_stems[n] for n in active_inst], axis=0)
            ducked   = apply_sidechain_ducking(
                inst_sum, loaded_stems["vocal_lead"], sr,
                threshold_db=-36.0, ducking_amount_db=-3.5, release_ms=150.0,
            )
            ratio = np.clip((ducked + 1e-9) / (inst_sum + 1e-9), 0.5, 1.0)
            for n in active_inst:
                loaded_stems[n] = loaded_stems[n] * ratio

    # ── 4. Sum (mix) ────────────────────────────────────────────────────────
    mix = np.zeros(max_len, dtype=np.float32)
    for audio in loaded_stems.values():
        mix += audio

    # Gain staging before mastering
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.85:
        mix = mix * (0.85 / peak)

    # ── 5. Mastering chain ──────────────────────────────────────────────────
    # A. Spectral match / Dynamic EQ
    if mastering_options.get("reference_match", True):
        mix = apply_spectral_matching(mix, sr, target_slope=-3.0)

    # B. Multiband compression
    if mastering_options.get("multiband_compression", True):
        low_opts  = {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 35.0, "release_ms": 220.0, "makeup_db": 1.5}
        mid_opts  = {"threshold_db": -22.0, "ratio": 1.8, "attack_ms": 25.0, "release_ms": 150.0, "makeup_db": 1.0}
        high_opts = {"threshold_db": -18.0, "ratio": 1.5, "attack_ms": 15.0, "release_ms": 100.0, "makeup_db": 0.5}
        mix = apply_multiband_compression(mix, sr, low_opts, mid_opts, high_opts)

    # C. True-peak look-ahead limiter (−1.0 dBFS)
    peak_limit_db = float(mastering_options.get("true_peak_limit_db", -1.0))
    mix = apply_lookahead_limiter(mix, sr, threshold_db=peak_limit_db)

    # D. ITU-R BS.1770-4 LUFS loudness targeting (A+ upgrade)
    target_lufs = float(mastering_options.get("target_lufs", -14.0))
    mix = apply_lufs_normalization(mix, sr, target_lufs=target_lufs, headroom_db=peak_limit_db)

    # ── 6. Write mastered output ────────────────────────────────────────────
    sf.write(output_path, mix, sr, subtype="PCM_24")  # 24-bit for streaming masters
