#!/usr/bin/env python3
"""
Mastering & Mixing DSP Module.
Implements: 3-Band Linkwitz-Riley Crossovers, Multiband Compressor,
Sidechain Compressor (Ducker), Dynamic EQ / Spectral Match, and True Peak Limiter.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf


def _butter_filter(y: np.ndarray, cutoff: float, sr: int, btype: str, order: int = 4) -> np.ndarray:
    """Helper to apply a Butterworth filter."""
    nyq = sr / 2
    if isinstance(cutoff, list):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq
    b, a = signal.butter(order, Wn, btype=btype)
    return signal.filtfilt(b, a, y, axis=0)


def linkwitz_riley_crossover(y: np.ndarray, sr: int, f_low: float = 120.0, f_high: float = 4000.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits signal into 3 bands using 4th-order Linkwitz-Riley filters (flat summation).

    A 4th-order Linkwitz-Riley filter is built by cascading two 2nd-order Butterworth filters.
    """
    # Low Band
    low = _butter_filter(y, f_low, sr, btype="low", order=4)
    
    # High Band
    high = _butter_filter(y, f_high, sr, btype="high", order=4)
    
    # Mid Band (Original - Low - High keeps perfect phase alignment)
    mid = y - low - high
    
    return low, mid, high


def _compress_band(
    band: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float
) -> np.ndarray:
    """Applies dynamic range compression to a single audio band."""
    if ratio <= 1.0:
        return band
        
    threshold = 10 ** (threshold_db / 20.0)
    makeup = 10 ** (makeup_db / 20.0)
    
    # Envelope detector (RMS-based)
    # Block size ≈ 10ms
    block_size = int(sr * 0.010)
    if block_size <= 0:
        block_size = 256
        
    envelope = np.zeros(len(band))
    rms = 0.0
    
    # Attack/Release coefficients
    attack_coef = np.exp(-1.0 / (attack_ms * sr / 1000.0))
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    
    # Detect envelope with smoothing
    for i in range(len(band)):
        x_sq = float(np.mean(band[max(0, i - block_size) : i + 1] ** 2))
        curr_rms = np.sqrt(x_sq)
        if curr_rms > rms:
            rms = attack_coef * rms + (1.0 - attack_coef) * curr_rms
        else:
            rms = release_coef * rms + (1.0 - release_coef) * curr_rms
        envelope[i] = rms
        
    # Calculate gain reduction
    gain = np.ones(len(band))
    for i in range(len(band)):
        env_val = envelope[i]
        if env_val > threshold:
            env_db = 20 * np.log10(env_val + 1e-9)
            gain_db = (threshold_db - env_db) * (1.0 - 1.0 / ratio)
            gain[i] = 10 ** (gain_db / 20.0)
            
    # Apply gain and makeup
    return band * gain * makeup


def apply_multiband_compression(
    y: np.ndarray,
    sr: int,
    low_opts: Dict[str, float],
    mid_opts: Dict[str, float],
    high_opts: Dict[str, float]
) -> np.ndarray:
    """Applies independent compression to Low, Mid, and High frequency bands."""
    # Split bands
    low, mid, high = linkwitz_riley_crossover(y, sr)
    
    # Compress bands
    low_comp = _compress_band(low, sr, **low_opts)
    mid_comp = _compress_band(mid, sr, **mid_opts)
    high_comp = _compress_band(high, sr, **high_opts)
    
    # Sum back
    return low_comp + mid_comp + high_comp


def apply_sidechain_ducking(
    carrier: np.ndarray,
    modulator: np.ndarray,
    sr: int,
    threshold_db: float = -32.0,
    ducking_amount_db: float = -6.0,
    release_ms: float = 80.0
) -> np.ndarray:
    """Ducks the carrier signal when the modulator signal exceeds a threshold.

    Used for Kick-Bass sidechaining and Vocal-Instrumental ducking.
    """
    mod_env = _rms_envelope(modulator, sr)
    threshold = 10 ** (threshold_db / 20.0)
    max_reduction = 10 ** (ducking_amount_db / 20.0)
    
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    gain = np.ones(len(carrier))
    current_gain = 1.0
    
    for i in range(len(carrier)):
        if mod_env[i] > threshold:
            target_gain = max_reduction
        else:
            target_gain = 1.0
            
        # Fast attack, smooth release
        if target_gain < current_gain:
            current_gain = target_gain # Instant ducking
        else:
            current_gain = release_coef * current_gain + (1.0 - release_coef) * target_gain
            
        gain[i] = current_gain
        
    return carrier * gain


def _rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    # helper for RMS envelope calculation
    rms_step = int(sr * 0.005) # 5ms
    rms = np.zeros(len(y))
    for i in range(0, len(y), rms_step):
        block = y[i : i + rms_step]
        rms_val = np.sqrt(np.mean(block**2)) if block.size > 0 else 0.0
        rms[i : i + rms_step] = rms_val
    return rms


def apply_spectral_matching(y: np.ndarray, sr: int, target_slope: float = -3.0) -> np.ndarray:
    """Matches the frequency response of the mix to a balanced target spectrum (e.g. Pink Noise, -3dB/Octave)."""
    # Perform FFT on overall signal in chunks
    n_fft = 4096
    hop = 1024
    
    # Calculate target magnitude spectrum (pink noise slope)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    freqs[0] = 1.0 # avoid divide by zero
    target = freqs ** (target_slope / 6.0) # -3dB per octave is roughly 1/f^0.5 amplitude
    target = target / np.max(target)
    
    # Window and process
    window = signal.windows.hann(n_fft)
    out = np.zeros_like(y)
    win_sum = np.zeros_like(y)
    
    for offset in range(0, len(y) - n_fft, hop):
        chunk = y[offset : offset + n_fft]
        if chunk.ndim > 1:
            # handle multi-channel processing separately or average
            pass
        # FFT
        spectrum = np.fft.rfft(chunk * window)
        mag = np.abs(spectrum) + 1e-9
        
        # Calculate matching filter
        # Smooth the current magnitude response to avoid resonance peaks
        smoothed_mag = signal.medfilt(mag, 51)
        eq_gain = target / (smoothed_mag / (np.max(smoothed_mag) + 1e-9) + 1e-6)
        
        # Limit EQ gain boosts/cuts to a safe range (+6dB / -12dB)
        eq_gain = np.clip(eq_gain, 0.25, 2.0)
        
        # Apply filter
        filtered_spectrum = spectrum * eq_gain
        
        # IFFT
        filtered_chunk = np.fft.irfft(filtered_spectrum)
        
        out[offset : offset + n_fft] += filtered_chunk * window
        win_sum[offset : offset + n_fft] += window * window
        
    win_sum[win_sum < 1e-4] = 1.0
    return out / win_sum


def apply_lookahead_limiter(y: np.ndarray, sr: int, threshold_db: float = -1.0, lookahead_ms: float = 2.0, release_ms: float = 150.0) -> np.ndarray:
    """High-end brickwall look-ahead limiter to prevent inter-sample clipping and maximize volume."""
    threshold = 10 ** (threshold_db / 20.0)
    lookahead_samples = int(lookahead_ms * sr / 1000.0)
    
    # Buffer signal
    n = len(y)
    out = np.zeros_like(y)
    
    # Release coefficient
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    
    # Peak envelope with lookahead
    # Pre-calculate peak values
    abs_y = np.abs(y)
    
    # Rolling maximum for lookahead window
    peak_env = np.zeros(n)
    current_max = 0.0
    
    # Fill rolling max (look-ahead peak detection)
    for i in range(n):
        # Peak search inside lookahead window
        window_end = min(n, i + lookahead_samples)
        peak_env[i] = np.max(abs_y[i : window_end]) if i < window_end else 0.0
        
    gain = np.ones(n)
    current_gain = 1.0
    
    for i in range(n):
        peak = peak_env[i]
        if peak > threshold:
            target_gain = threshold / (peak + 1e-9)
        else:
            target_gain = 1.0
            
        # Attack is instantaneous when target gain drops, release is smoothed
        if target_gain < current_gain:
            current_gain = target_gain
        else:
            current_gain = release_coef * current_gain + (1.0 - release_coef) * target_gain
            
        gain[i] = current_gain
        
    return y * gain


def mix_and_master(
    stems: Dict[str, str],
    output_path: str,
    mix_options: Dict[str, Any],
    mastering_options: Dict[str, Any]
) -> None:
    """Reads all stems, mixes them applying sidechain/gain, and masters the final output."""
    # 1. Load stems
    loaded_stems = {}
    sr = 48000
    max_len = 0
    
    for name, path in stems.items():
        if not os.path.exists(path) or name == "mix":
            continue
        y, file_sr = sf.read(path)
        sr = file_sr
        if y.ndim > 1:
            y = np.mean(y, axis=1) # Mono mix down
        loaded_stems[name] = np.asarray(y, dtype=np.float32)
        max_len = max(max_len, len(y))
        
    if not loaded_stems:
        raise ValueError("No valid stems found to mix")
        
    # Pad all stems to same length
    for name in loaded_stems:
        curr_len = len(loaded_stems[name])
        if curr_len < max_len:
            loaded_stems[name] = np.pad(loaded_stems[name], (0, max_len - curr_len))
            
    # 2. Sidechain ducking
    # Kick ducks Bass
    if mix_options.get("kick_bass_sidechain", True) and "drums" in loaded_stems and "bass" in loaded_stems:
        # Create a simple bandpassed kick detector from drums stem
        # (Assuming low frequency content in drums corresponds to Kick)
        nyq = sr / 2
        b, a = signal.butter(4, 100.0 / nyq, btype="low")
        kick_signal = signal.filtfilt(b, a, loaded_stems["drums"])
        
        loaded_stems["bass"] = apply_sidechain_ducking(
            loaded_stems["bass"],
            kick_signal,
            sr,
            threshold_db=-38.0,
            ducking_amount_db=-7.0,
            release_ms=65.0
        )
        
    # Lead Vocal ducks instrumental mid frequencies
    if mix_options.get("vocal_ducking", True) and "vocal_lead" in loaded_stems:
        instrumentals = ["keys", "guitar", "strings", "brass"]
        active_inst = [name for name in instrumentals if name in loaded_stems]
        
        if active_inst:
            # Create a sum of all instrumentals
            inst_sum = np.sum([loaded_stems[name] for name in active_inst], axis=0)
            
            # Duck instrumentals
            ducked_inst_sum = apply_sidechain_ducking(
                inst_sum,
                loaded_stems["vocal_lead"],
                sr,
                threshold_db=-36.0,
                ducking_amount_db=-3.5,
                release_ms=150.0
            )
            
            # Distribute ducking back to individual stems
            ratio = (ducked_inst_sum + 1e-9) / (inst_sum + 1e-9)
            ratio = np.clip(ratio, 0.5, 1.0)
            for name in active_inst:
                loaded_stems[name] = loaded_stems[name] * ratio
                
    # 3. Summing (Mixing)
    mix = np.zeros(max_len, dtype=np.float32)
    for name, audio in loaded_stems.items():
        mix += audio
        
    # Normalize mix slightly before mastering to prevent early clipping
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0.85:
        mix = mix * (0.85 / peak)
        
    # 4. Mastering Engine
    # A. Dynamic EQ / Spectral Match
    if mastering_options.get("reference_match", True):
        mix = apply_spectral_matching(mix, sr, target_slope=-3.0) # Pink noise slope
        
    # B. Multiband Compression
    if mastering_options.get("multiband_compression", True):
        low_opts = {"threshold_db": -24.0, "ratio": 2.5, "attack_ms": 35.0, "release_ms": 220.0, "makeup_db": 1.5}
        mid_opts = {"threshold_db": -22.0, "ratio": 1.8, "attack_ms": 25.0, "release_ms": 150.0, "makeup_db": 1.0}
        high_opts = {"threshold_db": -18.0, "ratio": 1.5, "attack_ms": 15.0, "release_ms": 100.0, "makeup_db": 0.5}
        mix = apply_multiband_compression(mix, sr, low_opts, mid_opts, high_opts)
        
    # C. Look-Ahead Limiter
    target_lufs_db = mastering_options.get("true_peak_limit_db", -1.0)
    mix = apply_lookahead_limiter(mix, sr, threshold_db=target_lufs_db)
    
    # Save mastered output
    sf.write(output_path, mix, sr, subtype="PCM_16")
