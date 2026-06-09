#!/usr/bin/env python3
"""
Vocal Production & Alignment DSP Module.
Implements: Gate, De-esser, Pitch Correction (Pro Pitch Lock), Timing Alignment,
Double-tracking, and Harmonization.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf


def _rms_envelope(y: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Computes a smooth RMS amplitude envelope."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Interpolate back to original length
    x_old = np.linspace(0, len(y), len(rms))
    x_new = np.arange(len(y))
    return np.interp(x_new, x_old, rms)


def apply_gate(y: np.ndarray, threshold_db: float = -45.0, attack_ms: float = 10.0, release_ms: float = 120.0, sr: int = 48000) -> np.ndarray:
    """Applies a smooth noise gate based on threshold DB."""
    envelope = _rms_envelope(y)
    threshold = 10 ** (threshold_db / 20.0)
    
    # Calculate envelope filter coefficients
    attack_coef = np.exp(-1.0 / (attack_ms * sr / 1000.0))
    release_coef = np.exp(-1.0 / (release_ms * sr / 1000.0))
    
    gain = np.zeros_like(y)
    current_gain = 1.0
    
    for i in range(len(y)):
        target_gain = 1.0 if envelope[i] >= threshold else 0.0
        if target_gain > current_gain:
            current_gain = attack_coef * current_gain + (1.0 - attack_coef) * target_gain
        else:
            current_gain = release_coef * current_gain + (1.0 - release_coef) * target_gain
        gain[i] = current_gain
        
    return y * gain


def apply_deesser(y: np.ndarray, sr: int = 48000, f_low: float = 5000.0, f_high: float = 9000.0, threshold_db: float = -38.0) -> np.ndarray:
    """Applies a de-esser by bandpass filtering the sibilant region and reducing gain when active."""
    # Design Bandpass filter for sibilance detection
    nyq = sr / 2
    b, a = signal.butter(4, [f_low / nyq, f_high / nyq], btype="band")
    sibilants = signal.filtfilt(b, a, y)
    
    # Calculate sibilants energy
    sib_env = _rms_envelope(sibilants)
    overall_env = _rms_envelope(y)
    
    threshold = 10 ** (threshold_db / 20.0)
    out = np.copy(y)
    
    # Design a dynamic notch filter around 6.5kHz
    b_notch, a_notch = signal.iirnotch(6500.0 / nyq, 1.5)
    
    for i in range(len(y)):
        # If sibilance is high and represents a significant portion of overall signal
        if sib_env[i] > threshold and sib_env[i] > 0.3 * (overall_env[i] + 1e-6):
            # Apply dynamic attenuation (filter input samples through notch filter)
            # Simplification: crossfade with filtered version
            # (Just process blockwise or sample-by-sample for smooth dynamic EQ)
            pass
            
    # Practical fallback: dynamic shelving filter or block-wise gain reduction in sibilance band
    gain = np.ones_like(y)
    for i in range(len(y)):
        if sib_env[i] > threshold and sib_env[i] > 0.25 * (overall_env[i] + 1e-6):
            # Dynamic attenuation in the sibilance region
            ratio = threshold / (sib_env[i] + 1e-9)
            gain[i] = max(0.25, ratio) # Max 12dB reduction
            
    # Re-filter sibilant frequencies and attenuate them
    sibilant_gain = np.interp(np.arange(len(y)), np.arange(len(y)), gain)
    attenuated_sibilants = sibilants * sibilant_gain
    
    # Subtract original sibilants and add back attenuated sibilants
    return y - sibilants + attenuated_sibilants


def _diatonic_shift(note: float, scale_key: str, scale_mode: str, interval: int = 3) -> float:
    """Computes the diatonic shift in semitones for a given interval (e.g. 3rd, 5th)."""
    # Scale degree intervals: major scale = [2, 2, 1, 2, 2, 2, 1]
    # Minor scale = [2, 1, 2, 2, 1, 2, 2]
    # Simple lookup based on semitone offsets
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_idx = pitch_classes.index(scale_key.upper()) if scale_key.upper() in pitch_classes else 0
    
    # Degrees in semitones relative to key
    if scale_mode.lower() == "minor":
        degrees = [0, 2, 3, 5, 7, 8, 10]
    else:
        degrees = [0, 2, 4, 5, 7, 9, 11]
        
    midi_note = int(round(note))
    note_in_key = (midi_note - key_idx) % 12
    
    # Find closest degree
    closest_deg = min(range(len(degrees)), key=lambda idx: min(abs(note_in_key - degrees[idx]), abs(note_in_key - degrees[idx] - 12)))
    
    # Shift diatonic degree
    target_deg_idx = (closest_deg + (interval - 1)) % len(degrees)
    octave_shift = (closest_deg + (interval - 1)) // len(degrees)
    
    target_semitone = degrees[target_deg_idx] + octave_shift * 12
    current_semitone = degrees[closest_deg]
    
    return float(target_semitone - current_semitone)


def apply_pitch_correction(
    y: np.ndarray,
    sr: int = 48000,
    f0_series: List[float] = None,
    scale_key: str = "C",
    scale_mode: str = "major",
    correction_amount: float = 0.95
) -> np.ndarray:
    """Applies time-varying pitch correction to lock vocal to scale degrees."""
    if f0_series is None or len(f0_series) == 0:
        # Detect pitch using YIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            hop_length=512
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_list = f0.tolist()
    else:
        f0_list = f0_series
        
    f0 = np.array(f0_list)
    hop = 512
    frame_dur = hop / sr
    n_frames = len(f0)
    
    # Slices for overlap-add
    win_length = int(sr * 0.150) # 150ms windows
    hop_length = int(win_length * 0.25) # 75% overlap
    window = signal.windows.hann(win_length)
    
    out_buf = np.zeros(len(y), dtype=np.float32)
    win_sum = np.zeros(len(y), dtype=np.float32)
    
    # Pre-calculate semitone shift for each STFT frame
    shifts = np.zeros(n_frames)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_idx = pitch_classes.index(scale_key.upper()) if scale_key.upper() in pitch_classes else 0
    scale_intervals = [0, 2, 3, 5, 7, 8, 10] if scale_mode.lower() == "minor" else [0, 2, 4, 5, 7, 9, 11]
    
    for i in range(n_frames):
        hz = f0[i]
        if hz <= 40.0: # Unvoiced / silence
            shifts[i] = 0.0
            continue
        midi = librosa.hz_to_midi(hz)
        midi_note_in_key = (int(round(midi)) - key_idx) % 12
        
        # Find closest scale degree
        closest_scale_note = min(scale_intervals, key=lambda note: min(abs(midi_note_in_key - note), abs(midi_note_in_key - note - 12)))
        target_midi = midi - midi_note_in_key + closest_scale_note
        
        # Compute correction shift
        diff = target_midi - midi
        shifts[i] = diff * correction_amount
        
    # Overlap-add loop
    for offset in range(0, len(y) - win_length, hop_length):
        chunk = y[offset : offset + win_length]
        mid_time = (offset + win_length / 2) / sr
        frame_idx = min(n_frames - 1, int(mid_time / frame_dur))
        
        shift = shifts[frame_idx]
        if abs(shift) > 0.05:
            # Shift pitch of this window chunk
            try:
                shifted = librosa.effects.pitch_shift(chunk, sr=sr, n_steps=shift)
            except Exception:
                shifted = chunk
        else:
            shifted = chunk
            
        out_buf[offset : offset + win_length] += shifted * window
        win_sum[offset : offset + win_length] += window * window
        
    # Normalize envelope overlap
    win_sum[win_sum < 1e-4] = 1.0
    return out_buf / win_sum


def align_vocal_timing(y: np.ndarray, sr: int, downbeats: List[float]) -> np.ndarray:
    """Aligns vocal phrase onsets to the downbeat / beat grid by shifting chunks."""
    if not downbeats:
        return y
        
    # Simple vocal onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
    
    if len(onset_times) == 0:
        return y
        
    out = np.copy(y)
    # Group vocal activity into phrases based on silence threshold
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    threshold = 0.012
    voiced_frames = rms > threshold
    
    changes = np.diff(voiced_frames.astype(np.int32), prepend=0, append=0)
    starts = np.where(changes == 1)[0] * 512 / sr
    ends = np.where(changes == -1)[0] * 512 / sr
    
    for start, end in zip(starts, ends):
        if end - start < 0.2:
            continue
        # Find first onset inside this phrase
        phrase_onsets = [t for t in onset_times if start <= t <= end]
        if not phrase_onsets:
            continue
        first_onset = phrase_onsets[0]
        
        # Find closest beat
        closest_beat = min(downbeats, key=lambda b: abs(b - first_onset))
        offset = closest_beat - first_onset
        
        # If shift is reasonable (e.g. < 150ms), align vocal phrase
        if abs(offset) < 0.150:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            shift_samples = int(offset * sr)
            
            phrase = y[start_sample:end_sample]
            # Zero out original location
            out[start_sample:end_sample] = 0.0
            
            # Add back shifted phrase (clamped to bounds)
            target_start = max(0, start_sample + shift_samples)
            target_end = min(len(y), end_sample + shift_samples)
            
            actual_len = target_end - target_start
            if actual_len > 0:
                out[target_start:target_end] += phrase[:actual_len]
                
    return out


def generate_doubles_and_harmonies(
    y_clean: np.ndarray,
    sr: int,
    midi_contract: Dict[str, Any],
    vocal_chain_options: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Generates wide panned doubles and a 3-part vocal harmony (third/fifth intervals)."""
    scale_key = str(midi_contract.get("key", "C")).split()[0]
    scale_mode = "minor" if "minor" in str(midi_contract.get("key", "C")).lower() else "major"
    
    f0_series = midi_contract.get("f0_series", [])
    stems = {}
    
    # 1. Double tracking (slightly delayed + detuned)
    # Left Double: delayed by 18ms, detuned +8 cents
    delay_samples_l = int(0.018 * sr)
    l_double = np.zeros_like(y_clean)
    try:
        detuned_l = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=0.08) # +8 cents
    except Exception:
        detuned_l = y_clean
    l_double[delay_samples_l:] = detuned_l[:-delay_samples_l]
    
    # Right Double: delayed by 24ms, detuned -8 cents
    delay_samples_r = int(0.024 * sr)
    r_double = np.zeros_like(y_clean)
    try:
        detuned_r = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=-0.08) # -8 cents
    except Exception:
        detuned_r = y_clean
    r_double[delay_samples_r:] = detuned_r[:-delay_samples_r]
    
    stems["vocal_double_l"] = l_double * 0.72
    stems["vocal_double_r"] = r_double * 0.72
    
    # 2. Harmonization (diasontic third & fifth)
    if vocal_chain_options.get("harmonies") == "key_chord_guided":
        # Generate Third Harmony
        harmony_3 = np.zeros_like(y_clean)
        # Shift pitch of vocal by diatonic third
        try:
            # Diatonic shift is time-varying. For simplicity we shift by 3.5 semitones (average third)
            # Or run a custom frame-by-frame pitch-shifter.
            # Using pitch shift by +3 semitones for minor key, +4 semitones for major key
            steps = 3.0 if scale_mode == "minor" else 4.0
            harmony_3 = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=steps)
        except Exception:
            harmony_3 = y_clean
            
        # Generate Fifth Harmony
        harmony_5 = np.zeros_like(y_clean)
        try:
            harmony_5 = librosa.effects.pitch_shift(y_clean, sr=sr, n_steps=7.0) # perfect fifth
        except Exception:
            harmony_5 = y_clean
            
        stems["vocal_harmony_3"] = harmony_3 * 0.60
        stems["vocal_harmony_5"] = harmony_5 * 0.52
        
    return stems


def process_vocal(vocal_path: str, midi_contract: Dict[str, Any], vocal_chain_options: Dict[str, Any]) -> Dict[str, str]:
    """Runs the full Vocal Production DSP pipeline and returns paths to processed stems."""
    y, sr = sf.read(vocal_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    
    # 1. Gate & De-esser
    if vocal_chain_options.get("noise_reduction", True):
        y = apply_gate(y, threshold_db=-46.0, sr=sr)
    if vocal_chain_options.get("de_ess", True):
        y = apply_deesser(y, sr=sr)
        
    # 2. Timing Alignment
    if vocal_chain_options.get("timing_correction") == "beat_grid":
        downbeats = midi_contract.get("downbeats", [])
        y = align_vocal_timing(y, sr, downbeats)
        
    # 3. Pitch Correction
    scale = str(midi_contract.get("key", "C"))
    scale_key = scale.split()[0]
    scale_mode = "minor" if "minor" in scale.lower() else "major"
    
    f0_series = midi_contract.get("f0_series", [])
    
    y_clean = apply_pitch_correction(
        y, 
        sr=sr, 
        f0_series=f0_series, 
        scale_key=scale_key, 
        scale_mode=scale_mode,
        correction_amount=0.92
    )
    
    out_dir = Path(vocal_path).parent / "processed_vocals"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    main_vocal_path = out_dir / "vocal_lead.wav"
    sf.write(str(main_vocal_path), y_clean, sr, subtype="PCM_16")
    
    stems_paths = {"vocal_lead": str(main_vocal_path)}
    
    # 4. Generate Doubles and Harmonies
    stems_audio = generate_doubles_and_harmonies(y_clean, sr, midi_contract, vocal_chain_options)
    for name, audio in stems_audio.items():
        p = out_dir / f"{name}.wav"
        sf.write(str(p), audio, sr, subtype="PCM_16")
        stems_paths[name] = str(p)
        
    return stems_paths
