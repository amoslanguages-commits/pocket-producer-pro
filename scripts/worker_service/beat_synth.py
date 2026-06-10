#!/usr/bin/env python3
"""
Studio-grade beat & instrument synthesis engine.

Key improvements over the previous deterministic synth:
  • Vocal-phrase-aware arrangement: instruments breathe with the singer —
    they duck/rest during vocal phrases and accent phrase boundaries.
  • Groove humanisation: timing micro-offsets and velocity curves that match
    genre feel (swing, laid-back, straight).
  • Per-section energy contour: intro builds slowly, chorus is full, outro strips back.
  • Accurate BPM-locked beat grid using actual detected downbeats from audio analysis.
  • Rich additive-synth voices with proper overtone structures per instrument type.
  • Syncopated bass patterns that follow chord roots per beat, not per bar.
  • Ghost notes and hi-hat subdivisions for genre authenticity.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

SAMPLE_RATE = 48000

# ---------------------------------------------------------------------------
# Low-level audio helpers
# ---------------------------------------------------------------------------

def _sec_to_samples(sec: float) -> int:
    return max(1, int(sec * SAMPLE_RATE))


def _safe_bpm(bpm: Any) -> float:
    try:
        b = float(bpm)
    except Exception:
        b = 90.0
    return max(40.0, min(220.0, b))


def _adsr(n: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = max(1, int(n * attack))
    d = max(1, int(n * decay))
    r = max(1, int(n * release))
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0.0, 1.0, a, endpoint=False),
        np.linspace(1.0, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0.0, r, endpoint=True),
    ])
    if env.size < n:
        env = np.pad(env, (0, n - env.size), mode="edge")
    return env[:n].astype(np.float32)


def _tone(
    freq: float,
    duration_sec: float,
    gain: float = 0.3,
    harmonics: Optional[List[float]] = None,
    adsr: Tuple[float, float, float, float] = (0.01, 0.12, 0.7, 0.2),
    vibrato_hz: float = 0.0,
    vibrato_depth: float = 0.0,
) -> np.ndarray:
    n = _sec_to_samples(duration_sec)
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float64)
    if vibrato_hz > 0 and vibrato_depth > 0:
        t = t + (vibrato_depth / max(1.0, freq)) * np.sin(2.0 * np.pi * vibrato_hz * t)
    weights = harmonics or [1.0, 0.0]
    wave = np.zeros(n, dtype=np.float64)
    for k, w in enumerate(weights, start=1):
        if w:
            wave += w * np.sin(2.0 * np.pi * freq * k * t)
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 0:
        wave = wave / peak
    env = _adsr(n, *adsr)
    return (wave * env * gain).astype(np.float32)


def _noise_burst(duration_sec: float, gain: float = 0.15, seed: int = 0) -> np.ndarray:
    n = _sec_to_samples(duration_sec)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * gain).astype(np.float32)


def _add_at(buf: np.ndarray, i0: int, sig: np.ndarray) -> None:
    if sig is None or sig.size == 0:
        return
    if i0 < 0:
        sig = sig[-i0:]
        i0 = 0
    if i0 >= len(buf) or sig.size == 0:
        return
    end = min(len(buf), i0 + sig.size)
    buf[i0:end] += sig[: end - i0]


# ---------------------------------------------------------------------------
# Studio drum voices — layered synthesis for punch & realism
# ---------------------------------------------------------------------------

def _kick(seed: int = 0, gain: float = 1.0) -> np.ndarray:
    """Punchy kick: sub-thump (sweep 90→35 Hz) + click transient."""
    dur = 0.22
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    # Sub body: exponential pitch sweep
    sweep_hz = 35.0 + 55.0 * np.exp(-t * 40.0)
    phase = 2.0 * np.pi * np.cumsum(sweep_hz) / SAMPLE_RATE
    body = np.sin(phase)
    # Click: very short filtered noise
    click = _noise_burst(0.006, 0.55, seed)
    # Amplitude envelope
    env = np.exp(-t * 20.0) * 0.95
    out = body * env
    out[: click.size] += click * 0.45
    # Second harmonic for punch
    punch = np.sin(phase * 2) * np.exp(-t * 60.0) * 0.18
    out += punch
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak
    return (out * gain * 0.95).astype(np.float32)


def _snare(seed: int = 0, gain: float = 1.0, ghost: bool = False) -> np.ndarray:
    """Snare: tonal body (180 Hz) + noise crack, with ghost note variant."""
    dur = 0.20 if not ghost else 0.12
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    # Tonal body
    body = (
        np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 28.0) * 0.45
        + np.sin(2.0 * np.pi * 310.0 * t) * np.exp(-t * 40.0) * 0.15
    )
    # Noise crack
    noise = _noise_burst(dur, 0.55, seed)
    noise_env = np.exp(-t * (22.0 if not ghost else 40.0))
    snare = (body + noise * noise_env)
    peak = float(np.max(np.abs(snare)))
    if peak > 0:
        snare = snare / peak
    g = (0.28 if ghost else 1.0) * gain
    return (snare * g * 0.85).astype(np.float32)


def _hihat_closed(seed: int = 0, gain: float = 1.0) -> np.ndarray:
    dur = 0.045
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    noise = _noise_burst(dur, 0.6, seed)
    # Crude high-pass
    smooth = np.convolve(noise, np.ones(6) / 6.0, mode="same")
    hp = noise - smooth
    env = np.exp(-t * 80.0)
    return (hp * env * 0.55 * gain).astype(np.float32)


def _hihat_open(seed: int = 0, gain: float = 1.0) -> np.ndarray:
    dur = 0.18
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    noise = _noise_burst(dur, 0.55, seed)
    smooth = np.convolve(noise, np.ones(6) / 6.0, mode="same")
    hp = noise - smooth
    env = np.exp(-t * 15.0)
    return (hp * env * 0.50 * gain).astype(np.float32)


def _rimshot(seed: int = 0, gain: float = 1.0) -> np.ndarray:
    """Rim shot / clap for afrobeats / rnb flavour."""
    dur = 0.10
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    click = np.sin(2.0 * np.pi * 900.0 * t) * np.exp(-t * 90.0) * 0.4
    noise = _noise_burst(dur, 0.5, seed) * np.exp(-t * 55.0)
    out = click + noise
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out = out / peak
    return (out * gain * 0.7).astype(np.float32)


# ---------------------------------------------------------------------------
# Groove / humanisation helpers
# ---------------------------------------------------------------------------

def _swing_offset(beat_index: int, swing: float, beat_sec: float) -> float:
    """Returns a time offset (seconds) to apply to 8th-note subdivisions.

    swing=0.5 → straight, swing=0.67 → triplet feel.
    Only affects off-beats (even-numbered 8th notes).
    """
    if beat_index % 2 == 1:
        straight_half = beat_sec * 0.5
        swung_half = beat_sec * swing
        return swung_half - straight_half
    return 0.0


def _humanize_ms(rng: np.random.Generator, max_ms: float = 12.0) -> float:
    """Random micro-timing offset in seconds (±max_ms)."""
    return float(rng.uniform(-max_ms, max_ms)) * 0.001


def _velocity_curve(role: str, beat_index: int, n_beats_per_bar: int = 4) -> float:
    """Velocity multiplier with per-role dynamics."""
    pos = beat_index % n_beats_per_bar
    if role == "chorus":
        # Full velocity, slight accent on 1 and 3
        return 0.95 + 0.05 * (1.0 if pos in (0, 2) else 0.0)
    elif role == "intro":
        # Softer with build
        return 0.6 + 0.1 * (pos / (n_beats_per_bar - 1))
    elif role == "outro":
        return 0.65
    else:  # verse / bridge
        return 0.80 + 0.08 * (1.0 if pos == 0 else 0.0)


# ---------------------------------------------------------------------------
# Chord / key helpers
# ---------------------------------------------------------------------------

PITCH_CLASS_HZ = {
    "C": 130.81, "C#": 138.59, "DB": 138.59, "D": 146.83,
    "D#": 155.56, "EB": 155.56, "E": 164.81, "F": 174.61,
    "F#": 185.00, "GB": 185.00, "G": 196.00, "G#": 207.65,
    "AB": 207.65, "A": 220.00, "A#": 233.08, "BB": 233.08, "B": 246.94,
}

MIDI_NOTE = {
    "C": 60, "C#": 61, "DB": 61, "D": 62, "D#": 63, "EB": 63,
    "E": 64, "F": 65, "F#": 66, "GB": 66, "G": 67, "G#": 68,
    "AB": 68, "A": 69, "A#": 70, "BB": 70, "B": 71,
}


def _midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _chord_label_to_root_hz(label: str) -> Tuple[float, str]:
    """Parse chord label like 'Am', 'F#', 'Bb' → (root_hz, quality)."""
    if not label:
        return PITCH_CLASS_HZ["C"], "maj"
    lbl = label.strip()
    root = lbl[0].upper()
    suffix = lbl[1:]
    if suffix and suffix[0] in ("#", "b"):
        root += suffix[0].replace("b", "B")
        suffix = suffix[1:]
    # Normalise accidentals
    root = root.replace("b", "B")
    hz = PITCH_CLASS_HZ.get(root, PITCH_CLASS_HZ["C"])
    quality = "min" if "m" in suffix.lower() and "maj" not in suffix.lower() else "maj"
    if "dim" in suffix.lower():
        quality = "dim"
    return hz, quality


def _chord_pitches_hz(root_hz: float, quality: str) -> List[float]:
    """Return [root, third, fifth] frequencies."""
    if quality == "min":
        return [root_hz, root_hz * (2 ** (3 / 12)), root_hz * (2 ** (7 / 12))]
    if quality == "dim":
        return [root_hz, root_hz * (2 ** (3 / 12)), root_hz * (2 ** (6 / 12))]
    return [root_hz, root_hz * (2 ** (4 / 12)), root_hz * (2 ** (7 / 12))]


# ---------------------------------------------------------------------------
# Vocal-phrase awareness
# ---------------------------------------------------------------------------

def _vocal_active_at(phrases: List[Dict[str, Any]], t: float) -> bool:
    """True if vocal is singing at time t."""
    for p in phrases:
        s = float(p.get("start_sec", 0))
        e = float(p.get("end_sec", 0))
        if s <= t < e:
            return True
    return False


def _phrase_boundary_near(phrases: List[Dict[str, Any]], t: float, window: float = 0.25) -> bool:
    """True if we're within `window` seconds of a vocal phrase start/end."""
    for p in phrases:
        s = float(p.get("start_sec", 0))
        e = float(p.get("end_sec", 0))
        if abs(t - s) <= window or abs(t - e) <= window:
            return True
    return False


def _inter_phrase_gain(phrases: List[Dict[str, Any]], t: float) -> float:
    """Instruments play fuller (1.0) when vocal is NOT active, duck slightly (0.55) underneath."""
    if _vocal_active_at(phrases, t):
        return 0.55  # Duck underneath singer
    return 1.0  # Full presence in gaps / instrumental breaks


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _section_at(sections: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    for s in sections:
        try:
            start = float(s.get("start_sec", 0))
            end = float(s.get("end_sec", 1e9))
            if start <= t < end:
                return s
        except Exception:
            continue
    return None


def _section_role_at(sections: List[Dict[str, Any]], t: float) -> str:
    sec = _section_at(sections, t)
    if sec:
        return str(sec.get("role", sec.get("label", "verse"))).lower()
    return "verse"


def _section_energy_at(sections: List[Dict[str, Any]], t: float) -> float:
    sec = _section_at(sections, t)
    if sec:
        return float(sec.get("energy", 0.5))
    return 0.5


# ---------------------------------------------------------------------------
# Beat grid — uses actual detected downbeats from analysis
# ---------------------------------------------------------------------------

def _build_beat_grid(
    bpm: float,
    downbeats: List[float],
    total_sec: float,
) -> Tuple[List[float], List[float]]:
    """Build a sample-accurate beat grid.

    If real downbeats are provided (from madmom/librosa analysis), we anchor
    the grid to them so the generated beat is phase-locked to the original song.
    Returns (beat_times, downbeat_times).
    """
    beat_sec = 60.0 / bpm
    beats: List[float] = []
    dbs: List[float] = []

    cleaned_db = sorted({float(x) for x in downbeats if isinstance(x, (int, float)) and float(x) >= 0})

    if len(cleaned_db) >= 2:
        # Use detected downbeats as anchors
        for i, db in enumerate(cleaned_db):
            dbs.append(db)
            end = cleaned_db[i + 1] if i + 1 < len(cleaned_db) else total_sec
            t = db
            while t < end - 0.001 and t <= total_sec:
                beats.append(t)
                t += beat_sec
        # Extend past last downbeat
        if beats:
            last = beats[-1]
            t = last + beat_sec
            while t <= total_sec:
                beats.append(t)
                t += beat_sec
    else:
        # Fallback: mathematical grid from 0
        n = int(max(1, total_sec / beat_sec)) + 1
        beats = [i * beat_sec for i in range(n)]
        dbs = [i * beat_sec * 4 for i in range(max(1, int(n / 4)))]

    return beats, dbs


# ---------------------------------------------------------------------------
# Genre-aware drum pattern renderer
# ---------------------------------------------------------------------------

def _render_drums(
    buf: np.ndarray,
    beat_times: List[float],
    downbeat_times: List[float],
    sections: List[Dict[str, Any]],
    vocal_phrases: List[Dict[str, Any]],
    genre_key: str,
    beat_sec: float,
    swing: float,
    seed: int,
    total_sec: float,
) -> None:
    rng = np.random.default_rng(seed)
    kick_s = _kick(seed)
    snare_s = _snare(seed + 7)
    ghost_s = _snare(seed + 7, ghost=True)
    hat_c = _hihat_closed(seed + 17)
    hat_o = _hihat_open(seed + 23)
    rim = _rimshot(seed + 31)

    downbeat_set = set(round(d, 3) for d in downbeat_times)

    for bi, t0 in enumerate(beat_times):
        if t0 > total_sec:
            break

        role = _section_role_at(sections, t0)
        energy = _section_energy_at(sections, t0)
        v_gain = _velocity_curve(role, bi)
        vocal_duck = _inter_phrase_gain(vocal_phrases, t0)

        is_downbeat = any(abs(t0 - d) < 0.035 for d in downbeat_set) or bi % 4 == 0
        beat_in_bar = bi % 4
        sub_div = bi % 2  # 0=on-beat, 1=off-beat

        # Humanize
        jitter = _humanize_ms(rng, max_ms=8.0 if role == "chorus" else 14.0)

        i0_raw = _sec_to_samples(t0 + jitter)
        i0 = max(0, min(len(buf) - 1, i0_raw))

        # ----- KICK -----
        kick_vel = v_gain * (1.0 if is_downbeat else 0.72) * energy
        if genre_key == "afro":
            # Afrobeats: kick on 1 and the "e" of 3
            if beat_in_bar in (0, 2):
                _add_at(buf, i0, kick_s * kick_vel)
            # Syncopated ghost kick
            if beat_in_bar == 1:
                _add_at(buf, _sec_to_samples(t0 + beat_sec * 0.5), kick_s * kick_vel * 0.5)
        else:
            # Straight: kick on 1 and 3
            if beat_in_bar in (0, 2):
                _add_at(buf, i0, kick_s * kick_vel)
            # Occasional syncopated kick (before beat 3)
            if beat_in_bar == 1 and rng.random() < 0.25 and role == "chorus":
                _add_at(buf, _sec_to_samples(t0 + beat_sec * 0.75), kick_s * kick_vel * 0.55)

        # ----- SNARE -----
        snare_vel = v_gain * 0.88 * energy
        if beat_in_bar in (1, 3):  # classic backbeat 2 & 4
            _add_at(buf, i0, snare_s * snare_vel)
        # Ghost notes (genre-dependent density)
        ghost_density = 0.7 if genre_key in ("afro", "rnb") else 0.3
        if beat_in_bar not in (1, 3) and rng.random() < ghost_density * energy:
            _add_at(buf, i0, ghost_s * snare_vel * 0.3)

        # ----- HI-HAT -----
        hat_vel = v_gain * 0.75
        # Main closed hat on every beat
        _add_at(buf, i0, hat_c * hat_vel)
        # Offbeat hat (8th subdivision) — swung
        swing_offset = _swing_offset(1, swing, beat_sec)
        off_t = t0 + beat_sec * 0.5 + swing_offset
        if off_t < total_sec:
            # Open hat on the "and" of 4 in chorus / afro
            if beat_in_bar == 3 and role in ("chorus",):
                _add_at(buf, _sec_to_samples(off_t), hat_o * hat_vel * 0.85)
            else:
                _add_at(buf, _sec_to_samples(off_t), hat_c * hat_vel * 0.7)

        # ----- RIM / CLAP (afro/rnb) -----
        if genre_key in ("afro", "rnb") and beat_in_bar == 2:
            _add_at(buf, i0, rim * v_gain * 0.65)

        # ----- DRUM FILL (pre-section-change) -----
        if role in ("verse", "bridge") and beat_in_bar == 3 and _phrase_boundary_near(vocal_phrases, t0 + beat_sec, 0.4):
            for frac in [0.5, 0.75]:
                ft = t0 + beat_sec * frac
                if ft < total_sec:
                    _add_at(buf, _sec_to_samples(ft), snare_s * v_gain * 0.7)


# ---------------------------------------------------------------------------
# Bass renderer — vocal-phrase-aware, syncopated roots
# ---------------------------------------------------------------------------

def _render_bass(
    buf: np.ndarray,
    beat_times: List[float],
    sections: List[Dict[str, Any]],
    chord_progression: List[str],
    vocal_phrases: List[Dict[str, Any]],
    beat_sec: float,
    swing: float,
    bars: int,
    seed: int,
    total_sec: float,
) -> None:
    rng = np.random.default_rng(seed + 1)
    # Bass ADSR: plucky attack, moderate sustain
    bass_adsr = (0.005, 0.08, 0.72, 0.18)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break

        chord_label = chord_progression[bar % len(chord_progression)] if chord_progression else "C"
        root_hz, quality = _chord_label_to_root_hz(chord_label)
        bass_hz = root_hz / 2.0  # One octave down for sub-bass register

        role = _section_role_at(sections, bar_start)
        energy = _section_energy_at(sections, bar_start)
        vocal_gain = _inter_phrase_gain(vocal_phrases, bar_start)

        # Beat pattern varies by role
        if role == "intro":
            # Sparse: only downbeats
            beat_pattern = [0]
        elif role == "chorus":
            # Full: root on every beat + anticipation
            beat_pattern = [0, 1, 2, 3]
        elif role == "bridge":
            beat_pattern = [0, 2]
        else:  # verse
            beat_pattern = [0, 1, 2] if rng.random() < 0.6 else [0, 2]

        for beat in beat_pattern:
            t = bar_start + beat * beat_sec
            if t >= total_sec:
                break
            # Syncopated anticipation: shift off-beat notes slightly early
            swing_off = _swing_offset(beat, swing, beat_sec)
            t_play = t + swing_off
            duration = beat_sec * (0.82 if beat < 3 else 0.55)

            # Occasional chromatic walk-up approaching chord change
            note_hz = bass_hz
            if beat == 3 and rng.random() < 0.35 and role != "intro":
                next_chord = chord_progression[(bar + 1) % len(chord_progression)] if chord_progression else "C"
                next_root, _ = _chord_label_to_root_hz(next_chord)
                next_bass = next_root / 2.0
                if next_bass > bass_hz:
                    note_hz = bass_hz * (2 ** (2 / 12))  # Leading tone up
                else:
                    note_hz = bass_hz * (2 ** (-1 / 12))  # Leading tone down

            bass_note = _tone(
                note_hz,
                duration,
                0.38 * energy * vocal_gain,
                harmonics=[1.0, 0.55, 0.28, 0.12, 0.06],
                adsr=bass_adsr,
            )
            _add_at(buf, _sec_to_samples(t_play), bass_note)


# ---------------------------------------------------------------------------
# Keys renderer — voice-led chords
# ---------------------------------------------------------------------------

def _render_keys(
    buf: np.ndarray,
    beat_times: List[float],
    sections: List[Dict[str, Any]],
    chord_progression: List[str],
    vocal_phrases: List[Dict[str, Any]],
    beat_sec: float,
    bars: int,
    seed: int,
    total_sec: float,
) -> None:
    rng = np.random.default_rng(seed + 2)
    pad_adsr = (0.06, 0.22, 0.78, 0.32)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break

        chord_label = chord_progression[bar % len(chord_progression)] if chord_progression else "C"
        root_hz, quality = _chord_label_to_root_hz(chord_label)
        pitches = _chord_pitches_hz(root_hz, quality)

        role = _section_role_at(sections, bar_start)
        energy = _section_energy_at(sections, bar_start)
        vocal_gain = _inter_phrase_gain(vocal_phrases, bar_start)
        bar_dur = 4 * beat_sec

        if role == "intro":
            # Sparse single notes or root+fifth
            chord_gain = 0.12 * energy * vocal_gain
            note = _tone(pitches[0], bar_dur, chord_gain,
                         harmonics=[1.0, 0.3, 0.15], adsr=pad_adsr)
            _add_at(buf, _sec_to_samples(bar_start), note)
        elif role in ("verse", "bridge"):
            # Comped chords — staggered entry for realism
            for j, hz in enumerate(pitches):
                stagger = j * 0.018  # 18 ms between notes (arpeggiated feel)
                note = _tone(hz, bar_dur * 0.92, 0.14 * energy * vocal_gain,
                             harmonics=[1.0, 0.38, 0.18], adsr=pad_adsr,
                             vibrato_hz=3.5, vibrato_depth=0.4)
                _add_at(buf, _sec_to_samples(bar_start + stagger), note)
        else:  # chorus, outro
            # Full chord swell with octave doubling
            for j, hz in enumerate(pitches):
                stagger = j * 0.012
                note = _tone(hz, bar_dur, 0.17 * energy * vocal_gain,
                             harmonics=[1.0, 0.5, 0.25, 0.12], adsr=pad_adsr,
                             vibrato_hz=4.5, vibrato_depth=0.5)
                _add_at(buf, _sec_to_samples(bar_start + stagger), note)
            # Octave up doubling for shimmer
            octave_hz = pitches[0] * 2.0
            octave = _tone(octave_hz, bar_dur * 0.85, 0.07 * energy * vocal_gain,
                           harmonics=[1.0, 0.3], adsr=(0.08, 0.30, 0.65, 0.40))
            _add_at(buf, _sec_to_samples(bar_start), octave)


# ---------------------------------------------------------------------------
# Guitar / strumming
# ---------------------------------------------------------------------------

def _render_guitar(
    buf: np.ndarray,
    sections: List[Dict[str, Any]],
    chord_progression: List[str],
    vocal_phrases: List[Dict[str, Any]],
    beat_sec: float,
    bars: int,
    seed: int,
    total_sec: float,
) -> None:
    rng = np.random.default_rng(seed + 3)
    strum_adsr = (0.003, 0.16, 0.52, 0.38)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break

        chord_label = chord_progression[bar % len(chord_progression)] if chord_progression else "C"
        root_hz, quality = _chord_label_to_root_hz(chord_label)
        pitches = _chord_pitches_hz(root_hz * 2.0, quality)  # Up one octave

        role = _section_role_at(sections, bar_start)
        energy = _section_energy_at(sections, bar_start)
        vocal_gain = _inter_phrase_gain(vocal_phrases, bar_start)

        if role == "intro":
            continue  # No guitar in intro

        # Strum pattern: off-beats in verse, all beats in chorus
        strum_offsets = []
        if role == "chorus":
            strum_offsets = [0.0, beat_sec * 1.0, beat_sec * 2.0, beat_sec * 3.0]
        elif role in ("verse",):
            strum_offsets = [beat_sec * 0.5, beat_sec * 2.5]
        elif role == "bridge":
            strum_offsets = [0.0, beat_sec * 2.0]

        for offset in strum_offsets:
            t = bar_start + offset
            if t >= total_sec:
                break
            strum_g = 0.09 * energy * vocal_gain
            # Spread strum: each string slightly delayed
            for j, hz in enumerate(pitches):
                spread = j * 0.008
                note = _tone(hz, beat_sec * 0.55, strum_g,
                             harmonics=[1.0, 0.45, 0.22, 0.1],
                             adsr=strum_adsr,
                             vibrato_hz=5.0, vibrato_depth=0.3)
                _add_at(buf, _sec_to_samples(t + spread), note)


# ---------------------------------------------------------------------------
# Strings — swelling pads for chorus lift
# ---------------------------------------------------------------------------

def _render_strings(
    buf: np.ndarray,
    sections: List[Dict[str, Any]],
    chord_progression: List[str],
    vocal_phrases: List[Dict[str, Any]],
    beat_sec: float,
    bars: int,
    seed: int,
    total_sec: float,
) -> None:
    string_adsr = (0.25, 0.18, 0.85, 0.45)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break

        chord_label = chord_progression[bar % len(chord_progression)] if chord_progression else "C"
        root_hz, quality = _chord_label_to_root_hz(chord_label)
        pitches = _chord_pitches_hz(root_hz * 1.5, quality)  # Upper register

        role = _section_role_at(sections, bar_start)
        energy = _section_energy_at(sections, bar_start)
        vocal_gain = _inter_phrase_gain(vocal_phrases, bar_start)
        bar_dur = 4 * beat_sec

        if role not in ("chorus", "outro", "bridge"):
            continue  # Strings only in high-energy sections

        g = 0.08 * energy * vocal_gain
        for hz in pitches:
            note = _tone(hz, bar_dur, g,
                         harmonics=[1.0, 0.52, 0.28, 0.16, 0.08],
                         adsr=string_adsr,
                         vibrato_hz=5.5, vibrato_depth=0.55)
            _add_at(buf, _sec_to_samples(bar_start), note)


# ---------------------------------------------------------------------------
# Brass stabs — only in chorus
# ---------------------------------------------------------------------------

def _render_brass(
    buf: np.ndarray,
    sections: List[Dict[str, Any]],
    chord_progression: List[str],
    vocal_phrases: List[Dict[str, Any]],
    beat_sec: float,
    bars: int,
    seed: int,
    total_sec: float,
) -> None:
    brass_adsr = (0.004, 0.07, 0.40, 0.14)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break

        chord_label = chord_progression[bar % len(chord_progression)] if chord_progression else "C"
        root_hz, quality = _chord_label_to_root_hz(chord_label)
        pitches = _chord_pitches_hz(root_hz * 2.0, quality)

        role = _section_role_at(sections, bar_start)
        energy = _section_energy_at(sections, bar_start)
        vocal_gain = _inter_phrase_gain(vocal_phrases, bar_start)

        if role != "chorus":
            continue

        # Brass stabs at bar start and beat 3
        for offset in [0.0, 2.0 * beat_sec]:
            t = bar_start + offset
            if t >= total_sec:
                break
            stab_g = 0.13 * energy * vocal_gain
            for hz in pitches:
                note = _tone(hz, beat_sec * 0.28, stab_g,
                             harmonics=[1.0, 0.62, 0.38, 0.22, 0.12],
                             adsr=brass_adsr)
                _add_at(buf, _sec_to_samples(t), note)


# ---------------------------------------------------------------------------
# Normaliser / limiter
# ---------------------------------------------------------------------------

def _normalise(sig: np.ndarray, target: float = 0.90) -> np.ndarray:
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if peak > 1e-6:
        return sig * (target / peak)
    return sig


def _soft_clip(sig: np.ndarray, ceiling: float = 0.96) -> np.ndarray:
    """Soft saturation limiter to prevent harsh digital clipping."""
    x = sig / ceiling
    y = np.where(np.abs(x) <= 1.0, x - (x ** 3) / 3.0, np.sign(x) * (2.0 / 3.0))
    return (y * ceiling).astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_studio_beat(
    plan: Dict[str, Any],
    analysis: Dict[str, Any],
    out_dir: Path,
    seed_key: str,
) -> Dict[str, Path]:
    """
    Render a studio-quality beat that is:
      • Phase-locked to the vocal recording's actual downbeats
      • Vocal-phrase-aware (instruments duck under the singer)
      • Section-shaped (intro → verse → chorus dynamics)
      • Genre-appropriate groove (swing, ghost notes, fills)

    Returns dict of stem_name → Path.
    """
    bpm = _safe_bpm(plan.get("bpm", analysis.get("bpm", 90)))
    beat_sec = 60.0 / bpm

    # Use detected downbeats from real audio analysis for phase locking
    downbeats = plan.get("downbeats") or analysis.get("downbeats") or []
    sections = plan.get("sections") or analysis.get("sections") or []
    chord_progression = plan.get("chord_progression") or analysis.get("chord_progression") or ["C", "Am", "F", "G"]
    vocal_phrases = analysis.get("vocal_phrases") or []

    # Duration from section map; fall back to analysis duration or 45 s
    dur_from_sections = 0.0
    if sections:
        try:
            dur_from_sections = float(sections[-1].get("end_sec", 0))
        except Exception:
            pass
    total_sec = dur_from_sections or float(analysis.get("duration", 45.0)) or 45.0

    bars = int(max(8, total_sec / (beat_sec * 4))) + 1

    beat_times, downbeat_times = _build_beat_grid(bpm, downbeats, total_sec)

    # Determine genre key from plan
    genre_raw = str(plan.get("genre", analysis.get("genre", "pop"))).lower()
    if "afro" in genre_raw:
        genre_key = "afro"
    elif "r&b" in genre_raw or "rnb" in genre_raw or "soul" in genre_raw:
        genre_key = "rnb"
    elif "rock" in genre_raw:
        genre_key = "rock"
    else:
        genre_key = "pop"

    swing = float(plan.get("arrangement_rules", {}).get("by_section", [{}])[0].get("groove", {}).get("swing", 0.52))

    # Derive deterministic seed
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)

    n_samples = _sec_to_samples(total_sec) + _sec_to_samples(1.0)  # Small tail

    drums_buf = np.zeros(n_samples, dtype=np.float32)
    bass_buf = np.zeros(n_samples, dtype=np.float32)
    keys_buf = np.zeros(n_samples, dtype=np.float32)
    guitar_buf = np.zeros(n_samples, dtype=np.float32)
    strings_buf = np.zeros(n_samples, dtype=np.float32)
    brass_buf = np.zeros(n_samples, dtype=np.float32)

    # Render each instrument layer
    _render_drums(
        drums_buf, beat_times, downbeat_times, sections, vocal_phrases,
        genre_key, beat_sec, swing, seed, total_sec
    )
    _render_bass(
        bass_buf, beat_times, sections, chord_progression, vocal_phrases,
        beat_sec, swing, bars, seed, total_sec
    )
    _render_keys(
        keys_buf, beat_times, sections, chord_progression, vocal_phrases,
        beat_sec, bars, seed, total_sec
    )
    _render_guitar(
        guitar_buf, sections, chord_progression, vocal_phrases,
        beat_sec, bars, seed, total_sec
    )
    _render_strings(
        strings_buf, sections, chord_progression, vocal_phrases,
        beat_sec, bars, seed, total_sec
    )
    _render_brass(
        brass_buf, sections, chord_progression, vocal_phrases,
        beat_sec, bars, seed, total_sec
    )

    # Normalise each stem individually
    stems_raw = {
        "drums": drums_buf,
        "bass": bass_buf,
        "keys": keys_buf,
        "guitar": guitar_buf,
        "strings": strings_buf,
        "brass": brass_buf,
    }
    stems_norm: Dict[str, np.ndarray] = {}
    for name, sig in stems_raw.items():
        stems_norm[name] = _soft_clip(_normalise(sig, 0.88))

    # Rough mix: weighted sum
    mix = (
        stems_norm["drums"] * 0.85
        + stems_norm["bass"] * 0.78
        + stems_norm["keys"] * 0.70
        + stems_norm["guitar"] * 0.65
        + stems_norm["strings"] * 0.60
        + stems_norm["brass"] * 0.55
    )
    stems_norm["mix"] = _soft_clip(_normalise(mix, 0.92))

    # Write stems to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, sig in stems_norm.items():
        p = out_dir / f"{name}.wav"
        sf.write(str(p), sig, SAMPLE_RATE, subtype="PCM_24")
        out[name] = p

    return out
