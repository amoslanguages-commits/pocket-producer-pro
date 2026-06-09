#!/usr/bin/env python3
"""
Phase 2 render worker scaffold (FastAPI).

Flow:
1) Receives render payload from Supabase start-production-job.
2) Builds a simple deterministic arrangement for stems.
3) Renders stems as WAV files.
4) Uploads stems to public object storage URL (optional, if configured).
5) Calls Supabase worker-callback with completion payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pretty_midi
import requests
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

SAMPLE_RATE = 48000
MASTER_GAIN = 0.8


class StemRef(BaseModel):
    name: str
    url: str


class VersionRef(BaseModel):
    version_name: str = "Version A"
    version_type: str = "Original"
    genre: str = "Render Worker"
    preview_audio_path: Optional[str] = None
    full_audio_path: Optional[str] = None
    artist_vocal_path: Optional[str] = None


class RenderRequest(BaseModel):
    job_id: str
    project_id: str
    mode: str = "render_worker"
    callback_url: str
    callback_secret: Optional[str] = None
    analysis: Dict[str, Any] = Field(default_factory=dict)
    arrangement_plan: Dict[str, Any] = Field(default_factory=dict)
    midi_render_contract: Dict[str, Any] = Field(default_factory=dict)


class ProductionContracts(BaseModel):
    arrangement: Dict[str, Any] = Field(default_factory=dict)
    vocal_chain: Dict[str, Any] = Field(default_factory=dict)
    mix_chain: Dict[str, Any] = Field(default_factory=dict)
    mastering_chain: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class WorkerConfig:
    upload_base_url: str
    upload_token: str
    worker_auth_token: str
    sf2_path: str
    sample_render_base_url: str
    sample_render_token: str
    supabase_url: str
    supabase_service_role_key: str
    storage_bucket: str
    signed_url_ttl: int


def _worker_config() -> WorkerConfig:
    try:
        ttl = int(os.getenv("WORKER_SIGNED_URL_TTL", "31536000").strip() or "31536000")
    except ValueError:
        ttl = 31536000
    return WorkerConfig(
        upload_base_url=os.getenv("WORKER_UPLOAD_BASE_URL", "").strip(),
        upload_token=os.getenv("WORKER_UPLOAD_TOKEN", "").strip(),
        worker_auth_token=os.getenv("WORKER_AUTH_TOKEN", "").strip(),
        sf2_path=os.getenv("WORKER_SF2_PATH", "/app/soundfonts/default.sf2").strip(),
        sample_render_base_url=os.getenv("SAMPLE_RENDER_BASE_URL", "").strip(),
        sample_render_token=os.getenv("SAMPLE_RENDER_TOKEN", "").strip(),
        supabase_url=os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip(),
        storage_bucket=os.getenv("WORKER_STORAGE_BUCKET", "song-masters").strip() or "song-masters",
        signed_url_ttl=ttl,
    )


def _sec_to_samples(sec: float) -> int:
    return max(1, int(sec * SAMPLE_RATE))


def _adsr(n: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """Amplitude envelope (fractions of the note length) for natural dynamics."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    a = max(1, int(n * attack))
    d = max(1, int(n * decay))
    r = max(1, int(n * release))
    s = max(0, n - a - d - r)
    env = np.concatenate(
        [
            np.linspace(0.0, 1.0, a, endpoint=False),
            np.linspace(1.0, sustain, d, endpoint=False),
            np.full(s, sustain),
            np.linspace(sustain, 0.0, r, endpoint=True),
        ]
    )
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
    """Additive-synth tone with harmonics, ADSR and optional vibrato.

    `harmonics` are amplitude weights for partials 1..k (fundamental first).
    A richer partial stack + envelope makes the deterministic fallback sound
    like an instrument rather than a raw sine.
    """
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


def _noise(duration_sec: float, gain: float = 0.1, seed: int = 0) -> np.ndarray:
    n = _sec_to_samples(duration_sec)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * gain).astype(np.float32)


def _kick(seed: int = 0) -> np.ndarray:
    """Punchy kick: pitch-swept sine (≈90→45 Hz) with fast amplitude decay."""
    dur = 0.16
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    sweep = 45.0 + 45.0 * np.exp(-t * 38.0)
    phase = 2.0 * np.pi * np.cumsum(sweep) / SAMPLE_RATE
    body = np.sin(phase)
    click = _noise(0.004, 0.5, seed=seed) 
    env = np.exp(-t * 26.0)
    out = body * env
    out[: click.size] += click * 0.4
    return (out * 0.9).astype(np.float32)


def _snare(seed: int = 0) -> np.ndarray:
    """Snare: 180 Hz body + bright noise, both with snappy decays."""
    dur = 0.18
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-t * 24.0) * 0.5
    noise = _noise(dur, 0.6, seed=seed)
    noise_env = np.exp(-t * 18.0)
    return ((body + noise * noise_env) * 0.7).astype(np.float32)


def _hat(seed: int = 0, closed: bool = True) -> np.ndarray:
    """Hi-hat: high-passed white noise with a very short decay."""
    dur = 0.05 if closed else 0.12
    n = _sec_to_samples(dur)
    t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    noise = _noise(dur, 0.5, seed=seed)
    # crude high-pass: subtract a smoothed version to keep the bright content.
    smooth = np.convolve(noise, np.ones(8) / 8.0, mode="same")
    hp = noise - smooth
    env = np.exp(-t * (60.0 if closed else 22.0))
    return (hp * env * 0.5).astype(np.float32)


def _add_at(buf: np.ndarray, i0: int, sig: np.ndarray) -> None:
    """Add `sig` into `buf` starting at sample `i0`, clamped to bounds.

    Bulletproof against length/offset mismatches (the historical source of the
    'operands could not be broadcast together' render failure).
    """
    if sig is None or sig.size == 0:
        return
    if i0 < 0:
        sig = sig[-i0:]
        i0 = 0
    if i0 >= len(buf) or sig.size == 0:
        return
    end = min(len(buf), i0 + sig.size)
    buf[i0:end] += sig[: end - i0]


def _mix(signals: List[np.ndarray]) -> np.ndarray:
    if not signals:
        return np.zeros(_sec_to_samples(1.0), dtype=np.float32)
    n = max(len(s) for s in signals)
    out = np.zeros(n, dtype=np.float32)
    for s in signals:
        out[: len(s)] += s
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def _section_duration_sec(plan: Dict[str, Any], fallback_total: float = 45.0) -> float:
    sections = plan.get("sections", [])
    if isinstance(sections, list) and sections:
        try:
            start = float(sections[0].get("start_sec", 0.0))
            end = float(sections[-1].get("end_sec", fallback_total))
            return max(8.0, end - start)
        except Exception:
            return fallback_total
    return fallback_total


def _plan_rules(plan: Dict[str, Any]) -> Dict[str, Any]:
    rules = plan.get("arrangement_rules", {})
    return rules if isinstance(rules, dict) else {}


def _by_section(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = _plan_rules(plan)
    rows = rules.get("by_section", plan.get("sections", []))
    return rows if isinstance(rows, list) else []


def _transitions(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = _plan_rules(plan).get("transitions", [])
    return rows if isinstance(rows, list) else []


def _section_at(plan: Dict[str, Any], t: float) -> Optional[Dict[str, Any]]:
    for sec in _by_section(plan):
        try:
            start = float(sec.get("start_sec", 0.0))
            end = float(sec.get("end_sec", 1e9))
            if start <= t < end:
                return sec
        except Exception:
            continue
    return None


def _instrument_enabled(plan: Dict[str, Any], t: float, name: str) -> bool:
    sec = _section_at(plan, t)
    if not sec:
        return True
    inst = sec.get("instrumentation", {}).get(name, {})
    if isinstance(inst, dict):
        return bool(inst.get("enabled", True))
    return True


def _section_gain(plan: Dict[str, Any], t: float) -> float:
    sec = _section_at(plan, t)
    if not sec:
        return 1.0
    role = str(sec.get("role", "")).lower()
    lift_db = 0.0
    if role == "chorus":
        for tr in _transitions(plan):
            try:
                lift_db = max(lift_db, float(tr.get("chorus_lift_db", 0.0)))
            except Exception:
                continue
    if lift_db <= 0.0:
        return 1.0
    return float(10 ** (lift_db / 20.0))


def _is_breakdown_at(plan: Dict[str, Any], t: float) -> bool:
    sec = _section_at(plan, t)
    if not sec:
        return False
    label = str(sec.get("label", ""))
    for tr in _transitions(plan):
        if str(tr.get("from", "")) == label and tr.get("breakdown"):
            return True
    return False


def _fill_beats_before(plan: Dict[str, Any], t: float) -> float:
    sec = _section_at(plan, t)
    if not sec:
        return 0.0
    label = str(sec.get("label", ""))
    try:
        end = float(sec.get("end_sec", 0.0))
    except Exception:
        return 0.0
    if abs(t - end) > 0.35:
        return 0.0
    for tr in _transitions(plan):
        if str(tr.get("from", "")) == label and tr.get("fill_before_change"):
            try:
                return float(tr.get("drum_fill_beats", 0.5))
            except Exception:
                return 0.5
    return 0.0


def _beat_grid(plan: Dict[str, Any], total_sec: float) -> List[float]:
    bpm = float(plan.get("bpm", 90))
    beat_sec = 60.0 / max(40.0, min(220.0, bpm))
    downbeats = plan.get("downbeats", [])
    if not isinstance(downbeats, list):
        downbeats = []
    cleaned = sorted({float(x) for x in downbeats if isinstance(x, (int, float)) and float(x) >= 0.0})
    beats: List[float] = []
    if len(cleaned) >= 2:
        for i, db in enumerate(cleaned):
            end = cleaned[i + 1] if i + 1 < len(cleaned) else total_sec
            t = db
            while t < end - 0.001 and t <= total_sec:
                beats.append(t)
                t += beat_sec
        return beats
    n = int(max(1, total_sec / beat_sec))
    return [i * beat_sec for i in range(n)]


def _chord_at_bar(plan: Dict[str, Any], bar: int) -> str:
    chords = plan.get("chord_progression", [])
    if not isinstance(chords, list) or not chords:
        return "C"
    by_section = _by_section(plan)
    if by_section:
        sec = by_section[bar % len(by_section)]
        anchor = sec.get("harmony", {}).get("chord_anchor")
        if anchor:
            return str(anchor)
    return str(chords[bar % len(chords)])


def _midi_note_from_name(name: str) -> int:
    root = name.strip().upper()
    table = {
        "C": 60,
        "C#": 61,
        "DB": 61,
        "D": 62,
        "D#": 63,
        "EB": 63,
        "E": 64,
        "F": 65,
        "F#": 66,
        "GB": 66,
        "G": 67,
        "G#": 68,
        "AB": 68,
        "A": 69,
        "A#": 70,
        "BB": 70,
        "B": 71,
    }
    return table.get(root, 60)


def _chord_to_pitches(chord: str) -> List[int]:
    if not chord:
        return [60, 64, 67]
    label = chord.strip()
    root = label[0]
    suffix = label[1:]
    if suffix.startswith("#") or suffix.startswith("b"):
        root += suffix[0]
        suffix = suffix[1:]
    base = _midi_note_from_name(root)
    low = suffix.lower()
    if "dim" in low:
        return [base, base + 3, base + 6]
    if "m" in low and "maj" not in low:
        return [base, base + 3, base + 7]
    return [base, base + 4, base + 7]


def _build_midi_contract_from_plan(plan: Dict[str, Any], req: RenderRequest) -> Dict[str, Any]:
    contract = req.midi_render_contract or {}
    if contract:
        return contract
    return {
        "version": "midi-render-v1",
        "sample_rate": SAMPLE_RATE,
        "bpm": float(plan.get("bpm", req.analysis.get("bpm", 90))),
        "key": str(plan.get("key", req.analysis.get("key", "C"))),
        "tempo_map": req.analysis.get("tempo_map", []),
        "downbeats": plan.get("downbeats", req.analysis.get("downbeats", [])),
        "sections": plan.get("sections", req.analysis.get("sections", [])),
        "chord_progression": plan.get("chord_progression", req.analysis.get("chord_progression", [])),
        "render_targets": ["drums", "bass", "keys", "guitar", "strings", "brass", "mix"],
        "instruments": {
            "drums": {"sound": "studio_kit", "midi_channel": 10},
            "bass": {"sound": "finger_bass", "midi_channel": 2},
            "keys": {"sound": "grand_piano", "midi_channel": 3},
            "guitar": {"sound": "clean_guitar", "midi_channel": 4},
            "strings": {"sound": "ensemble_strings", "midi_channel": 5},
            "brass": {"sound": "pop_brass", "midi_channel": 6},
        },
    }


def _build_midi_from_contract(
    contract: Dict[str, Any],
    midi_path: Path,
    total_sec: float,
) -> Dict[str, Any]:
    bpm = float(contract.get("bpm", 90))
    beat_sec = 60.0 / max(40.0, min(220.0, bpm))
    bar_sec = beat_sec * 4.0
    bars = int(max(8, total_sec / bar_sec))

    chord_progression = contract.get("chord_progression", [])
    if not isinstance(chord_progression, list) or not chord_progression:
        chord_progression = ["C", "Am", "F", "G"]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    name_to_program = {
        "drums": (0, True),
        "bass": (33, False),
        "keys": (0, False),
        "guitar": (27, False),
        "strings": (48, False),
        "brass": (61, False),
    }
    tracks: Dict[str, pretty_midi.Instrument] = {}
    for name, (program, is_drum) in name_to_program.items():
        inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
        tracks[name] = inst
        pm.instruments.append(inst)

    for bar in range(bars):
        bar_start = bar * bar_sec
        chord = str(chord_progression[bar % len(chord_progression)])
        pitches = _chord_to_pitches(chord)

        # Drums: kick/snare/hats.
        for beat in range(4):
            t = bar_start + beat * beat_sec
            kick = pretty_midi.Note(velocity=108, pitch=36, start=t, end=t + 0.1)
            tracks["drums"].notes.append(kick)
            if beat in (1, 3):
                snare = pretty_midi.Note(velocity=98, pitch=38, start=t, end=t + 0.09)
                tracks["drums"].notes.append(snare)
            hat_t = t + beat_sec * 0.5
            hat = pretty_midi.Note(velocity=70, pitch=42, start=hat_t, end=hat_t + 0.05)
            tracks["drums"].notes.append(hat)

        # Bass roots per beat.
        bass_pitch = max(36, pitches[0] - 24)
        for beat in range(4):
            t = bar_start + beat * beat_sec
            tracks["bass"].notes.append(
                pretty_midi.Note(velocity=86, pitch=bass_pitch, start=t, end=t + beat_sec * 0.9)
            )

        # Keys full bar chord.
        for p in pitches:
            tracks["keys"].notes.append(
                pretty_midi.Note(velocity=74, pitch=p, start=bar_start, end=bar_start + bar_sec)
            )
        # Guitar stabs on offbeats.
        for beat in (0, 2):
            t = bar_start + beat * beat_sec + beat_sec * 0.5
            for p in pitches:
                tracks["guitar"].notes.append(
                    pretty_midi.Note(velocity=64, pitch=p + 12, start=t, end=t + beat_sec * 0.45)
                )
        # Strings sustained.
        for p in pitches:
            tracks["strings"].notes.append(
                pretty_midi.Note(velocity=56, pitch=p + 12, start=bar_start, end=bar_start + bar_sec)
            )
        # Brass only every 4 bars.
        if bar % 4 == 3:
            t = bar_start + bar_sec - beat_sec
            for p in pitches:
                tracks["brass"].notes.append(
                    pretty_midi.Note(velocity=68, pitch=p + 5, start=t, end=t + beat_sec * 0.8)
                )

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(midi_path))
    return {"bars": bars, "bpm": bpm, "midi_path": str(midi_path)}


def _render_midi_with_fluidsynth(
    midi_path: Path,
    wav_path: Path,
    sf2_path: str,
) -> bool:
    try:
        import shutil
        import subprocess

        if not shutil.which("fluidsynth"):
            return False
        if not sf2_path or not Path(sf2_path).exists():
            return False
        cmd = [
            "fluidsynth",
            "-ni",
            sf2_path,
            str(midi_path),
            "-F",
            str(wav_path),
            "-r",
            str(SAMPLE_RATE),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode == 0 and wav_path.exists()
    except Exception:
        return False


def _render_midi_with_sample_service(
    cfg: WorkerConfig,
    midi_path: Path,
    output_stem_dir: Path,
    contract: Dict[str, Any],
) -> Optional[Dict[str, Path]]:
    if not cfg.sample_render_base_url:
        return None
    try:
        headers = {"Content-Type": "application/json"}
        if cfg.sample_render_token:
            headers["Authorization"] = f"Bearer {cfg.sample_render_token}"
        with midi_path.open("rb") as f:
            midi_b64 = f.read().hex()
        payload = {
            "midi_hex": midi_b64,
            "contract": contract,
            "return_mode": "signed_urls",
        }
        res = requests.post(
            f"{cfg.sample_render_base_url.rstrip('/')}/render-midi",
            json=payload,
            headers=headers,
            timeout=120,
        )
        if not res.ok:
            return None
        data = res.json() if res.content else {}
        stems = data.get("stems", {})
        if not isinstance(stems, dict) or not stems:
            return None
        output_stem_dir.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Path] = {}
        for name, url in stems.items():
            if not isinstance(url, str) or not url:
                continue
            rr = requests.get(url, timeout=60)
            if not rr.ok:
                continue
            target = output_stem_dir / f"{name}.wav"
            target.write_bytes(rr.content)
            out[str(name)] = target
        return out if out else None
    except Exception:
        return None


def _render_stems(plan: Dict[str, Any], out_dir: Path, seed_key: str) -> Dict[str, Path]:
    bpm = float(plan.get("bpm", 90))
    length_sec = _section_duration_sec(plan)
    beat_sec = 60.0 / max(40.0, min(220.0, bpm))
    bars = int(max(8, length_sec / (beat_sec * 4)))
    total_sec = bars * 4 * beat_sec

    drums = np.zeros(_sec_to_samples(total_sec), dtype=np.float32)
    bass = np.zeros_like(drums)
    keys = np.zeros_like(drums)
    guitar = np.zeros_like(drums)
    strings = np.zeros_like(drums)
    brass = np.zeros_like(drums)

    # Deterministic random seed per job.
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)

    # Drums: dedicated kick/snare/hat voices with realistic envelopes.
    kick = _kick(seed)
    snare = _snare(seed + 7)
    hat_closed = _hat(seed + 17, closed=True)
    hat_open = _hat(seed + 23, closed=False)

    beats = _beat_grid(plan, total_sec)
    downbeats = sorted(
        {float(x) for x in (plan.get("downbeats") or []) if isinstance(x, (int, float))}
    )

    for bi, t0 in enumerate(beats):
        if t0 >= total_sec:
            break
        i0 = _sec_to_samples(t0)
        is_downbeat = any(abs(t0 - db) < 0.02 for db in downbeats) or bi % 4 == 0
        breakdown = _is_breakdown_at(plan, t0)
        drum_gain = 0.35 if breakdown else 1.0

        if _instrument_enabled(plan, t0, "drums"):
            if is_downbeat or bi % 2 == 0:
                _add_at(drums, i0, kick * drum_gain)
            if bi % 2 == 1:
                _add_at(drums, i0, snare * drum_gain)
            _add_at(drums, i0, hat_closed * 0.8 * drum_gain)
            ih = _sec_to_samples(min(t0 + beat_sec * 0.5, total_sec - 0.01))
            _add_at(drums, ih, (hat_open if bi % 4 == 3 else hat_closed) * drum_gain)

        fill_beats = _fill_beats_before(plan, t0)
        if fill_beats > 0 and _instrument_enabled(plan, t0, "drums"):
            for fb in range(int(max(1, round(fill_beats * 2)))):
                ft = t0 - (fb + 1) * beat_sec * 0.5
                if ft < 0:
                    break
                _add_at(drums, _sec_to_samples(ft), snare * 0.85)

    # Harmony / bass scaffold using chord roots from progression when available.
    key = str(plan.get("key", "C")).split()[0].upper()
    root_table = {
        "C": 130.81,
        "C#": 138.59,
        "DB": 138.59,
        "D": 146.83,
        "D#": 155.56,
        "EB": 155.56,
        "E": 164.81,
        "F": 174.61,
        "F#": 185.00,
        "GB": 185.00,
        "G": 196.00,
        "G#": 207.65,
        "AB": 207.65,
        "A": 220.00,
        "A#": 233.08,
        "BB": 233.08,
        "B": 246.94,
    }
    root = root_table.get(key, 130.81)

    # Bass: saw-like fingered bass (harmonic stack) with a plucky envelope.
    pad_adsr = (0.05, 0.20, 0.72, 0.30)
    brass_adsr = (0.003, 0.06, 0.45, 0.12)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        if bar_start >= total_sec:
            break
        bar_gain = _section_gain(plan, bar_start)
        breakdown = _is_breakdown_at(plan, bar_start)
        chord_name = _chord_at_bar(plan, bar)
        pitches = _chord_to_pitches(chord_name)
        bar_root_hz = 440.0 * (2 ** ((pitches[0] - 69) / 12.0))

        b_note = _tone(
            bar_root_hz / 2.0,
            beat_sec * 0.85,
            0.34 * bar_gain,
            harmonics=[1.0, 0.5, 0.28, 0.14, 0.07],
            adsr=(0.004, 0.10, 0.78, 0.16),
        )
        if _instrument_enabled(plan, bar_start, "bass") and not breakdown:
            for beat in range(4):
                t = bar_start + beat * beat_sec
                if t < total_sec:
                    _add_at(bass, _sec_to_samples(t), b_note)

        chord = _mix(
            [
                _tone(bar_root_hz, 4 * beat_sec, 0.16, harmonics=[1.0, 0.4, 0.2], adsr=pad_adsr),
                _tone(bar_root_hz * 1.5, 4 * beat_sec, 0.12, harmonics=[1.0, 0.35], adsr=pad_adsr),
                _tone(bar_root_hz * 2.0, 4 * beat_sec, 0.10, harmonics=[1.0, 0.3], adsr=pad_adsr),
            ]
        )
        if _instrument_enabled(plan, bar_start, "keys"):
            _add_at(keys, _sec_to_samples(bar_start), chord * bar_gain)

        g_pad = _tone(
            bar_root_hz * 2.5,
            beat_sec * 2,
            0.06 * bar_gain,
            harmonics=[1.0, 0.3, 0.15],
            adsr=(0.02, 0.18, 0.6, 0.35),
            vibrato_hz=5.0,
            vibrato_depth=0.6,
        )
        if _instrument_enabled(plan, bar_start, "guitar"):
            _add_at(guitar, _sec_to_samples(bar_start), g_pad)

        s_pad = _tone(
            bar_root_hz * 1.25,
            beat_sec * 4,
            0.07 * bar_gain,
            harmonics=[1.0, 0.5, 0.3, 0.18],
            adsr=(0.22, 0.20, 0.82, 0.40),
        )
        if _instrument_enabled(plan, bar_start, "strings"):
            _add_at(strings, _sec_to_samples(bar_start), s_pad)

        brass_stab = _mix(
            [
                _tone(bar_root_hz * 2.0, beat_sec * 0.22, 0.14, harmonics=[1.0, 0.55, 0.35, 0.2], adsr=brass_adsr),
                _tone(bar_root_hz * 2.5, beat_sec * 0.22, 0.11, harmonics=[1.0, 0.4, 0.25], adsr=brass_adsr),
                _tone(bar_root_hz * 3.0, beat_sec * 0.22, 0.08, harmonics=[1.0, 0.35], adsr=brass_adsr),
            ]
        )
        if _instrument_enabled(plan, bar_start, "brass"):
            _add_at(brass, _sec_to_samples(bar_start), brass_stab * bar_gain)

    # Glue and limit.
    stems = {
        "drums": drums * 0.9,
        "bass": bass * 0.9,
        "keys": keys * 0.9,
        "guitar": guitar * 0.9,
        "strings": strings * 0.9,
        "brass": brass * 0.82,
    }
    for k in list(stems.keys()):
        peak = float(np.max(np.abs(stems[k]))) if stems[k].size else 0.0
        if peak > 0.99:
            stems[k] = stems[k] * (0.99 / peak)

    mix = _mix(list(stems.values())) * MASTER_GAIN
    stems["mix"] = mix

    out: Dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        p = out_dir / f"{name}.wav"
        sf.write(str(p), audio, SAMPLE_RATE, subtype="PCM_16")
        out[name] = p
    return out


def _upload_to_supabase_storage(path: Path, remote_path: str, cfg: WorkerConfig) -> str:
    """Upload a file to Supabase Storage and return a durable signed URL.

    Uses the Storage REST API with the service-role key:
      1) POST .../object/{bucket}/{path}      (x-upsert) to store the bytes
      2) POST .../object/sign/{bucket}/{path} to mint a long-lived signed URL
    """
    bucket = cfg.storage_bucket
    object_path = remote_path.lstrip("/")
    base = f"{cfg.supabase_url}/storage/v1"
    auth = {"Authorization": f"Bearer {cfg.supabase_service_role_key}"}

    upload_url = f"{base}/object/{bucket}/{object_path}"
    with path.open("rb") as f:
        up = requests.post(
            upload_url,
            data=f,
            headers={**auth, "Content-Type": "audio/wav", "x-upsert": "true"},
            timeout=120,
        )
    up.raise_for_status()

    sign_url = f"{base}/object/sign/{bucket}/{object_path}"
    signed = requests.post(
        sign_url,
        json={"expiresIn": cfg.signed_url_ttl},
        headers={**auth, "Content-Type": "application/json"},
        timeout=30,
    )
    signed.raise_for_status()
    signed_path = (signed.json() or {}).get("signedURL") or (signed.json() or {}).get("signedUrl")
    if not signed_path:
        raise RuntimeError("Supabase Storage did not return a signed URL")
    return f"{base}{signed_path}" if signed_path.startswith("/") else f"{base}/{signed_path}"


def _upload_file(path: Path, remote_path: str, cfg: WorkerConfig) -> str:
    # Durable, mobile-consumable URLs are required (no file:// fallback).
    # Preferred path: upload straight to Supabase Storage and return a signed URL.
    if cfg.supabase_url and cfg.supabase_service_role_key:
        return _upload_to_supabase_storage(path, remote_path, cfg)

    # Fallback: generic PUT-able object store (WORKER_UPLOAD_BASE_URL).
    if cfg.upload_base_url:
        url = f"{cfg.upload_base_url.rstrip('/')}/{remote_path.lstrip('/')}"
        headers = {}
        if cfg.upload_token:
            headers["Authorization"] = f"Bearer {cfg.upload_token}"
        with path.open("rb") as f:
            r = requests.put(url, data=f, headers=headers, timeout=60)
        r.raise_for_status()
        return url

    raise RuntimeError(
        "No durable upload target configured. Set SUPABASE_URL + "
        "SUPABASE_SERVICE_ROLE_KEY (preferred) or WORKER_UPLOAD_BASE_URL."
    )


def _post_progress(req: RenderRequest, progress: int, step: str) -> None:
    headers = {"Content-Type": "application/json"}
    if req.callback_secret:
        headers["x-worker-secret"] = req.callback_secret
    requests.post(
        req.callback_url,
        json={
            "job_id": req.job_id,
            "project_id": req.project_id,
            "mode": "render_worker",
            "status": "processing",
            "progress_percent": progress,
            "current_step": step,
        },
        headers=headers,
        timeout=30,
    )


def _post_complete(
    req: RenderRequest,
    plan: Dict[str, Any],
    stem_urls: Dict[str, str],
    contracts: ProductionContracts,
) -> None:
    headers = {"Content-Type": "application/json"}
    if req.callback_secret:
        headers["x-worker-secret"] = req.callback_secret

    versions: List[VersionRef] = [
        VersionRef(
            version_name="Version A",
            version_type="Original",
            genre=str(plan.get("genre", "Render Worker")),
            preview_audio_path=stem_urls.get("mix"),
            full_audio_path=stem_urls.get("mix"),
            artist_vocal_path=None,
        )
    ]
    stems: List[StemRef] = [StemRef(name=n, url=u) for n, u in stem_urls.items()]

    payload = {
        "job_id": req.job_id,
        "project_id": req.project_id,
        "mode": "render_worker",
        "status": "completed",
        "progress_percent": 100,
        "current_step": "Ready",
        "analysis": req.analysis,
        "arrangement_plan": plan,
        "midi_render_contract": req.midi_render_contract,
        "production_contracts": contracts.model_dump(),
        "versions": [v.model_dump() for v in versions],
        "stems": [s.model_dump() for s in stems],
    }
    r = requests.post(req.callback_url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


app = FastAPI(title="Sing2Song Phase 2 Render Worker")

# Optional pipeline routes (/analyze, /arrange) co-hosted for single Render deploy.
_PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
_PIPELINE_ANALYZE = None
_PIPELINE_ARRANGE = None
if _PIPELINE_DIR.is_dir():
    import sys

    sys.path.insert(0, str(_PIPELINE_DIR))
    try:
        from analyze_audio import analyze as pipeline_analyze  # type: ignore
        from render_arrangement import build_arrangement as pipeline_arrange  # type: ignore

        _PIPELINE_ANALYZE = pipeline_analyze
        _PIPELINE_ARRANGE = pipeline_arrange
    except Exception:
        _PIPELINE_ANALYZE = None
        _PIPELINE_ARRANGE = None


def _pipeline_auth_ok(request: Request) -> None:
    token = os.getenv("PIPELINE_AUTH_TOKEN", "").strip()
    if not token:
        return
    header = request.headers.get("x-pipeline-auth", "").strip()
    if header != token:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health() -> Dict[str, str]:
    return {
        "ok": "true",
        "pipeline": "ready" if _PIPELINE_ANALYZE is not None else "unavailable",
    }


class PipelineAnalyzeBody(BaseModel):
    audio_url: str
    export_midi: bool = True


class PipelineArrangeBody(BaseModel):
    analysis: Dict[str, Any] = Field(default_factory=dict)
    genre: str = "Pop"


@app.post("/analyze")
def pipeline_analyze_route(body: PipelineAnalyzeBody, request: Request) -> Dict[str, Any]:
    if _PIPELINE_ANALYZE is None:
        raise HTTPException(status_code=503, detail="pipeline analysis unavailable")
    _pipeline_auth_ok(request)
    if not body.audio_url.strip():
        raise HTTPException(status_code=400, detail="audio_url required")
    with tempfile.TemporaryDirectory(prefix="s2s_analyze_") as tmp:
        wav = Path(tmp) / "input.wav"
        midi_out = Path(tmp) / "melody.mid" if body.export_midi else None
        res = requests.get(body.audio_url.strip(), timeout=180)
        if not res.ok:
            raise HTTPException(status_code=502, detail=f"audio download failed ({res.status_code})")
        wav.write_bytes(res.content)
        result = _PIPELINE_ANALYZE(str(wav), midi_output=str(midi_out) if midi_out else None)
        result.setdefault("f0_series", [])
        return result


@app.post("/arrange")
def pipeline_arrange_route(body: PipelineArrangeBody, request: Request) -> Dict[str, Any]:
    if _PIPELINE_ARRANGE is None:
        raise HTTPException(status_code=503, detail="pipeline arrange unavailable")
    _pipeline_auth_ok(request)
    if not body.analysis:
        raise HTTPException(status_code=400, detail="analysis required")
    return _PIPELINE_ARRANGE(body.analysis, genre=body.genre or "Pop")


@app.post("/render")
def render(req: RenderRequest, request: Request) -> Dict[str, Any]:
    cfg = _worker_config()
    if cfg.worker_auth_token:
        incoming = request.headers.get("x-worker-auth", "").strip()
        if incoming != cfg.worker_auth_token:
            raise HTTPException(status_code=401, detail="Unauthorized worker caller")

    if req.mode != "render_worker":
        raise HTTPException(status_code=400, detail="Unsupported mode")

    plan = req.arrangement_plan or {}
    if not plan:
        plan = {
            "bpm": req.analysis.get("bpm", 90),
            "key": req.analysis.get("key", "C"),
            "sections": req.analysis.get("sections", []),
            "chord_progression": req.analysis.get("chord_progression", []),
            "genre": "Render Worker",
        }
    contracts = ProductionContracts(
        arrangement=plan,
        vocal_chain=plan.get("vocal_chain", {}),
        mix_chain=plan.get("mix_chain", {}),
        mastering_chain=plan.get("mastering_chain", {}),
    )
    midi_contract = _build_midi_contract_from_plan(plan, req)

    try:
        _post_progress(req, 20, "Building MIDI contract")
        with tempfile.TemporaryDirectory(prefix="s2s-render-") as td:
            td_path = Path(td)
            total_sec = _section_duration_sec(plan)
            midi_path = td_path / "arrangement.mid"
            midi_meta = _build_midi_from_contract(midi_contract, midi_path, total_sec)

            _post_progress(req, 36, "Rendering MIDI")
            out_dir = Path(td) / "stems"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Engine priority for high-end instrument rendering:
            #   1) pro sampled-instrument render service (Kontakt/Spitfire-class)
            #   2) FluidSynth (local sampled soundfont)
            #   3) deterministic synth fallback (flow validation)
            engine = "deterministic_synth_fallback"
            sample_engine_stems = _render_midi_with_sample_service(
                cfg,
                midi_path,
                td_path / "sample_stems",
                midi_contract,
            )

            # Always produce a deterministic baseline so every render target has
            # audio, then override with higher-fidelity engines where available.
            stems = _render_stems(plan, out_dir, seed_key=f"{req.project_id}:{req.job_id}")

            midi_mix = td_path / "midi_mix.wav"
            rendered = _render_midi_with_fluidsynth(midi_path, midi_mix, cfg.sf2_path)
            if rendered:
                engine = "fluidsynth"
                y, sr = sf.read(str(midi_mix), always_2d=False)
                if isinstance(y, np.ndarray) and y.ndim > 1:
                    y = np.mean(y, axis=1)
                y = np.asarray(y, dtype=np.float32)
                peak = float(np.max(np.abs(y))) if y.size else 0.0
                if peak > 0:
                    y = y * min(0.98 / peak, 1.0)
                sf.write(str(out_dir / "mix.wav"), y, sr if sr > 0 else SAMPLE_RATE, subtype="PCM_16")
                stems["mix"] = out_dir / "mix.wav"

            if sample_engine_stems:
                # Pro sampled stems take precedence (highest fidelity).
                engine = "sample_render_service"
                stems.update(sample_engine_stems)
                if "mix" in sample_engine_stems:
                    stems["mix"] = sample_engine_stems["mix"]

            # ============================================================
            # VOCAL & MASTERING DSP HOOKS
            # ============================================================
            artist_vocal_url = req.analysis.get("original_audio_path") or plan.get("vocal_url") or req.analysis.get("audio_url")
            
            # If a vocal is provided, apply vocal processing, mix, and master the output
            if artist_vocal_url:
                try:
                    _post_progress(req, 50, "Processing vocal track")
                    raw_vocal_path = td_path / "vocal_raw.wav"
                    res_voc = requests.get(artist_vocal_url, timeout=180)
                    if res_voc.ok:
                        raw_vocal_path.write_bytes(res_voc.content)
                        
                        # Dynamically load DSP modules
                        try:
                            from vocal_dsp import process_vocal
                            from mastering_dsp import mix_and_master
                            
                            _post_progress(req, 60, "Aligning and tuning vocal")
                            # Process vocal (autotune, gate, de-ess, harmonies, alignment)
                            vocal_stems = process_vocal(
                                str(raw_vocal_path),
                                midi_contract,
                                plan.get("vocal_chain", {})
                            )
                            
                            for k_v, p_v in vocal_stems.items():
                                stems[k_v] = Path(p_v)
                            
                            _post_progress(req, 68, "Mixing and mastering final release")
                            # Mix and master final stereo output
                            mix_wav_path = out_dir / "mix.wav"
                            mix_and_master(stems, str(mix_wav_path), plan.get("mix_chain", {}), plan.get("mastering_chain", {}))
                            stems["mix"] = mix_wav_path
                            
                            engine += "+vocal_dsp+mastering_dsp"
                        except Exception as dsp_err:
                            print(f"DSP Pipeline failed, using baseline: {dsp_err}")
                except Exception as voc_err:
                    print(f"Vocal download or processing failed: {voc_err}")

            _post_progress(req, 72, "Uploading stems")

            stem_urls: Dict[str, str] = {}
            for name, p in stems.items():
                remote = f"{req.project_id}/{req.job_id}/{name}.wav"
                stem_urls[name] = _upload_file(p, remote, cfg)

            _post_progress(req, 92, "Finalizing song version")
            # Attach runtime render metadata for callback introspection.
            req.midi_render_contract = {
                **midi_contract,
                "runtime": {
                    "engine": engine,
                    "release_grade": engine in ("sample_render_service", "fluidsynth") or "vocal_dsp" in engine,
                    "sample_render_service": bool(sample_engine_stems),
                    "fluidsynth": bool(rendered),
                    "midi_meta": midi_meta,
                },
            }
            _post_complete(req, plan, stem_urls, contracts)

        return {
            "ok": True,
            "job_id": req.job_id,
            "project_id": req.project_id,
            "stems": stem_urls,
        }
    except Exception as e:
        headers = {"Content-Type": "application/json"}
        if req.callback_secret:
            headers["x-worker-secret"] = req.callback_secret
        requests.post(
            req.callback_url,
            json={
                "job_id": req.job_id,
                "project_id": req.project_id,
                "mode": "render_worker",
                "status": "failed",
                "current_step": "Failed",
                "error_message": str(e),
            },
            headers=headers,
            timeout=30,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("render_worker:app", host="0.0.0.0", port=8081, reload=True)
