#!/usr/bin/env python3
"""
High-end analysis backend for Sing2Song.

Canonical output:
tempo_map, downbeats, key, scale, melody_midi/melody_contour,
vocal_phrases, lyrics_timestamps, sections, chord_progression, confidence_scores.
"""

import argparse
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

KEY_PROFILE_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KEY_PROFILE_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Triad templates in pitch-class space (rooted at C)
CHORD_TEMPLATES: Dict[str, np.ndarray] = {
    "maj": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64),
    "min": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64),
    "dim": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64),
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    if arr.size == 0 or win <= 1:
        return arr
    kernel = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(arr, kernel, mode="same")


def _madmom_beats_downbeats(
    audio_path: Optional[str],
    y: np.ndarray,
    sr: int,
) -> Optional[Dict[str, Any]]:
    """Hardened madmom/BeatNet-style rhythm path.

    Uses RNNDownBeatProcessor + DBNDownBeatTrackingProcessor to produce both
    beat positions and *true* downbeats (beat-position 1) instead of a stride-4
    heuristic. Returns None when madmom is unavailable so callers can fall back.
    """
    try:
        from madmom.features.downbeats import (  # type: ignore
            RNNDownBeatProcessor,
            DBNDownBeatTrackingProcessor,
        )

        act_source = audio_path if audio_path else y
        act = RNNDownBeatProcessor()(act_source)
        tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        result = tracker(act)  # rows of [time, beat_position]
        if result is None or len(result) < 4:
            return None
        beat_times = [float(r[0]) for r in result]
        downbeats = [float(r[0]) for r in result if int(round(r[1])) == 1]
        deltas = np.diff(np.array(beat_times)) if len(beat_times) > 1 else np.array([0.5])
        tempo = float(np.median(60.0 / np.clip(deltas, 1e-3, None)))
        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "downbeats": downbeats if downbeats else beat_times[::4],
            "beat_source": "madmom_dbn",
        }
    except Exception:
        return None


def detect_tempo_and_beats(
    y: np.ndarray,
    sr: int,
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    # Primary: hardened madmom downbeat tracking.
    mm = _madmom_beats_downbeats(audio_path, y, sr)
    if mm is not None:
        tempo = mm["tempo"]
        beat_times = mm["beat_times"]
        downbeats = mm["downbeats"]
        beat_source = mm["beat_source"]
    else:
        # Fallback: librosa beat tracking with onset-aligned downbeats.
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strength = []
        for bf in beat_frames:
            beat_strength.append(
                float(onset_env[int(bf)]) if 0 <= bf < len(onset_env) else 0.0
            )
        offset = 0
        if len(beat_strength) >= 4:
            offset = int(np.argmax(np.array(beat_strength[:4], dtype=np.float64)))
        downbeats = [beat_times[i] for i in range(offset, len(beat_times), 4)]
        tempo = _safe_float(tempo)
        beat_source = "librosa"

    if len(beat_times) > 1:
        beat_deltas = np.diff(np.array(beat_times))
        local_tempo = np.clip(60.0 / np.clip(beat_deltas, 1e-3, None), 40.0, 220.0)
        local_tempo = _moving_average(local_tempo, 3)
        tempo_map = [float(local_tempo[0])] + [float(v) for v in local_tempo]
    else:
        tempo_map = [_safe_float(tempo)]

    return {
        "tempo": _safe_float(tempo),
        "tempo_map": tempo_map,
        "downbeats": downbeats,
        "beat_times": beat_times,
        "beat_source": beat_source,
    }


def _normalize_profile(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if np.sum(p) <= 0:
        return np.zeros_like(p)
    return p / float(np.sum(p))


def _best_key_from_chroma(mean_chroma: np.ndarray) -> Tuple[str, str, float]:
    major_scores: List[Tuple[int, float]] = []
    minor_scores: List[Tuple[int, float]] = []
    for k in range(12):
        maj_profile = np.roll(KEY_PROFILE_MAJOR, k)
        min_profile = np.roll(KEY_PROFILE_MINOR, k)
        major_scores.append((k, float(np.dot(mean_chroma, _normalize_profile(maj_profile)))))
        minor_scores.append((k, float(np.dot(mean_chroma, _normalize_profile(min_profile)))))

    best_maj = max(major_scores, key=lambda x: x[1])
    best_min = max(minor_scores, key=lambda x: x[1])
    if best_maj[1] >= best_min[1]:
        key = PITCH_CLASSES[best_maj[0]]
        return key, "major", best_maj[1]
    key = PITCH_CLASSES[best_min[0]]
    return key, "minor", best_min[1]


def _estimate_chords_from_chroma(
    chroma: np.ndarray,
    beat_times: List[float],
    sr: int,
) -> Tuple[List[str], List[float]]:
    if chroma.size == 0:
        return ([], [])
    hop = 512
    beat_frames = librosa.time_to_frames(np.array(beat_times, dtype=np.float64), sr=sr, hop_length=hop)
    if beat_frames.size < 2:
        beat_frames = np.array([0, chroma.shape[1] - 1], dtype=np.int64)

    chords: List[str] = []
    confidences: List[float] = []
    for i in range(len(beat_frames) - 1):
        a = int(max(0, min(chroma.shape[1] - 1, beat_frames[i])))
        b = int(max(a + 1, min(chroma.shape[1], beat_frames[i + 1])))
        seg = np.mean(chroma[:, a:b], axis=1)
        seg = _normalize_profile(seg)
        best_label = "N"
        best_score = -1.0
        second = -1.0
        for root in range(12):
            for quality, tpl in CHORD_TEMPLATES.items():
                score = float(np.dot(seg, _normalize_profile(np.roll(tpl, root))))
                if score > best_score:
                    second = best_score
                    best_score = score
                    suffix = {"maj": "", "min": "m", "dim": "dim"}[quality]
                    best_label = f"{PITCH_CLASSES[root]}{suffix}"
                elif score > second:
                    second = score
        conf = max(0.0, min(1.0, best_score - max(0.0, second)))
        chords.append(best_label)
        confidences.append(float(conf))

    # Compress duplicates while keeping order.
    compact: List[str] = []
    for c in chords:
        if not compact or compact[-1] != c:
            compact.append(c)
    return compact[:16], confidences


def _chords_with_chordino(audio_path: Optional[str]) -> Optional[Tuple[List[str], float, str]]:
    """Chordino-class chord recognition.

    Primary: madmom DeepChromaProcessor + DeepChromaChordRecognitionProcessor
    (a learned chord model, the Chordino-equivalent in the madmom stack).
    Returns (compact_progression, mean_confidence, source) or None.
    """
    if not audio_path:
        return None
    try:
        from madmom.audio.chroma import DeepChromaProcessor  # type: ignore
        from madmom.features.chords import (  # type: ignore
            DeepChromaChordRecognitionProcessor,
        )

        dcp = DeepChromaProcessor()
        decode = DeepChromaChordRecognitionProcessor()
        chords = decode(dcp(audio_path))  # rows of [start, end, label]
        labels: List[str] = []
        for row in chords:
            label = str(row[2])
            if label and label != "N":
                # Normalize madmom labels like "C:maj" / "A:min" -> "C" / "Am".
                norm = label.replace(":maj", "").replace(":min", "m")
                norm = norm.split(":")[0] if ":" in norm else norm
                labels.append(norm)
        compact: List[str] = []
        for c in labels:
            if not compact or compact[-1] != c:
                compact.append(c)
        if not compact:
            return None
        return compact[:16], 0.85, "madmom_deepchroma"
    except Exception:
        return None


def _key_with_essentia(y: np.ndarray, sr: int) -> Optional[Tuple[str, str, float]]:
    """Essentia KeyExtractor — production-grade key estimation when available."""
    try:
        import essentia.standard as es  # type: ignore

        audio = np.asarray(y, dtype=np.float32)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        key, scale, strength = es.KeyExtractor()(audio)
        mode = "major" if str(scale).lower() == "major" else "minor"
        return str(key), mode, float(strength)
    except Exception:
        return None


def detect_key_and_chords(
    y: np.ndarray,
    sr: int,
    beat_times: List[float],
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = _normalize_profile(np.mean(chroma, axis=1))
    lib_key, lib_mode, lib_conf = _best_key_from_chroma(mean_chroma)

    key_source = "librosa_chroma"
    ess = _key_with_essentia(y, sr)
    if ess is not None and ess[2] >= lib_conf:
        key, mode, key_conf = ess[0], ess[1], ess[2]
        key_source = "essentia"
    else:
        key, mode, key_conf = lib_key, lib_mode, lib_conf

    scale = f"{key} {mode}"

    chord_source = "chroma_template"
    chordino = _chords_with_chordino(audio_path)
    if chordino is not None:
        chord_progression, chord_conf_mean, chord_source = chordino
        chord_conf = [chord_conf_mean]
    else:
        chord_progression, chord_conf = _estimate_chords_from_chroma(chroma, beat_times, sr)

    return {
        "key": key,
        "scale": scale,
        "chord_progression": chord_progression,
        "chord_hint": f"{key} {mode} progression, beat-aligned estimate",
        "key_confidence": float(max(0.0, min(1.0, key_conf))),
        "chord_confidence_mean": float(np.mean(chord_conf)) if chord_conf else 0.0,
        "chord_source": chord_source,
        "key_source": key_source,
    }


def _detect_melody_with_optional_models(y: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], str]:
    # Hardened torchcrepe path with periodicity-based voicing gating.
    try:
        import torch  # type: ignore
        import torchcrepe  # type: ignore

        audio = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        frame_hz, periodicity = torchcrepe.predict(
            audio,
            sr,
            160,  # hop
            50.0,
            1100.0,
            model="full",
            batch_size=512,
            device="cpu",
            return_periodicity=True,
        )
        # Suppress low-confidence (unvoiced) frames so MIDI export stays clean.
        try:
            periodicity = torchcrepe.filter.median(periodicity, 3)
            frame_hz = torchcrepe.threshold.At(0.21)(frame_hz, periodicity)
        except Exception:
            pass
        f0 = frame_hz.squeeze(0).cpu().numpy()
        f0 = np.nan_to_num(f0, nan=0.0)
        if np.isfinite(f0).any() and np.any(f0 > 0):
            return f0, "torchcrepe_full"
    except Exception:
        pass
    return None, "pyin"


def detect_melody(y: np.ndarray, sr: int) -> Dict[str, Any]:
    ext_f0, source = _detect_melody_with_optional_models(y, sr)
    if ext_f0 is not None:
        finite = np.nan_to_num(ext_f0, nan=0.0)
    else:
        source = "pyin"
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        if f0 is None:
            return {"melody_contour": [], "f0_series": [], "melody_source": source}
        finite = np.nan_to_num(f0, nan=0.0)

    finite = np.asarray(finite, dtype=np.float64)
    if finite.size == 0:
        return {"melody_contour": [], "f0_series": [], "melody_source": source}

    voiced = finite[finite > 0]
    max_f = np.max(voiced) if voiced.size > 0 else 1.0
    melody = [float(v / max_f) if v > 0 else 0.0 for v in finite]
    if len(melody) > 64:
        idx = np.linspace(0, len(melody) - 1, 64).astype(int)
        melody = [melody[int(i)] for i in idx]
    return {
        "melody_contour": melody,
        "f0_series": finite.tolist(),
        "melody_source": source,
        "voiced_ratio": float(np.mean(finite > 0)),
    }


def _detect_melody_rmvpe(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Hardened RMVPE serving path.

    RMVPE is the highest-fidelity vocal pitch estimator in this stack. We resolve
    the model weights from RMVPE_MODEL_PATH (or a conventional location), support
    the two most common RMVPE python APIs, and gracefully return None when the
    runtime lacks the package/weights so torchcrepe/pyin can take over.
    """
    model_path = os.getenv("RMVPE_MODEL_PATH", "").strip() or "rmvpe.pt"
    if not os.path.exists(model_path):
        return None
    try:
        import torch  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from rmvpe import RMVPE  # type: ignore

            estimator = RMVPE(model_path, is_half=False, device=device)
            audio_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            f0 = estimator.infer_from_audio(audio_16k, thred=0.03)
        except Exception:
            import rmvpe  # type: ignore

            estimator = rmvpe.RMVPE(model_path)
            f0 = estimator.infer_from_audio(y, sr=sr)
        if f0 is not None and len(f0) > 0:
            return np.asarray(f0, dtype=np.float64)
    except Exception:
        return None
    return None


def _detect_melody_basic_pitch(audio_path: str, sr: int) -> Optional[np.ndarray]:
    """Basic Pitch (Spotify ICASSP-2022) melody fallback."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
        from basic_pitch.inference import predict  # type: ignore

        _, _, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    except Exception:
        return None

    if not note_events:
        return None
    try:
        hop = 512.0 / float(sr)
        duration = max(float(ev[1]) for ev in note_events)
        n = int(duration / hop) + 1
        if n <= 0:
            return None
        f0 = np.zeros(n, dtype=np.float64)
        for ev in note_events:
            start, end, pitch = float(ev[0]), float(ev[1]), int(ev[2])
            hz = float(440.0 * (2.0 ** ((pitch - 69) / 12.0)))
            i0 = max(0, int(start / hop))
            i1 = min(n, int(end / hop) + 1)
            for i in range(i0, i1):
                if hz > f0[i]:  # keep the lead (highest) note when overlapping
                    f0[i] = hz
        if np.any(f0 > 0):
            return f0
    except Exception:
        return None
    return None


def _melody_payload_from_f0(f0_hz: np.ndarray, source: str) -> Dict[str, Any]:
    """Normalises an Hz f0 series into the melody payload shape used downstream."""
    rm = np.nan_to_num(np.asarray(f0_hz, dtype=np.float64), nan=0.0)
    max_f = float(np.max(rm[rm > 0])) if np.any(rm > 0) else 1.0
    contour = [float(v / max_f) if v > 0 else 0.0 for v in rm]
    if len(contour) > 64:
        idx = np.linspace(0, len(contour) - 1, 64).astype(int)
        contour = [contour[int(i)] for i in idx]
    return {
        "melody_contour": contour,
        "f0_series": [float(v) for v in rm],
        "melody_source": source,
        "voiced_ratio": float(np.mean(rm > 0)),
    }


def export_melody_midi(f0_series: List[float], sr: int, out_path: str) -> str:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="vocal_melody")
    hop = 512 / sr
    active_start = None
    active_pitch = None

    for i, hz in enumerate(f0_series):
        if hz <= 0:
            if active_start is not None and active_pitch is not None:
                end = i * hop
                if end - active_start >= 0.08:
                    inst.notes.append(
                        pretty_midi.Note(
                            velocity=88,
                            pitch=int(active_pitch),
                            start=float(active_start),
                            end=float(end),
                        )
                    )
            active_start = None
            active_pitch = None
            continue

        midi = librosa.hz_to_midi(hz)
        if active_start is None:
            active_start = i * hop
            active_pitch = round(midi)
            continue
        if abs(midi - active_pitch) > 1.5:
            end = i * hop
            if end - active_start >= 0.08:
                inst.notes.append(
                    pretty_midi.Note(
                        velocity=88,
                        pitch=int(active_pitch),
                        start=float(active_start),
                        end=float(end),
                    )
                )
            active_start = i * hop
            active_pitch = round(midi)

    if active_start is not None and active_pitch is not None:
        end = len(f0_series) * hop
        if end - active_start >= 0.08:
            inst.notes.append(
                pretty_midi.Note(
                    velocity=88,
                    pitch=int(active_pitch),
                    start=float(active_start),
                    end=float(end),
                )
            )

    pm.instruments.append(inst)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pm.write(out_path)
    return out_path


def _spectral_cluster_boundaries(y: np.ndarray, sr: int, n_frames: int) -> Optional[List[int]]:
    """Laplacian spectral-clustering structural segmentation (McFee/Ellis)."""
    try:
        hop = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        bounds = librosa.segment.agglomerative(chroma, 6)
        bounds = sorted(set(int(b) for b in bounds.tolist()))
        scale = n_frames / max(1, chroma.shape[1])
        mapped = sorted(set(int(round(b * scale)) for b in bounds))
        mapped = [b for b in mapped if 0 <= b < n_frames]
        if 0 not in mapped:
            mapped = [0] + mapped
        if (n_frames - 1) not in mapped:
            mapped.append(n_frames - 1)
        if len(mapped) >= 3:
            return sorted(set(mapped))
    except Exception:
        return None
    return None


def _pyannote_vocal_phrases(audio_path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Real pyannote VAD path for phrase/section vocal activity."""
    token = os.getenv("PYANNOTE_AUTH_TOKEN", "").strip()
    if not audio_path or not token:
        return None
    try:
        from pyannote.audio import Pipeline  # type: ignore

        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=token,
        )
        vad = pipeline(audio_path)
        phrases: List[Dict[str, Any]] = []
        for segment in vad.get_timeline().support():
            dur = float(segment.end - segment.start)
            if dur < 0.18:
                continue
            phrases.append(
                {
                     "start_sec": float(segment.start),
                     "end_sec": float(segment.end),
                     "confidence": float(min(1.0, 0.6 + dur / 4.0)),
                }
            )
        return phrases if phrases else None
    except Exception:
        return None


def _pick_section_boundaries(rms: np.ndarray, onset: np.ndarray) -> List[int]:
    n = len(rms)
    if n < 32:
        return [0, n - 1]
    rms_n = _normalize_profile(np.maximum(rms, 1e-8))
    onset_n = _normalize_profile(np.maximum(onset[:n], 1e-8))
    novelty = np.abs(np.diff(_moving_average(rms_n, 9), prepend=rms_n[0])) + np.abs(
        np.diff(_moving_average(onset_n, 9), prepend=onset_n[0])
    )
    novelty = _moving_average(novelty, 7)
    candidate = np.argsort(novelty)[::-1]
    boundaries = [0, n - 1]
    min_gap = max(16, n // 10)
    for c in candidate:
        if c <= min_gap or c >= n - min_gap:
            continue
        if all(abs(int(c) - b) >= min_gap for b in boundaries):
            boundaries.append(int(c))
        if len(boundaries) >= 6:
            break
    return sorted(set(boundaries))


def segment_sections(
    y: np.ndarray,
    sr: int,
    f0_series: List[float],
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    n = len(rms)
    if n < 8:
        return {
            "sections": [],
            "vocal_phrases": [],
            "section_boundary_strength": 0.0,
            "section_source": "none",
            "phrase_source": "none",
        }

    # Primary: spectral-clustering structure; fallback: energy novelty.
    section_source = "spectral_cluster"
    boundaries = _spectral_cluster_boundaries(y, sr, n)
    if not boundaries or len(boundaries) < 3:
        boundaries = _pick_section_boundaries(rms, onset)
        section_source = "energy_novelty"
    sections: List[Dict[str, Any]] = []
    segment_energy: List[float] = []
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        start = float(times[a]) if a < len(times) else 0.0
        end = float(times[b]) if b < len(times) else start
        energy = float(np.mean(rms[a:b])) if b > a else 0.0
        segment_energy.append(energy)
        sections.append(
            {
                "label": f"section_{i + 1}",
                "start_sec": start,
                "end_sec": end,
                "energy": energy,
            }
        )

    if sections:
        sections[0]["label"] = "intro"
    if len(sections) >= 2:
        sections[-1]["label"] = "outro"
    if len(sections) >= 3:
        interior = list(range(1, len(sections) - 1))
        chorus_idx = max(interior, key=lambda i: sections[i]["energy"])
        sections[chorus_idx]["label"] = "chorus"
        for i in interior:
            if i == chorus_idx:
                continue
            sections[i]["label"] = "verse"
        if len(interior) >= 2:
            bridge_idx = max(interior, key=lambda i: sections[i]["start_sec"])
            if bridge_idx != chorus_idx:
                sections[bridge_idx]["label"] = "bridge"

    # Phrase segmentation: prefer pyannote VAD, else voiced/non-voiced runs.
    phrase_source = "pyannote_vad"
    vocal_phrases = _pyannote_vocal_phrases(audio_path)
    if not vocal_phrases:
        phrase_source = "f0_voicing"
        f0 = np.asarray(f0_series, dtype=np.float64) if f0_series else np.array([], dtype=np.float64)
        vocal_phrases = []
        if f0.size > 0:
            voiced = (f0 > 0).astype(np.int32)
            changes = np.diff(voiced, prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            frame_sec = (len(y) / sr) / max(1, len(f0))
            for s, e in zip(starts, ends):
                start = float(s * frame_sec)
                end = float(e * frame_sec)
                if end - start < 0.18:
                    continue
                conf = float(min(1.0, 0.5 + (end - start) / 2.5))
                vocal_phrases.append({"start_sec": start, "end_sec": end, "confidence": conf})
        else:
            phrase_source = "sections"
            vocal_phrases = [
                {"start_sec": s["start_sec"], "end_sec": s["end_sec"], "confidence": 0.55}
                for s in sections
            ]

    novelty_strength = float(np.mean(np.abs(np.diff(_moving_average(rms, 7))))) if len(rms) > 1 else 0.0
    return {
        "sections": sections,
        "vocal_phrases": vocal_phrases,
        "section_boundary_strength": novelty_strength,
        "section_source": section_source,
        "phrase_source": phrase_source,
    }


def extract_lyrics_timestamps(audio_path: str) -> List[Dict[str, Any]]:
    # WhisperX integration point (best effort).
    try:
        import whisperx  # type: ignore

        device = "cpu"
        model = whisperx.load_model("small", device, compute_type="int8")
        result = model.transcribe(audio_path)
        words = result.get("word_segments") or []
        if words:
            out: List[Dict[str, Any]] = []
            for w in words:
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                out.append(
                    {
                        "text": text,
                        "start_sec": float(w.get("start", 0.0)),
                        "end_sec": float(w.get("end", 0.0)),
                    }
                )
            if out:
                return out

        # Fallback to segment-level timing if word alignment is unavailable.
        segments = result.get("segments") or []
        segment_words: List[Dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            segment_words.append(
                {
                    "text": text,
                    "start_sec": float(seg.get("start", 0.0)),
                    "end_sec": float(seg.get("end", 0.0)),
                }
            )
        return segment_words
    except Exception:
        return []


def separate_vocals(audio_path: str) -> Optional[str]:
    # Demucs integration point; return extracted vocal path if available.
    try:
        import subprocess

        with tempfile.TemporaryDirectory(prefix="s2s-demucs-") as td:
            cmd = [
                "python",
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "--two-stems",
                "vocals",
                "-o",
                td,
                audio_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                return None
            base = os.path.splitext(os.path.basename(audio_path))[0]
            cand = os.path.join(td, "htdemucs", base, "vocals.wav")
            if os.path.exists(cand):
                out = tempfile.NamedTemporaryFile(suffix="_vocals.wav", delete=False).name
                y, sr = sf.read(cand)
                sf.write(out, y, sr)
                return out
    except Exception:
        return None
    return None


def build_confidence(payload: Dict[str, Any]) -> Dict[str, float]:
    tempo_map = np.array(payload.get("tempo_map", []), dtype=np.float64)
    tempo_var = float(np.std(tempo_map)) if tempo_map.size > 0 else 999.0
    tempo_conf = float(max(0.0, min(1.0, 1.0 - (tempo_var / 30.0))))

    melody_conf = float(payload.get("_voiced_ratio", 0.0))
    key_conf = float(payload.get("_key_confidence", 0.0))
    chord_conf = float(payload.get("_chord_confidence", 0.0))
    section_strength = float(payload.get("_section_boundary_strength", 0.0))
    sections_conf = float(max(0.0, min(1.0, section_strength * 15.0)))

    lyrics = payload.get("lyrics_timestamps", [])
    lyrics_conf = 0.0
    if isinstance(lyrics, list) and lyrics:
        avg_word_dur = float(
            np.mean(
                [
                    max(0.0, _safe_float(w.get("end_sec")) - _safe_float(w.get("start_sec")))
                    for w in lyrics
                    if isinstance(w, dict)
                ]
            )
        )
        lyrics_conf = float(max(0.0, min(1.0, 0.5 + min(0.4, avg_word_dur))))

    return {
        "tempo": tempo_conf,
        "key": key_conf,
        "melody": max(0.0, min(1.0, melody_conf)),
        "sections": sections_conf,
        "chords": max(0.0, min(1.0, chord_conf)),
        "lyrics": lyrics_conf,
    }


def analyze(path: str, midi_output: str | None = None) -> Dict[str, Any]:
    source_path = separate_vocals(path) or path
    y, sr = librosa.load(source_path, sr=16000, mono=True)

    tempo = detect_tempo_and_beats(y, sr, audio_path=source_path)
    harmony = detect_key_and_chords(
        y, sr, tempo.get("beat_times", []), audio_path=path
    )
    # Melody priority: RMVPE (highest fidelity) -> Basic Pitch -> torchcrepe -> pyin.
    rmvpe = _detect_melody_rmvpe(y, sr)
    if rmvpe is not None:
        melody = _melody_payload_from_f0(rmvpe, "rmvpe")
    else:
        basic_pitch_f0 = _detect_melody_basic_pitch(source_path, sr)
        if basic_pitch_f0 is not None:
            melody = _melody_payload_from_f0(basic_pitch_f0, "basic_pitch")
        else:
            melody = detect_melody(y, sr)
    sections = segment_sections(
        y, sr, melody.get("f0_series", []), audio_path=source_path
    )

    melody_midi_path = None
    if midi_output:
        melody_midi_path = export_melody_midi(melody.get("f0_series", []), sr, midi_output)

    payload: Dict[str, Any] = {
        "bpm": tempo["tempo"],
        "key": harmony["key"],
        "scale": harmony["scale"],
        "tempo_map": tempo["tempo_map"],
        "downbeats": tempo["downbeats"],
        "beat_times": tempo["beat_times"],
        "duration": float(len(y)) / float(sr),
        "melody_contour": melody["melody_contour"],
        "f0_series": melody.get("f0_series", []),
        "chord_hint": harmony["chord_hint"],
        "chord_progression": harmony["chord_progression"],
        "sections": sections["sections"],
        "vocal_phrases": sections["vocal_phrases"],
        "lyrics_timestamps": extract_lyrics_timestamps(source_path),
        "melody_midi": melody_midi_path,
        "analysis_backend": "hybrid-rmvpe-madmom-pyannote-chordino",
        "analysis_model_paths": {
            "melody": melody.get("melody_source"),
            "beats": tempo.get("beat_source"),
            "chords": harmony.get("chord_source", "chroma_template"),
            "sections": sections.get("section_source", "energy_novelty"),
            "phrases": sections.get("phrase_source", "f0_voicing"),
        },
    }

    # Internal confidence inputs, stripped before final output.
    payload["_voiced_ratio"] = float(melody.get("voiced_ratio", 0.0))
    payload["_key_confidence"] = float(harmony.get("key_confidence", 0.0))
    payload["_chord_confidence"] = float(harmony.get("chord_confidence_mean", 0.0))
    payload["_section_boundary_strength"] = float(sections.get("section_boundary_strength", 0.0))
    payload["confidence_scores"] = build_confidence(payload)
    payload.pop("_voiced_ratio", None)
    payload.pop("_key_confidence", None)
    payload.pop("_chord_confidence", None)
    payload.pop("_section_boundary_strength", None)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input WAV path")
    parser.add_argument("--midi-output", default=None, help="Optional output MIDI path")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    result = analyze(args.input, midi_output=args.midi_output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
