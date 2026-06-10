#!/usr/bin/env python3
"""
Arrangement + render planner.
Consumes analysis JSON and outputs a section-aware render plan for stems.
"""

import argparse
import json
from typing import Any, Dict, List, Tuple


def _energy_bucket(v: float) -> str:
    if v >= 0.62:
        return "high"
    if v >= 0.4:
        return "mid"
    return "low"


# Genre-authentic articulation vocabularies (per-instrument performance idioms).
GENRE_ARTICULATIONS: Dict[str, Dict[str, Any]] = {
    "afro": {
        "drums": {"feel": "afro_shuffle", "ghost_density": 0.7, "swing": 0.56},
        "bass": {"style": "syncopated_pluck", "slide": True, "mute_palm": 0.3},
        "guitar": {"style": "highlife_arpeggio", "palm_mute": 0.2},
        "keys": {"voicing": "stacked_thirds", "comp_rhythm": "offbeat"},
        "strings": {"bowing": "legato_swell"},
    },
    "rnb": {
        "drums": {"feel": "laid_back", "ghost_density": 0.85, "swing": 0.55},
        "bass": {"style": "sub_legato", "slide": True, "mute_palm": 0.1},
        "guitar": {"style": "clean_chord_melody", "palm_mute": 0.15},
        "keys": {"voicing": "extended_9th_11th", "comp_rhythm": "sustained_lush"},
        "strings": {"bowing": "soft_pad"},
    },
    "rock": {
        "drums": {"feel": "driving_8th", "ghost_density": 0.3, "swing": 0.5},
        "bass": {"style": "picked_root_drive", "slide": False, "mute_palm": 0.4},
        "guitar": {"style": "power_chord_stack", "palm_mute": 0.6},
        "keys": {"voicing": "wide_octaves", "comp_rhythm": "sustained"},
        "strings": {"bowing": "marcato_support"},
    },
    "pop": {
        "drums": {"feel": "tight_pop", "ghost_density": 0.4, "swing": 0.5},
        "bass": {"style": "picked_pocket", "slide": False, "mute_palm": 0.3},
        "guitar": {"style": "clean_stack", "palm_mute": 0.25},
        "keys": {"voicing": "close_triad", "comp_rhythm": "8th_pulse"},
        "strings": {"bowing": "legato_lift"},
    },
}


def _genre_key(genre: str) -> str:
    g = (genre or "Pop").strip().lower()
    if "afro" in g:
        return "afro"
    if "r&b" in g or "rnb" in g:
        return "rnb"
    if "rock" in g:
        return "rock"
    return "pop"


def _genre_profile(genre: str) -> Dict[str, Any]:
    key = _genre_key(genre)
    profiles = {
        "afro": {
            "drums": "afrobeats_live_hybrid_kit",
            "bass": "afro_groove_bass",
            "keys": "warm_electric_piano",
            "guitar": "highlife_clean_guitar",
            "strings": "cinematic_strings_ensemble",
            "brass": "afro_pop_brass_section",
            "swing": 0.56,
            "chorus_lift_db": 2.2,
        },
        "rnb": {
            "drums": "neo_soul_punch_kit",
            "bass": "sub_melodic_bass",
            "keys": "silky_epiano_pad",
            "guitar": "ambient_clean_guitar",
            "strings": "soft_strings",
            "brass": "muted_brass_stabs",
            "swing": 0.52,
            "chorus_lift_db": 1.6,
        },
        "rock": {
            "drums": "arena_rock_kit",
            "bass": "picked_rock_bass",
            "keys": "wide_rock_keys",
            "guitar": "rhythm_stack_guitars",
            "strings": "support_strings",
            "brass": "none",
            "swing": 0.5,
            "chorus_lift_db": 2.5,
        },
        "pop": {
            "drums": "studio_acoustic_kit",
            "bass": "premium_picked_bass",
            "keys": "concert_grand_piano",
            "guitar": "clean_electric_stack",
            "strings": "cinematic_strings_ensemble",
            "brass": "pop_brass_layer",
            "swing": 0.5,
            "chorus_lift_db": 1.8,
        },
    }
    profile = dict(profiles[key])
    profile["articulations"] = GENRE_ARTICULATIONS[key]
    profile["genre_key"] = key
    return profile


def _section_role(label: str, idx: int, total: int) -> str:
    l = (label or "").lower()
    if "intro" in l:
        return "intro"
    if "chorus" in l:
        return "chorus"
    if "bridge" in l:
        return "bridge"
    if "outro" in l or idx == total - 1:
        return "outro"
    return "verse"


def _develop_motif(
    base_motif: List[int],
    role: str,
    occurrence: int,
) -> Dict[str, Any]:
    """Higher-order motif development."""
    if not base_motif:
        base_motif = [0, 2, 4, 2]
    motif = list(base_motif)
    transform = "statement"

    if role == "chorus":
        motif = [d + 2 for d in motif]  # diatonic lift
        transform = "lift_expand"
        if occurrence >= 1:
            motif = motif + motif[:2]  # extend the hook on repeats
            transform = "lift_extend"
    elif role == "bridge":
        pivot = max(motif) if motif else 0
        motif = [pivot - d for d in motif]  # inversion around the peak
        transform = "inversion"
    elif role == "verse" and occurrence >= 1:
        motif = motif[1:] + motif[:1]  # rotate for variation on repeat
        transform = "sequence_rotate"
    elif role == "outro":
        motif = motif[: max(1, len(motif) // 2)]  # fragment / wind-down
        transform = "fragmentation"

    ornament = "none"
    if occurrence >= 2:
        ornament = "passing_tones" if role in {"verse", "bridge"} else "neighbor_tones"

    return {
        "degrees": motif,
        "transform": transform,
        "ornament": ornament,
        "occurrence": occurrence,
    }


def _section_arrangement_rules(
    sections: List[Dict[str, Any]],
    chords: List[str],
    profile: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_section: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    total = len(sections)
    if total == 0:
        return by_section, transitions

    articulations = profile.get("articulations", {})
    base_motif = [0, 2, 4, 2]  # seed scale-degree contour for theme development
    role_occurrence: Dict[str, int] = {}

    prev_chord = None
    motif_memory: List[str] = []
    for i, sec in enumerate(sections):
        label = str(sec.get("label", f"section_{i+1}"))
        role = _section_role(label, i, total)
        occurrence = role_occurrence.get(role, 0)
        role_occurrence[role] = occurrence + 1
        energy = float(sec.get("energy", 0.0))
        bucket = _energy_bucket(energy)

        instrumentation = {
            "drums": {"enabled": role != "intro" or bucket != "low", "intensity": bucket},
            "bass": {"enabled": role in {"verse", "chorus", "bridge"}, "pattern": "follow_chord_roots"},
            "keys": {"enabled": True, "pattern": "voice_lead_chords"},
            "guitar": {"enabled": role in {"chorus", "bridge"}, "pattern": "lift_texture"},
            "strings": {"enabled": role in {"chorus", "outro"}, "pattern": "long_support"},
            "brass": {"enabled": role == "chorus", "pattern": "stabs"},
        }
        # Optional brass layer only when profile supports it.
        if profile.get("brass") == "none":
            instrumentation["brass"]["enabled"] = False

        chord_slice = chords[i % len(chords)] if chords else None
        if chord_slice:
            motif_memory.append(str(chord_slice))
        melodic_motif = motif_memory[-2:] if len(motif_memory) >= 2 else motif_memory
        continuity = "stable" if prev_chord == chord_slice else "transition"
        prev_chord = chord_slice

        motif_development = _develop_motif(base_motif, role, occurrence)

        by_section.append(
            {
                "label": label,
                "role": role,
                "start_sec": sec.get("start_sec"),
                "end_sec": sec.get("end_sec"),
                "energy": energy,
                "groove": {
                    "swing": profile["swing"],
                    "humanize_ms": 12 if role != "chorus" else 9,
                },
                "harmony": {
                    "chord_anchor": chord_slice,
                    "bass_root_lock": True,
                    "phrase_continuity": continuity,
                    "motif_seed": melodic_motif,
                },
                "motif_development": motif_development,
                "instrumentation": instrumentation,
                "articulation": {
                    "drums": articulations.get("drums", {}),
                    "bass": articulations.get("bass", {}),
                    "keys": articulations.get("keys", {}),
                    "guitar": articulations.get("guitar", {}),
                    "strings": articulations.get("strings", {}),
                    "phrasing": "tight_ghost_notes" if role == "chorus" else "laid_back",
                    "key_voicing": "open_voicing" if role == "chorus" else "close_voicing",
                },
            }
        )

        if i < total - 1:
            next_label = str(sections[i + 1].get("label", f"section_{i+2}"))
            next_role = _section_role(next_label, i + 1, total)
            transitions.append(
                {
                    "from": label,
                    "to": next_label,
                    "fill_before_change": True,
                    "drum_fill_beats": 1 if next_role == "chorus" else 0.5,
                    "chorus_lift_db": profile["chorus_lift_db"] if next_role == "chorus" else 0.0,
                    "breakdown": role == "bridge" and next_role == "chorus",
                }
            )
    return by_section, transitions


def build_arrangement(analysis: Dict[str, Any], genre: str = "Pop") -> Dict[str, Any]:
    bpm = analysis.get("bpm", 90)
    sections = analysis.get("sections", []) if isinstance(analysis.get("sections"), list) else []
    chords = analysis.get("chord_progression", []) if isinstance(analysis.get("chord_progression"), list) else []
    downbeats = analysis.get("downbeats", []) if isinstance(analysis.get("downbeats"), list) else []
    profile = _genre_profile(genre)
    by_section, transitions = _section_arrangement_rules(sections, chords, profile)

    stems: List[Dict[str, Any]] = [
        {
            "name": "drums",
            "instrument": profile["drums"],
            "rules": ["lock_to_downbeats", "fills_pre_section_change"],
        },
        {
            "name": "bass",
            "instrument": profile["bass"],
            "rules": ["follow_chord_roots", "sidechain_with_kick"],
        },
        {
            "name": "keys",
            "instrument": profile["keys"],
            "rules": ["voice_lead_chords", "support_vocal_register"],
        },
        {
            "name": "guitar",
            "instrument": profile["guitar"],
            "rules": ["sparse_verse", "lift_in_chorus"],
        },
        {
            "name": "strings",
            "instrument": profile["strings"],
            "rules": ["chorus_lift", "outro_tail"],
        },
        {
            "name": "brass",
            "instrument": profile["brass"],
            "rules": ["optional_layer", "chorus_stabs_only"],
            "optional": True,
        },
    ]

    vocal_chain = {
        "noise_reduction": True,
        "de_ess": True,
        "pitch_correction": "melody_curve_locked",
        "timing_correction": "beat_grid",
        "compression": "vocal_bus",
        "eq": "presence_air",
        "saturation": "subtle_tube",
        "reverb": "plate_room_hybrid",
        "delay_throws": True,
        "harmonies": "key_chord_guided",
        "doubles_adlibs": True,
    }

    mix_chain = {
        "per_stem_eq": True,
        "per_stem_compression": True,
        "kick_bass_sidechain": True,
        "transient_shaping": True,
        "stereo_imaging": True,
        "reverb_sends": True,
        "delay_sends": True,
        "bus_compression": True,
        "vocal_ducking": True,
    }

    mastering_chain = {
        "target_lufs": -12.0,
        "true_peak_limit_db": -1.0,
        "multiband_compression": True,
        "dynamic_eq": True,
        "stereo_balance": True,
        "reference_match": True,
        "export": ["wav24", "mp3_320", "stems_zip", "release_pack"],
    }

    return {
        "genre": genre,
        "bpm": bpm,
        "sections": sections,
        "chord_progression": chords,
        "downbeats": downbeats,
        "vocal_phrases": analysis.get("vocal_phrases", []),
        "beat_times": analysis.get("beat_times", []),
        "duration": analysis.get("duration", 0.0),
        "arrangement_rules": {
            "by_section": by_section,
            "transitions": transitions,
            "verse_chorus_bridge": True,
        },
        "stems": stems,
        "vocal_chain": vocal_chain,
        "mix_chain": mix_chain,
        "mastering_chain": mastering_chain,
        "render_targets": [
            "drums.wav",
            "bass.wav",
            "keys.wav",
            "guitar.wav",
            "strings.wav",
            "brass.wav",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--genre", default="Pop")
    args = parser.parse_args()

    with open(args.analysis_json, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    plan = build_arrangement(analysis, genre=args.genre)
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
