#!/usr/bin/env python3
"""
Phase 2 render worker scaffold (FastAPI).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
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


def _worker_config() -> WorkerConfig:
    return WorkerConfig(
        upload_base_url=os.getenv("WORKER_UPLOAD_BASE_URL", "").strip(),
        upload_token=os.getenv("WORKER_UPLOAD_TOKEN", "").strip(),
        worker_auth_token=os.getenv("WORKER_AUTH_TOKEN", "").strip(),
    )


def _sec_to_samples(sec: float) -> int:
    return max(1, int(sec * SAMPLE_RATE))


def _tone(freq: float, duration_sec: float, gain: float = 0.3) -> np.ndarray:
    n = _sec_to_samples(duration_sec)
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float64)
    return (np.sin(2.0 * np.pi * freq * t) * gain).astype(np.float32)


def _noise(duration_sec: float, gain: float = 0.1, seed: int = 0) -> np.ndarray:
    n = _sec_to_samples(duration_sec)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * gain).astype(np.float32)


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

    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)

    kick = _tone(55.0, 0.09, 0.8)
    snare_noise = _noise(0.08, 0.35, seed=seed + 7)
    snare = snare_noise * np.linspace(1.0, 0.2, snare_noise.size, dtype=np.float32)
    hat = _noise(0.03, 0.12, seed=seed + 17)

    for bar in range(bars):
        bar_start = bar * 4 * beat_sec
        for beat in range(4):
            t0 = bar_start + beat * beat_sec
            i0 = _sec_to_samples(t0)
            if beat in (0, 2):
                avail = max(0, len(drums) - i0)
                if avail > 0:
                    drums[i0 : i0 + min(len(kick), avail)] += kick[: min(len(kick), avail)]
            if beat in (1, 3):
                avail = max(0, len(drums) - i0)
                if avail > 0:
                    drums[i0 : i0 + min(len(snare), avail)] += snare[: min(len(snare), avail)]
            ih = _sec_to_samples(t0 + beat_sec * 0.5)
            avail = max(0, len(drums) - ih)
            if avail > 0:
                drums[ih : ih + min(len(hat), avail)] += hat[: min(len(hat), avail)]

    key = str(plan.get("key", "C")).upper()
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

    b_note = _tone(root / 2.0, beat_sec * 0.85, 0.35)
    for bar in range(bars):
        for beat in range(4):
            i0 = _sec_to_samples((bar * 4 + beat) * beat_sec)
            bass[i0 : i0 + len(b_note)] += b_note[: max(0, len(bass) - i0)]

    chord = _mix(
        [
            _tone(root, 4 * beat_sec, 0.17),
            _tone(root * 1.5, 4 * beat_sec, 0.13),
            _tone(root * 2.0, 4 * beat_sec, 0.11),
        ]
    )
    for bar in range(bars):
        i0 = _sec_to_samples(bar * 4 * beat_sec)
        keys[i0 : i0 + len(chord)] += chord[: max(0, len(keys) - i0)]

    g_pad = _tone(root * 2.5, beat_sec * 2, 0.07)
    s_pad = _tone(root * 1.25, beat_sec * 4, 0.08)
    for bar in range(bars):
        i0 = _sec_to_samples(bar * 4 * beat_sec)
        guitar[i0 : i0 + len(g_pad)] += g_pad[: max(0, len(guitar) - i0)]
        strings[i0 : i0 + len(s_pad)] += s_pad[: max(0, len(strings) - i0)]

    stems = {
        "drums": drums * 0.9,
        "bass": bass * 0.9,
        "keys": keys * 0.9,
        "guitar": guitar * 0.9,
        "strings": strings * 0.9,
    }
    for k in list(stems.keys()):
        peak = float(np.max(np.abs(stems[k]))) if stems[k].size else 0.0
        if peak > 0.99:
            stems[k] = stems[k] * (0.99 / peak)

    stems["mix"] = _mix(list(stems.values())) * MASTER_GAIN

    out: Dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in stems.items():
        p = out_dir / f"{name}.wav"
        sf.write(str(p), audio, SAMPLE_RATE, subtype="PCM_16")
        out[name] = p
    return out


def _upload_file(path: Path, remote_path: str, cfg: WorkerConfig) -> str:
    # Upload is required for mobile-consumable durable URLs.
    if not cfg.upload_base_url:
        raise RuntimeError("WORKER_UPLOAD_BASE_URL is required for durable stem URLs")

    url = f"{cfg.upload_base_url.rstrip('/')}/{remote_path.lstrip('/')}"
    headers = {}
    if cfg.upload_token:
        headers["Authorization"] = f"Bearer {cfg.upload_token}"
    with path.open("rb") as f:
        r = requests.put(url, data=f, headers=headers, timeout=60)
    r.raise_for_status()
    return url


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
        "production_contracts": contracts.model_dump(),
        "versions": [v.model_dump() for v in versions],
        "stems": [s.model_dump() for s in stems],
    }
    r = requests.post(req.callback_url, json=payload, headers=headers, timeout=30)
    r.raise_for_status()


app = FastAPI(title="Pocket Producer Pro Render Worker")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/render")
def render(req: RenderRequest, request: Request) -> Dict[str, Any]:
    cfg = _worker_config()
    if cfg.worker_auth_token:
        incoming = request.headers.get("x-worker-auth", "").strip()
        if incoming != cfg.worker_auth_token:
            raise HTTPException(status_code=401, detail="Unauthorized worker caller")

    if req.mode != "render_worker":
        raise HTTPException(status_code=400, detail="Unsupported mode")

    plan = req.arrangement_plan or {
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

    try:
        _post_progress(req, 20, "Arranging stems")
        with tempfile.TemporaryDirectory(prefix="ppp-render-") as td:
            out_dir = Path(td) / "stems"
            stems = _render_stems(plan, out_dir, seed_key=f"{req.project_id}:{req.job_id}")
            _post_progress(req, 72, "Uploading stems")

            stem_urls: Dict[str, str] = {}
            for name, p in stems.items():
                remote = f"{req.project_id}/{req.job_id}/{name}.wav"
                stem_urls[name] = _upload_file(p, remote, cfg)

            _post_progress(req, 92, "Finalizing song version")
            _post_complete(req, plan, stem_urls, contracts)

        return {"ok": True, "job_id": req.job_id, "project_id": req.project_id, "stems": stem_urls}
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

