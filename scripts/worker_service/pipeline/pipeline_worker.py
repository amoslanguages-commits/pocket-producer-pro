#!/usr/bin/env python3
"""
HTTP service for server-side musical analysis + arrangement planning.

Endpoints:
  GET  /health
  POST /analyze   { "audio_url": "..." }  -> full analysis JSON
  POST /arrange   { "analysis": {...}, "genre": "Pop" } -> arrangement plan
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from analyze_audio import analyze
from render_arrangement import build_arrangement

app = FastAPI(title="Sing2Song Pipeline Worker", version="1.0.0")


class AnalyzeRequest(BaseModel):
    audio_url: str
    export_midi: bool = True


class ArrangeRequest(BaseModel):
    analysis: Dict[str, Any] = Field(default_factory=dict)
    genre: str = "Pop"


def _auth_token() -> str:
    return os.getenv("PIPELINE_AUTH_TOKEN", "").strip()


def _check_auth(request) -> None:
    token = _auth_token()
    if not token:
        return
    header = request.headers.get("x-pipeline-auth", "")
    if header != token:
        raise HTTPException(status_code=401, detail="unauthorized")


def _download_audio(url: str, dest: Path) -> None:
    res = requests.get(url, timeout=180)
    if not res.ok:
        raise HTTPException(status_code=502, detail=f"audio download failed ({res.status_code})")
    dest.write_bytes(res.content)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "pipeline_worker"}


@app.post("/analyze")
def analyze_endpoint(body: AnalyzeRequest, request: Request) -> Dict[str, Any]:
    _check_auth(request)
    if not body.audio_url.strip():
        raise HTTPException(status_code=400, detail="audio_url required")
    with tempfile.TemporaryDirectory(prefix="s2s_analyze_") as tmp:
        wav = Path(tmp) / "input.wav"
        midi_out = Path(tmp) / "melody.mid" if body.export_midi else None
        _download_audio(body.audio_url.strip(), wav)
        result = analyze(str(wav), midi_output=str(midi_out) if midi_out else None)
        # Expose f0 for client pitch correction (stripped from legacy payloads).
        if "f0_series" not in result and isinstance(result.get("melody_contour"), list):
            result.setdefault("f0_series", [])
        return result


@app.post("/arrange")
def arrange_endpoint(body: ArrangeRequest, request: Request) -> Dict[str, Any]:
    _check_auth(request)
    if not body.analysis:
        raise HTTPException(status_code=400, detail="analysis required")
    return build_arrangement(body.analysis, genre=body.genre or "Pop")
