# Phase 2 Worker Service (FastAPI)

This service is the in-repo scaffold for the `render_worker` pipeline.

It receives:
- `analysis`
- `arrangement_plan`
- `callback_url`
- `callback_secret`

Then it:
1. renders stems (`drums.wav`, `bass.wav`, `keys.wav`, `guitar.wav`, `strings.wav`, `mix.wav`)
2. uploads them (or returns local `file://` URLs if upload is not configured)
3. POSTs completion payload to Supabase `worker-callback`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r scripts/worker_service/requirements.txt
uvicorn scripts.worker_service.render_worker:app --host 0.0.0.0 --port 8081 --reload
```

Health:

```bash
curl http://localhost:8081/health
```

## Deploy persistently on Render

This repository now includes:

- `scripts/worker_service/Dockerfile`
- `render.yaml` (service blueprint)

Typical flow:

1. Push this repo to GitHub/GitLab/Bitbucket.
2. Connect that repo in Render.
3. Create service from `render.yaml` or use CLI `services create`.
4. Set resulting URL as Supabase Vault `RENDER_WORKER_URL` + `/render`.
