FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default soundfont path (fluid-soundfont-gm installs to this location)
ENV WORKER_SF2_PATH=/usr/share/sounds/sf2/FluidR3_GM.sf2

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       ffmpeg \
       libsndfile1 \
       fluidsynth \
       fluid-soundfont-gm \
       g++ \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 8080

CMD ["uvicorn", "render_worker:app", "--host", "0.0.0.0", "--port", "8080"]
