FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY scripts/worker_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY scripts/worker_service /app

EXPOSE 10000

CMD ["uvicorn", "render_worker:app", "--host", "0.0.0.0", "--port", "10000"]
