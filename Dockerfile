
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user. The application code stays owned by root, so
# the app can read it but not change it; only the data folders belong to it.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin modellore

COPY . .

RUN mkdir -p data/uploads artifacts lineage mlruns \
    && chown -R modellore:modellore data/uploads artifacts lineage mlruns

USER modellore

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
