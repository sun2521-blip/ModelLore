
# Stage 1: build the console. Only its static files reach the final image.
FROM node:24-slim AS console

WORKDIR /console

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# Stage 2: the API, which also serves the built console at /.
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
COPY --from=console /console/dist ./frontend/dist

RUN mkdir -p data/uploads artifacts lineage mlruns \
    && chown -R modellore:modellore data/uploads artifacts lineage mlruns

USER modellore

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
