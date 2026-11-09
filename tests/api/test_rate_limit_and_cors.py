import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from api import limits
from api.rate_limit import FixedWindowRateLimiter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_limiter_allows_up_to_the_limit_then_refuses():
    clock = FakeClock()
    limiter = FixedWindowRateLimiter(window_seconds=60, clock=clock)

    assert limiter.hit("client", limit=2) is None
    assert limiter.hit("client", limit=2) is None

    clock.now += 15
    assert limiter.hit("client", limit=2) == pytest.approx(45)


def test_limiter_opens_again_after_the_window():
    clock = FakeClock()
    limiter = FixedWindowRateLimiter(window_seconds=60, clock=clock)
    limiter.hit("client", limit=1)

    clock.now += 60
    assert limiter.hit("client", limit=1) is None


def test_limiter_counts_each_key_separately():
    limiter = FixedWindowRateLimiter(window_seconds=60, clock=FakeClock())
    limiter.hit("predict:1.2.3.4", limit=1)

    assert limiter.hit("predict:5.6.7.8", limit=1) is None
    assert limiter.hit("train:1.2.3.4", limit=1) is None


def test_limiter_forgets_expired_keys_when_full():
    clock = FakeClock()
    limiter = FixedWindowRateLimiter(window_seconds=60, clock=clock, max_tracked_keys=2)
    limiter.hit("a", limit=5)
    limiter.hit("b", limit=5)

    clock.now += 61
    limiter.hit("c", limit=5)

    assert set(limiter._windows) == {"c"}


def predict_once(api_client):
    return api_client.post(
        "/predict",
        json={"lineage_run_id": str(uuid4()), "features": [{"feature": 1.0}]},
    )


def test_predict_returns_429_with_retry_after(api_client, monkeypatch):
    monkeypatch.setitem(limits.RATE_LIMITS, "predict", 2)

    assert predict_once(api_client).status_code == 404
    assert predict_once(api_client).status_code == 404

    limited = predict_once(api_client)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


def test_requests_without_a_key_still_count(api_client, monkeypatch):
    monkeypatch.setitem(limits.RATE_LIMITS, "train", 2)
    api_client.headers.pop("X-API-Key")

    statuses = [
        api_client.post("/train", json={"config_name": "iris"}).status_code
        for _ in range(3)
    ]

    assert statuses == [401, 401, 429]


def test_limits_apply_per_endpoint_group(api_client, monkeypatch):
    monkeypatch.setitem(limits.RATE_LIMITS, "predict", 1)
    predict_once(api_client)
    assert predict_once(api_client).status_code == 429

    # Uploads have their own budget, and read-only endpoints aren't limited.
    assert api_client.post(
        "/datasets/upload",
        files={"file": ("d.csv", b"a,target\n1,0\n", "text/csv")},
    ).status_code != 429
    assert api_client.get("/datasets").status_code == 200


def test_no_cross_origin_access_by_default(api_client):
    preflight = api_client.options(
        "/predict",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    simple = api_client.get("/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in preflight.headers
    assert "access-control-allow-origin" not in simple.headers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("http://localhost:5173", ["http://localhost:5173"]),
        (" https://a.example/ , http://localhost:3000", ["https://a.example", "http://localhost:3000"]),
    ],
)
def test_cors_origins_parse(raw, expected):
    assert limits.cors_origins_from_env(raw) == expected


@pytest.mark.parametrize("raw", ["*", "https://*.example.com", "localhost:5173", "https://ok.example,*"])
def test_cors_origins_refuse_wildcards_and_bare_hosts(raw):
    with pytest.raises(ValueError):
        limits.cors_origins_from_env(raw)


def run_app_probe(cors_origins):
    """Import the app in a fresh process with MODELLORE_CORS_ORIGINS set."""
    probe = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from api.main import app\n"
        "client = TestClient(app)\n"
        "def preflight(origin):\n"
        "    r = client.options('/predict', headers={'Origin': origin,"
        " 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'x-api-key'})\n"
        "    return {k: v for k, v in r.headers.items() if k.startswith('access-control-')}\n"
        "print(json.dumps({'allowed': preflight('http://localhost:5173'),"
        " 'other': preflight('https://evil.example')}))\n"
    )
    env = {
        **os.environ,
        "MODELLORE_CORS_ORIGINS": cors_origins,
        "MLFLOWHUB_API_KEY": "probe-key",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_configured_origin_is_allowed_without_credentials():
    result = run_app_probe("http://localhost:5173")

    assert result.returncode == 0, result.stderr[-500:]
    headers = json.loads(result.stdout.strip().splitlines()[-1])

    assert headers["allowed"]["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-credentials" not in headers["allowed"]
    assert "access-control-allow-origin" not in headers["other"]


def test_wildcard_origin_stops_the_app_from_starting():
    result = run_app_probe("*")

    assert result.returncode != 0
    assert "wildcards are not allowed" in result.stderr
