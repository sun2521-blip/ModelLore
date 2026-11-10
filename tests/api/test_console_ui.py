"""The built console is served from the API's origin, and only from its build folder."""

import pytest

from api import console_ui


@pytest.fixture
def built_console(tmp_path, monkeypatch):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>ModelLore</title>", encoding="utf-8")
    (dist / "assets" / "index-abc123.js").write_text("console.log('console');", encoding="utf-8")
    (dist / "assets" / "index-abc123.css").write_text("body{}", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    # Something next to the build folder that must never be reachable through it.
    (tmp_path / "secret.txt").write_text("not for the browser", encoding="utf-8")
    monkeypatch.setattr(console_ui, "UI_DIR", dist)
    return dist


def test_console_page_is_served_with_security_headers(api_client, built_console):
    api_client.headers.pop("X-API-Key")  # the page itself is public; its data needs the key

    response = api_client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-cache"


def test_assets_are_served_and_cached(api_client, built_console):
    script = api_client.get("/assets/index-abc123.js")
    style = api_client.get("/assets/index-abc123.css")

    assert script.status_code == 200
    assert "javascript" in script.headers["content-type"]
    assert script.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert style.status_code == 200
    assert api_client.get("/favicon.svg").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/assets/../../secret.txt",
        "/assets/%2e%2e/%2e%2e/secret.txt",
        "/assets/..%2f..%2fsecret.txt",
        "/assets/..\\..\\secret.txt",
        "/assets/missing.js",
        "/assets/",
    ],
)
def test_nothing_outside_the_build_folder_is_served(api_client, built_console, path):
    response = api_client.get(path)

    assert response.status_code == 404
    assert "not for the browser" not in response.text


def test_ui_file_refuses_absolute_and_escaping_paths(built_console, tmp_path):
    assert console_ui.ui_file("index.html") is not None
    assert console_ui.ui_file("../secret.txt") is None
    assert console_ui.ui_file(str(tmp_path / "secret.txt")) is None


def test_unbuilt_console_explains_itself(api_client, tmp_path, monkeypatch):
    monkeypatch.setattr(console_ui, "UI_DIR", tmp_path / "no-build-here")

    response = api_client.get("/")

    assert response.status_code == 404
    assert "npm run build" in response.json()["detail"]


def test_api_routes_are_unchanged(api_client, built_console):
    health = api_client.get("/health")

    assert health.json() == {"status": "healthy"}
    assert "content-security-policy" not in health.headers
    assert api_client.get("/datasets").status_code == 200


def test_console_routes_stay_out_of_the_api_schema(api_client, built_console):
    paths = api_client.get("/openapi.json").json()["paths"]

    assert "/" not in paths
    assert not any(path.startswith("/assets") for path in paths)
