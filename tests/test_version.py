import tomllib
from pathlib import Path

from api.main import app


def test_api_version_matches_project_version():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert app.version == project["version"]
