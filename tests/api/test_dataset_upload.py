
import yaml
from uuid import UUID
from unittest.mock import patch

import pytest

from api import main


VALID_CSV = b"feature_a,feature_b,target\n1,2,0\n3,4,1\n"


def test_upload_valid_csv_returns_dataset_id(
    api_client, tmp_path, monkeypatch
):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)

    response = api_client.post(
        "/datasets/upload",
        files={"file": ("dataset.csv", VALID_CSV, "text/csv")},
    )

    assert response.status_code == 200
    body = response.json()
    dataset_id = UUID(body["dataset_id"])

    assert body["filename"] == f"{dataset_id}.csv"
    assert body["size_bytes"] == len(VALID_CSV)
    assert (upload_dir / f"{dataset_id}.csv").read_bytes() == VALID_CSV


def test_upload_rejects_non_csv_file(api_client):
    response = api_client.post(
        "/datasets/upload",
        files={"file": ("dataset.txt", b"some text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file(api_client):
    response = api_client.post(
        "/datasets/upload",
        files={"file": ("dataset.csv", b"", "text/csv")},
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "content",
    [
        b"feature,feature,target\n1,2,0\n",
        b"feature,,target\n1,2,0\n",
    ],
)
def test_upload_rejects_invalid_headers(api_client, content):
    response = api_client.post(
        "/datasets/upload",
        files={"file": ("dataset.csv", content, "text/csv")},
    )

    assert response.status_code == 400


def test_upload_rejects_invalid_utf8(api_client):
    response = api_client.post(
        "/datasets/upload",
        files={
            "file": (
                "dataset.csv",
                b"feature,target\n\xff,0\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 400


def test_upload_rejects_oversized_file(
    api_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)

    response = api_client.post(
        "/datasets/upload",
        files={
            "file": (
                "dataset.csv",
                b"feature,target\n1,0\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 413


def test_train_rejects_unknown_dataset_id(
    api_client, tmp_path, monkeypatch
):
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")

    response = api_client.post(
        "/train",
        json={
            "dataset_id": "12345678-1234-5678-1234-567812345678"
        },
    )

    assert response.status_code == 404


def test_train_uses_custom_target_column(
    api_client, tmp_path, monkeypatch
):
    upload_dir = tmp_path / "uploads"
    config_dir = tmp_path / "configs"
    upload_dir.mkdir()
    config_dir.mkdir()

    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(main, "ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(main, "LINEAGE_DIR", tmp_path / "lineage")

    dataset_id = UUID("12345678-1234-5678-1234-567812345678")
    (upload_dir / f"{dataset_id}.csv").write_bytes(
        b"feature,species\n1,cat\n2,dog\n"
    )

    (config_dir / "csv-test.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "name": "csv",
                    "path": "placeholder.csv",
                    "target_column": "target",
                },
                "model": {},
                "training": {},
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "api.main.train_model",
        return_value={
            "run_id": "test-lineage-id",
            "mlflow_run_id": "test-mlflow-id",
            "metrics": {},
            "parameters": {},
            "lineage_path": "lineage/test-lineage-id.json",
        },
    ) as mock_train:
        response = api_client.post(
            "/train",
            json={
                "dataset_id": str(dataset_id),
                "target_column": "species",
            },
        )

    assert response.status_code == 200

    config_path = mock_train.call_args.kwargs["config_path"]
    generated_config = yaml.safe_load(
        open(config_path, encoding="utf-8")
    )

    assert generated_config["dataset"]["target_column"] == "species"
    assert generated_config["dataset"]["path"] == str(
        (upload_dir / f"{dataset_id}.csv").resolve()
    )
