"""Negative tests for dataset uploads: names, sizes, shapes, and paths."""

from uuid import UUID

import pytest
from sqlalchemy import func, select

from api import main
from db.models import Dataset


VALID_CSV = b"feature_a,feature_b,target\n1,2,0\n3,4,1\n"


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    directory = tmp_path / "uploads"
    monkeypatch.setattr(main, "UPLOAD_DIR", directory)
    return directory


def upload(api_client, filename, content=VALID_CSV):
    return api_client.post(
        "/datasets/upload",
        files={"file": (filename, content, "text/csv")},
    )


def dataset_count(db_session_factory):
    with db_session_factory() as session:
        return session.scalar(select(func.count()).select_from(Dataset))


@pytest.mark.parametrize(
    "hostile_name",
    [
        "../../evil.csv",
        "..\\..\\evil.csv",
        "/etc/evil.csv",
        "C:\\Windows\\System32\\evil.csv",
        "uploads/../../evil.csv",
    ],
)
def test_upload_path_traversal_names_stay_inside_upload_dir(
    api_client, upload_dir, tmp_path, db_session_factory, hostile_name
):
    response = upload(api_client, hostile_name)

    assert response.status_code == 200
    dataset_id = UUID(response.json()["dataset_id"])

    # The only file written anywhere under tmp_path is the server-named copy.
    written = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert written == [upload_dir / f"{dataset_id}.csv"]

    with db_session_factory() as session:
        saved = session.get(Dataset, dataset_id)
        assert saved.original_filename == "evil.csv"
        assert saved.name == "evil"
        assert saved.file_path == str(upload_dir / f"{dataset_id}.csv")


def test_upload_strips_invisible_control_characters(
    api_client, upload_dir, db_session_factory
):
    response = upload(api_client, "\u202eevil\u200b.csv")

    assert response.status_code == 200
    with db_session_factory() as session:
        saved = session.get(Dataset, UUID(response.json()["dataset_id"]))
        assert saved.original_filename == "evil.csv"


def test_upload_caps_long_filenames_and_keeps_the_extension(
    api_client, upload_dir, db_session_factory
):
    response = upload(api_client, "a" * 400 + ".csv")

    assert response.status_code == 200
    with db_session_factory() as session:
        saved = session.get(Dataset, UUID(response.json()["dataset_id"]))
        assert len(saved.original_filename) == 255
        assert saved.original_filename.endswith(".csv")


@pytest.mark.parametrize("name", ["evil.csv.exe", "evil", ".csv", "evil.txt"])
def test_upload_rejects_names_that_are_not_csv(api_client, upload_dir, name):
    assert upload(api_client, name).status_code == 415


def test_upload_accepts_uppercase_extension(api_client, upload_dir):
    assert upload(api_client, "DATA.CSV").status_code == 200


def test_upload_rejects_declared_size_before_reading_the_body(
    api_client, upload_dir, monkeypatch, db_session_factory
):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    big = b"feature,target\n" + b"1,0\n" * 40_000  # ~160 KiB, beyond the envelope

    response = upload(api_client, "big.csv", big)

    assert response.status_code == 413
    assert not upload_dir.exists()
    assert dataset_count(db_session_factory) == 0


def test_upload_rejects_invalid_content_length_header(api_client, upload_dir):
    response = api_client.post(
        "/datasets/upload",
        content=b"x",
        headers={"Content-Length": "not-a-number", "Content-Type": "text/csv"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid Content-Length header"}


def test_upload_rejects_too_many_rows(api_client, upload_dir, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_ROWS", 2)

    response = upload(api_client, "rows.csv", b"a,target\n1,0\n2,1\n3,0\n")

    assert response.status_code == 413


def test_upload_rejects_too_many_columns(api_client, upload_dir, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_COLUMNS", 3)

    response = upload(api_client, "cols.csv", b"a,b,c,target\n1,2,3,0\n")

    assert response.status_code == 413


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"feature,target\n1,\x000\n", "NUL byte"),
        (b"feature,target\n1,0\n2,1,9\n", "ragged row"),
        (b"feature,target\n", "header only"),
        (b"target\n0\n1\n", "single column"),
        (b"f" * 300 + b",target\n1,0\n", "overlong header"),
        (b'feature,target\n"unterminated,0\n', "broken quoting"),
    ],
)
def test_upload_rejects_malformed_content(
    api_client, upload_dir, db_session_factory, content, reason
):
    response = upload(api_client, "bad.csv", content)

    assert response.status_code == 400, reason
    assert dataset_count(db_session_factory) == 0
    assert not any(upload_dir.glob("*")) if upload_dir.exists() else True


def test_train_refuses_a_csv_file_with_no_dataset_record(
    api_client, upload_dir
):
    upload_dir.mkdir()
    dataset_id = "12345678-1234-5678-1234-567812345678"
    (upload_dir / f"{dataset_id}.csv").write_bytes(VALID_CSV)

    response = api_client.post(
        "/train",
        json={"dataset_id": dataset_id, "target_column": "target"},
    )

    assert response.status_code == 404
