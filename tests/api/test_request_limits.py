from uuid import uuid4

import pytest

from api import limits


def predict_body(features, **extra):
    return {"lineage_run_id": str(uuid4()), "features": features, **extra}


def test_predict_rejects_too_many_rows(api_client):
    rows = [{"feature": 1.0}] * (limits.MAX_PREDICT_ROWS + 1)

    response = api_client.post("/predict", json=predict_body(rows))

    assert response.status_code == 422


def test_predict_rejects_too_many_columns_in_a_row(api_client):
    row = {f"feature_{i}": 1 for i in range(limits.MAX_UPLOAD_COLUMNS + 1)}

    response = api_client.post("/predict", json=predict_body([row]))

    assert response.status_code == 422


def test_predict_rejects_overlong_feature_name(api_client):
    row = {"f" * (limits.MAX_NAME_LENGTH + 1): 1.0}

    response = api_client.post("/predict", json=predict_body([row]))

    assert response.status_code == 422


def test_predict_rejects_overlong_feature_value(api_client):
    row = {"feature": "x" * (limits.MAX_FEATURE_VALUE_LENGTH + 1)}

    response = api_client.post("/predict", json=predict_body([row]))

    assert response.status_code == 422


def test_predict_rejects_empty_row(api_client):
    response = api_client.post("/predict", json=predict_body([{}]))

    assert response.status_code == 422


def test_predict_rejects_unknown_fields(api_client):
    response = api_client.post(
        "/predict",
        json=predict_body([{"feature": 1.0}], model_path="../../evil.joblib"),
    )

    assert response.status_code == 422


def test_train_rejects_unknown_fields(api_client):
    response = api_client.post(
        "/train",
        json={"config_name": "iris", "config_path": "/etc/passwd"},
    )

    assert response.status_code == 422


def test_train_rejects_overlong_target_column(api_client):
    response = api_client.post(
        "/train",
        json={
            "dataset_id": str(uuid4()),
            "target_column": "t" * (limits.MAX_NAME_LENGTH + 1),
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "run_id",
    ["not-a-run-id", "..%2F..%2Fetc", "0123456789ABCDEF0123456789ABCDEF", "a" * 33],
)
def test_run_lookup_rejects_malformed_run_ids(api_client, run_id):
    response = api_client.get(f"/runs/{run_id}")

    assert response.status_code in (404, 422)
    assert response.status_code != 200


def test_run_lookup_checks_auth_before_validating_the_id(api_client):
    api_client.headers.pop("X-API-Key")

    response = api_client.get("/runs/not-a-run-id")

    assert response.status_code == 401


def test_int_from_env_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("MODELLORE_TEST_LIMIT", raising=False)

    assert limits.int_from_env("MODELLORE_TEST_LIMIT", 7) == 7


def test_int_from_env_reads_override(monkeypatch):
    monkeypatch.setenv("MODELLORE_TEST_LIMIT", "42")

    assert limits.int_from_env("MODELLORE_TEST_LIMIT", 7) == 42


@pytest.mark.parametrize("bad_value", ["zero", "0", "-5", "1.5"])
def test_int_from_env_rejects_bad_values(monkeypatch, bad_value):
    monkeypatch.setenv("MODELLORE_TEST_LIMIT", bad_value)

    with pytest.raises(ValueError):
        limits.int_from_env("MODELLORE_TEST_LIMIT", 7)


def test_limits_are_public_and_match_the_server(api_client):
    from api import main

    api_client.headers.pop("X-API-Key")

    response = api_client.get("/limits")

    assert response.status_code == 200
    assert response.json() == {
        "max_upload_mb": limits.MAX_UPLOAD_MB,
        "max_upload_rows": main.MAX_UPLOAD_ROWS,
        "max_upload_columns": main.MAX_UPLOAD_COLUMNS,
        "max_predict_rows": limits.MAX_PREDICT_ROWS,
        "rate_limits_per_minute": limits.RATE_LIMITS,
    }


def test_upload_size_follows_the_configured_megabytes():
    from api import main

    assert main.MAX_UPLOAD_BYTES == limits.MAX_UPLOAD_MB * 1024 * 1024
    assert f"{limits.MAX_UPLOAD_MB} MiB" in main.UPLOAD_TOO_LARGE
