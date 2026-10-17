
import json

import pytest

import ml.verification as verification


@pytest.fixture
def lineage_record(tmp_path):
    config = {
        "dataset": {"name": "iris"},
        "training": {"random_seed": 42, "test_size": 0.2},
        "model": {
            "model_type": "random_forest",
            "task_type": "classification",
            "parameters": {"n_estimators": 5, "max_depth": 3},
        },
    }

    record = {
        "run_id": "original-run",
        "status": "FINISHED",
        "dataset_hash": "dataset-hash-123",
        "config_hash": verification.hash_config(config),
        "config_snapshot": config,
        "metrics": {"accuracy": 0.9, "f1": 0.88},
    }

    record_path = tmp_path / "original-run.json"
    record_path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    return record_path, record


def fake_training_result(original):
    return {
        "run_id": "verification-run",
        "dataset_hash": original["dataset_hash"],
        "config_hash": original["config_hash"],
        "metrics": dict(original["metrics"]),
    }


def test_fresh_lineage_record_verifies(
    lineage_record, monkeypatch
):
    record_path, original = lineage_record

    monkeypatch.setattr(
        verification,
        "train_model",
        lambda config_path: fake_training_result(original),
    )

    result = verification.verify_lineage(record_path)

    assert result["dataset_matches"] is True
    assert result["config_matches"] is True
    assert result["metrics_match"] is True
    assert result["verified"] is True


@pytest.mark.parametrize(
    "field",
    ["dataset_hash", "accuracy"],
)
def test_tampered_lineage_record_does_not_verify(
    lineage_record, monkeypatch, field
):
    record_path, original = lineage_record

    if field == "dataset_hash":
        original["dataset_hash"] = "tampered-dataset-hash"
    else:
        original["metrics"]["accuracy"] = 0.1

    record_path.write_text(
        json.dumps(original),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        verification,
        "train_model",
        lambda config_path: fake_training_result(
            {
                **original,
                "dataset_hash": "dataset-hash-123",
                "config_hash": original["config_hash"],
                "metrics": {"accuracy": 0.9, "f1": 0.88},
            }
        ),
    )

    result = verification.verify_lineage(record_path)

    assert result["verified"] is False
