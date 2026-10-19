
import argparse
import json
import math
import tempfile
from pathlib import Path

import yaml

from ml.reproducibility import hash_config
from ml.training.train import train_model


def verify_lineage(
    record_path: str | Path,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-8,
) -> dict:
    record_path = Path(record_path)

    with record_path.open("r", encoding="utf-8") as file:
        original = json.load(file)

    if original.get("status") != "FINISHED":
        raise ValueError("Cannot verify a lineage record that did not finish.")

    config = original.get("config_snapshot")
    if not isinstance(config, dict):
        raise ValueError(
            "This record has no saved configuration. "
            "Train again with the updated lineage code."
        )

    if hash_config(config) != original["config_hash"]:
        raise ValueError("Saved configuration hash does not match.")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "verify.yaml"
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        rerun = train_model(config_path=str(config_path))

    dataset_matches = (
        rerun["dataset_hash"] == original["dataset_hash"]
    )
    config_matches = (
        rerun["config_hash"] == original["config_hash"]
    )

    metric_comparison = {}
    for name, expected in original["metrics"].items():
        actual = rerun["metrics"].get(name)

        metric_comparison[name] = (
            actual is not None
            and math.isclose(
                float(expected),
                float(actual),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        )

    return {
        "original_run_id": original["run_id"],
        "verification_run_id": rerun["run_id"],
        "dataset_matches": dataset_matches,
        "config_matches": config_matches,
        "metrics_match": (
            bool(metric_comparison)
            and all(metric_comparison.values())
        ),
        "metric_comparison": metric_comparison,
        "verified": (
            dataset_matches
            and config_matches
            and bool(metric_comparison)
            and all(metric_comparison.values())
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify a saved ModelLore training run."
    )
    parser.add_argument("lineage_path", help="Path to a lineage JSON file")
    args = parser.parse_args()

    result = verify_lineage(args.lineage_path)
    print(json.dumps(result, indent=2))

    if not result["verified"]:
        raise SystemExit(1)
