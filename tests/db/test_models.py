
import uuid

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from db.base import Base
from db.models import Dataset, ModelRecord, TrainingJob


def test_dataset_training_job_and_model_relationships():
    engine = create_engine("sqlite://")

    Base.metadata.create_all(engine)

    try:
        with Session(engine) as session:
            dataset = Dataset(
                name="sample",
                original_filename="sample.csv",
                file_path="data/sample.csv",
                sha256="a" * 64,
                row_count=10,
                column_count=3,
                target_column="target",
            )
            session.add(dataset)
            session.flush()

            job = TrainingJob(
                dataset_id=dataset.id,
                config_name="csv-test",
                status="FINISHED",
                mlflow_run_id="test-run-001",
            )
            session.add(job)
            session.flush()

            model = ModelRecord(
                training_job_id=job.id,
                name="test-model",
                artifact_path="artifacts/test-model.joblib",
                version="1",
                metrics={"accuracy": 0.95},
            )
            session.add(model)
            session.commit()

            dataset_id = dataset.id
            job_id = job.id
            model_id = model.id

        with Session(engine) as session:
            saved_dataset = session.get(Dataset, dataset_id)
            saved_job = session.get(TrainingJob, job_id)
            saved_model = session.get(ModelRecord, model_id)

            assert saved_dataset is not None
            assert saved_dataset.name == "sample"
            assert len(saved_dataset.training_jobs) == 1

            assert saved_job is not None
            assert saved_job.dataset.name == "sample"
            assert saved_job.model is not None
            assert saved_job.model.name == "test-model"

            assert saved_model is not None
            assert saved_model.metrics == {"accuracy": 0.95}

            assert session.scalar(
                select(Dataset.id).where(Dataset.id == uuid.uuid4())
            ) is None
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
