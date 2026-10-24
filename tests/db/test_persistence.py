
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from db.base import Base
from db.models import Dataset, ModelRecord, TrainingJob


def test_training_job_and_model_persist_across_sessions():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as session:
            job = TrainingJob(
                config_name="iris",
                status="FINISHED",
                mlflow_run_id="mlflow-test-123",
            )
            session.add(job)
            session.flush()

            model = ModelRecord(
                training_job_id=job.id,
                name="iris-model",
                artifact_path="artifacts/iris.joblib",
                metrics={"accuracy": 0.95},
            )
            session.add(model)
            session.commit()

            job_id = job.id
            model_id = model.id

        with Session(engine) as session:
            saved_job = session.get(TrainingJob, job_id)
            saved_model = session.get(ModelRecord, model_id)

            assert saved_job is not None
            assert saved_job.status == "FINISHED"
            assert saved_job.mlflow_run_id == "mlflow-test-123"

            assert saved_model is not None
            assert saved_model.training_job_id == job_id
            assert saved_model.metrics == {"accuracy": 0.95}

    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_model_record_requires_existing_training_job():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as session:
            job = TrainingJob(
                config_name="iris",
                status="FINISHED",
            )
            session.add(job)
            session.commit()

            model = ModelRecord(
                training_job_id=job.id,
                name="iris-model",
                artifact_path="artifacts/iris.joblib",
            )
            session.add(model)
            session.commit()

            result = session.scalar(
                select(ModelRecord).where(
                    ModelRecord.training_job_id == job.id
                )
            )
            assert result is not None
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()
