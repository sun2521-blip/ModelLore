from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str
    task_type: str
    parameters: dict