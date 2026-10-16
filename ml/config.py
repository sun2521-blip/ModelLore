from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_type: str
    task_type: str
    parameters: dict


@dataclass
class TrainingConfig:
    random_seed: int = 42
    test_size: float = 0.2
