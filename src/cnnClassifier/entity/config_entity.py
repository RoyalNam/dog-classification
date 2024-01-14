from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataLoaderConfig:
    root_dir: Path
    train_dir: Path
    test_dir: Path
    image_transforms_path: Path
    weights: str
    params_batch_size: int


@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_model_path: Path
    params_weights: str
    params_model: str
    params_seed: int
    
@dataclass(frozen=True)
class TrainerConfig:
    root_dir: Path
    trained_model_path: Path
    updated_model_path: Path
    params_epochs: int
    params_lr: float
    mlflow_uri: str
    all_params: dict