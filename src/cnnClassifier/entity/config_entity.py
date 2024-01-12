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
