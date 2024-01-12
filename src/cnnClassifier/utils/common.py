import os
from pathlib import Path
import yaml
import joblib
import json
import base64
from typing import Any
from cnnClassifier import logger
import torch


def read_yaml(path: Path):
    try:
        with open(path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file: {path} loaded successfully")
            return content
    except Exception as e:
        logger.error(f"Error loading YAML file from {path}: {e}")
        raise


def create_dir(list_path: list, verbose=True):
    for path in list_path:
        try:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Created directory at: {path}")
        except Exception as e:
            logger.error(f"Error creating directory at {path}: {e}")
            raise


def save_model(model: torch.nn.Module, path: Path):
    torch.save(model, path)


def save_json(path: Path, data: dict):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
            logger.info(f"Json file saved at: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON file at {path}: {e}")
        raise


def load_json(path: Path):
    try:
        with open(path) as f:
            content = json.load(f)
            logger.info(f"Json file loaded successfully from: {path}")
            return content
    except Exception as e:
        logger.error(f"Error loading JSON file from {path}: {e}")
        raise


def save_bin(path: Path, data: Any):
    try:
        with open(path, "wb") as f:
            joblib.dump(data, f)
            logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        logger.error(f"Error saving BINARY file at {path}: {e}")
        raise


def load_bin(path: Path):
    try:
        with open(path, "r") as f:
            content = joblib.load(f)
            logger.info(f"Binary file loaded successfully from: {path}")
            return content
    except Exception as e:
        logger.error(f"Error loading BINARY file from {path}: {e}")
        raise


def get_size(path: Path):
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(image_data, filename):
    image_bytes = base64.b64decode(image_data)
    with open(filename, "wb") as img_file:
        img_file.write(image_bytes)


def encodeImage(cropped_image_path):
    with open(cropped_image_path, "wb") as f:
        return base64.b64encode(f.read())
    