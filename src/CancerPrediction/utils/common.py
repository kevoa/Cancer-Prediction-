import os
from box.exceptions import BoxValueError
import yaml
from mlProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> configBox:
     """
    Reads a YAML file and returns a ConfigBox type.

    Args:
        path_to_yaml (str): Path-like input.

    Raises:
        ValueError: If the YAML file is empty.

    Returns:
        ConfigBox: ConfigBox type.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e
    
    
@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create a list of directories.

    Args:
        path_to_directories (list): List of paths of directories.
        verbose (bool, optional): If True, logs each created directory. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save JSON data.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved in the JSON file.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON file data.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as class attributes instead of dict.
    """
    with open(path) as f:
        content = json.load(f)
    
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save binary file.

    Args:
        data (Any): Data to be saved as binary.
        path (Path): Path to binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load binary data.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Object stored in the file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: Size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"{size_in_kb} KB"
