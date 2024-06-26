from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: Dict
    sequences_to_remove: List[str]
    target_column: str
    columns_to_remove: List[str]
    
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    validated_data_file: Path
    transformed_train_data_path: Path
    transformed_test_data_path: Path
    target_column: str
    ordinal_features: List[str]
    nominal_features: List[str]
    