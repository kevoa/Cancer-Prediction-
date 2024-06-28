import os 
from CancerPrediction import logger
from dataclasses import dataclass
import pandas as pd
from CancerPrediction.utils.common import read_yaml, create_directories
from CancerPrediction.constants import *
from pathlib import Path
from CancerPrediction.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, df: pd.DataFrame, config: DataValidationConfig):
        self.df = df
        self.config = config
    
    def clean_text(self):
        sequences_to_remove = self.config.sequences_to_remove
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                for sequence in sequences_to_remove:
                    self.df[column] = self.df[column].apply(lambda x: x.replace(sequence, '') if isinstance(x, str) and sequence in x else x)
        return self.df

    def convert_columns(self):
        for column, dtype in self.config.all_schema.items():
            if dtype == 'float64' and self.df[column].dtype == 'object':
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
        return self.df

    def remove_normal_class(self):
        self.df = self.df[self.df[self.config.target_column] != 'Normal']
        return self.df

    def drop_columns(self):
        self.df = self.df.drop(columns=self.config.columns_to_remove)
        return self.df

    def detect_missing_values(self):
        missing_values = self.df.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)
        return self.df
    
    def validate_all_columns(self) -> bool:
        try:
            validation_status = None
            all_cols = list(self.df.columns)
            all_schema = self.config.all_schema.keys()
            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}. Column '{col}' not in schema.\n")
                    return validation_status
                
                # Verificación del tipo de datos
                expected_dtype = self.config.all_schema[col]
                if not pd.api.types.is_dtype_equal(self.df[col].dtype, expected_dtype):
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}. Column '{col}' expected type '{expected_dtype}' but got '{self.df[col].dtype}'.\n")
                    return validation_status

            validation_status = True
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e

    def validate(self):
        self.drop_columns()  # Eliminar columnas no deseadas antes de la validación
        self.remove_normal_class()  # Eliminar filas con 'Normal' antes de la validación
        self.clean_text()
        self.convert_columns()
        self.detect_missing_values()
        columns_valid = self.validate_all_columns()
        if not columns_valid:
            raise ValueError("Column validation failed.")
        return self.df

    def save_validated_data(self):
        self.df.to_excel(self.config.validated_data_file, index=False)
        print(f"Validated data saved to {self.config.validated_data_file}")