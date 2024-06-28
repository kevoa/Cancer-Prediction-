import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from CancerPrediction.entity.config_entity import DataTransformationConfig
from pathlib import Path
import joblib

class DataTransformation:
    def __init__(self, df: pd.DataFrame, config: DataTransformationConfig):
        self.df = df
        self.config = config
        self.label_encoders = {}
    
    def encode_labels(self):
        le = LabelEncoder()
        self.df[self.config.target_column] = le.fit_transform(self.df[self.config.target_column])
        self.label_encoders[self.config.target_column] = le
    
    def get_preprocessor(self):
        numeric_features = self.config.important_features

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor

    def transform(self):
        self.config.transformed_train_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.encode_labels()
        
        train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42)
        
        y_train = train_df.pop(self.config.target_column)
        y_test = test_df.pop(self.config.target_column)
        
        preprocessor = self.get_preprocessor()
        
        train_df_transformed = preprocessor.fit_transform(train_df[self.config.important_features])
        test_df_transformed = preprocessor.transform(test_df[self.config.important_features])
        
        feature_names = self.config.important_features
        
        train_df_transformed = pd.DataFrame(train_df_transformed, columns=feature_names)
        test_df_transformed = pd.DataFrame(test_df_transformed, columns=feature_names)
        
        train_df_transformed[self.config.target_column] = y_train.values
        test_df_transformed[self.config.target_column] = y_test.values
        
        train_df_transformed.to_excel(self.config.transformed_train_data_path, index=False)
        test_df_transformed.to_excel(self.config.transformed_test_data_path, index=False)
        
        for feature, encoder in self.label_encoders.items():
            joblib.dump(encoder, Path(self.config.root_dir) / f'{feature}_label_encoder.joblib')

        joblib.dump(preprocessor, Path(self.config.root_dir) / 'preprocessor.joblib')
        
        print("Data transformation completed successfully")
        print(f"Training data saved to {self.config.transformed_train_data_path}")
        print(f"Test data saved to {self.config.transformed_test_data_path}")
