import urllib.request as request
from CancerPrediction import logger 
from CancerPrediction.utils.common import get_size
from CancerPrediction.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)
from pathlib import Path 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Implementar DataTransformation en una celda de notebook para prueba
class DataTransformation:
    def __init__(self, df: pd.DataFrame, config: DataTransformationConfig):
        self.df = df
        self.config = config
        self.label_encoders = {}
    
    def encode_labels(self):
        # Codificar las características nominales usando LabelEncoder
        for feature in self.config.nominal_features:
            le = LabelEncoder()
            self.df[feature] = le.fit_transform(self.df[feature])
            self.label_encoders[feature] = le
        
        # Codificar la columna objetivo
        le = LabelEncoder()
        self.df[self.config.target_column] = le.fit_transform(self.df[self.config.target_column])
        self.label_encoders[self.config.target_column] = le
    
    def get_preprocessor(self):
        # Identificar características numéricas y categóricas
        numeric_features = self.df.select_dtypes(include=[float, int]).columns.tolist()
        ordinal_features = self.config.ordinal_features
        nominal_features = self.config.nominal_features

        # Eliminar las características ordinales, nominales y la columna objetivo de las características numéricas
        numeric_features = [feature for feature in numeric_features if feature not in ordinal_features + nominal_features + [self.config.target_column]]

        # Preprocesamiento para las características numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Imputación con la mediana
            ('scaler', StandardScaler())  # Estandarización
        ])

        # Preprocesamiento para las características categóricas ordinales
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(dtype=int))  # Codificación Ordinal
        ])

        # Preprocesamiento para las características categóricas nominales (binarias)
        nominal_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='if_binary', dtype=int))  # Codificación binaria
        ])

        # Combinación de los transformadores en un preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('ord', ordinal_transformer, ordinal_features),
                ('nom', nominal_transformer, nominal_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def transform(self):
        # Crear directorio si no existe
        self.config.transformed_train_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Codificar etiquetas
        self.encode_labels()
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        train_df, test_df = train_test_split(self.df, test_size=0.3, random_state=42)
        
        # Separar la columna objetivo para evitar que sea escalada
        y_train = train_df.pop(self.config.target_column)
        y_test = test_df.pop(self.config.target_column)
        
        preprocessor = self.get_preprocessor()
        
        # Aplicar la transformación
        train_df_transformed = preprocessor.fit_transform(train_df)
        test_df_transformed = preprocessor.transform(test_df)
        
        # Obtener los nombres de las columnas transformadas
        numeric_features = preprocessor.transformers_[0][2]
        ordinal_features = preprocessor.transformers_[1][2]
        nominal_features = preprocessor.transformers_[2][2]
        nominal_feature_names = preprocessor.transformers_[2][1]['onehot'].get_feature_names_out(nominal_features)
        
        # Combinar los nombres de las columnas transformadas
        feature_names = np.concatenate([numeric_features, ordinal_features, nominal_feature_names])
        
        # Convertir los datos transformados en DataFrame con las columnas originales
        train_df_transformed = pd.DataFrame(train_df_transformed, columns=feature_names)
        test_df_transformed = pd.DataFrame(test_df_transformed, columns=feature_names)
        
        # Añadir de nuevo la columna objetivo a los DataFrames transformados
        train_df_transformed[self.config.target_column] = y_train.values
        test_df_transformed[self.config.target_column] = y_test.values
        
        # Guardar los conjuntos de datos de entrenamiento y prueba transformados en Excel
        train_df_transformed.to_excel(self.config.transformed_train_data_path, index=False)
        test_df_transformed.to_excel(self.config.transformed_test_data_path, index=False)
        
        print("Data transformation completed successfully")
        print(f"Training data saved to {self.config.transformed_train_data_path}")
        print(f"Test data saved to {self.config.transformed_test_data_path}")