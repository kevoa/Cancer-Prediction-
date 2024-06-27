import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
import lightgbm as lgb
import xgboost as xgb
import torch
from CancerPrediction.entity.config_entity import ModelTrainerConfig
from CancerPrediction import logger
import json

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: dict, seed: int = 42):
        self.config = config
        self.params = params
        self.seed = seed
        set_seed(self.seed)

    def train(self):
        # Cargar datos de entrenamiento y prueba
        train_data = pd.read_excel(self.config.train_data_path)
        test_data = pd.read_excel(self.config.test_data_path)

        # Combinar datos de entrenamiento y prueba
        df_combined = pd.concat([train_data, test_data], axis=0)

        # Seleccionar características importantes
        X_real = df_combined[self.config.important_features]
        y_real = df_combined[self.config.target_column]

        # Identificar clases minoritarias
        class_counts = y_real.value_counts()
        minority_classes = class_counts[class_counts < class_counts.median()].index

        # Separar datos de clases minoritarias
        X_minority = X_real[y_real.isin(minority_classes)]
        y_minority = y_real[y_real.isin(minority_classes)]

        # Entrenar el modelo CTGAN solo con las clases minoritarias
        ctgan_params = self.params['CTGAN']
        model = CTGAN(**ctgan_params)
        model.fit(X_minority)

        # Generar datos sintéticos para las clases minoritarias
        synthetic_data_minority = model.sample(len(X_minority))

        # Asignar etiquetas correctas a los datos sintéticos generados
        synthetic_data_minority[self.config.target_column] = np.random.choice(minority_classes, len(synthetic_data_minority))

        # Separar características y etiquetas de los datos sintéticos generados
        X_synthetic = synthetic_data_minority[self.config.important_features]
        y_synthetic = synthetic_data_minority[self.config.target_column]

        # Combinar datos reales y datos sintéticos generados
        X_combined = pd.concat([X_real, X_synthetic], axis=0)
        y_combined = pd.concat([y_real, y_synthetic], axis=0)

        # Aplicar SMOTE para sobremuestrear las clases minoritarias en el conjunto combinado
        smote = SMOTE(random_state=self.seed)
        X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=self.seed, stratify=y_resampled)

        # Definir los modelos individuales con regularización
        rf_clf = RandomForestClassifier(random_state=self.seed, **self.params['RandomForest'])
        gb_clf = GradientBoostingClassifier(random_state=self.seed, **self.params['GradientBoosting'])
        lgbm_clf = lgb.LGBMClassifier(random_state=self.seed, **self.params['LightGBM'])
        xgb_clf = xgb.XGBClassifier(random_state=self.seed, **self.params['XGBoost'])

        # Definir el Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf_clf),
                ('gb', gb_clf),
                ('lgbm', lgbm_clf),
                ('xgb', xgb_clf)
            ],
            voting='soft'  # 'soft' uses predicted probabilities
        )

        # Entrenar el Voting Classifier con todos los datos resampleados
        voting_clf.fit(X_train, y_train)

        # Guardar el modelo entrenado
        joblib.dump(voting_clf, os.path.join(self.config.root_dir, self.config.model_name))

        # Evaluar el modelo
        y_pred = voting_clf.predict(X_test)
        print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

        # Guardar las características utilizadas para el entrenamiento
        model_features = X_train.columns.tolist()
        with open(os.path.join(self.config.root_dir, 'model_features.json'), 'w') as f:
            json.dump(model_features, f)

