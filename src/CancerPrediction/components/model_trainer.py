
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
from CancerPrediction import logger
from CancerPrediction.entity.config_entity import ModelTrainerConfig
import torch 

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        set_seed(self.seed)

    def train(self):
        # Cargar datos
        train_data = pd.read_excel(self.config.train_data_path)

        # Seleccionar características importantes
        X_train = train_data[self.config.important_features]
        y_train = train_data[self.config.target_column]

        # Identificar clases minoritarias
        class_counts = y_train.value_counts()
        minority_classes = class_counts[class_counts < class_counts.median()].index

        # Separar datos de clases minoritarias
        X_minority = X_train[y_train.isin(minority_classes)]
        y_minority = y_train[y_train.isin(minority_classes)]

        # Entrenar el modelo CTGAN solo con las clases minoritarias
        model = CTGAN(epochs=200)
        model.fit(X_minority)

        # Generar datos sintéticos para las clases minoritarias
        synthetic_data_minority = model.sample(len(X_minority))

        # Asignar etiquetas correctas a los datos sintéticos generados
        synthetic_data_minority[self.config.target_column] = np.random.choice(minority_classes, len(synthetic_data_minority))

        # Separar características y etiquetas de los datos sintéticos generados
        X_synthetic = synthetic_data_minority[self.config.important_features]
        y_synthetic = synthetic_data_minority[self.config.target_column]

        # Combinar datos reales y datos sintéticos generados
        X_combined = pd.concat([X_train, X_synthetic], axis=0)
        y_combined = pd.concat([y_train, y_synthetic], axis=0)

        # Aplicar SMOTE para sobremuestrear las clases minoritarias en el conjunto combinado
        smote = SMOTE(random_state=self.seed)
        X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=self.seed, stratify=y_resampled)

        # Definir los modelos individuales con regularización
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.seed)
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.seed)
        lgbm_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=self.seed)
        xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.seed)

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

        logger.info(f"Model training completed and saved to {os.path.join(self.config.root_dir, self.config.model_name)}")