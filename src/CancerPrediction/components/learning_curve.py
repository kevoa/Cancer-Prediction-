import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import joblib
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
import lightgbm as lgb
import xgboost as xgb
import os
import torch

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Ruta de los archivos de configuración y datos
train_data_path = os.path.join("artifacts", "data_transformation", "train_df.xlsx")
test_data_path = os.path.join("artifacts", "data_transformation", "test_df.xlsx")
model_path = os.path.join("artifacts", "model_trainer", "model.joblib")

# Cargar datos de entrenamiento y prueba
train_data = pd.read_excel(train_data_path)
test_data = pd.read_excel(test_data_path)
df_combined = pd.concat([train_data, test_data], axis=0)

# Seleccionar características importantes y la columna objetivo
important_features = [
    'sFas (pg/ml)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'CA 15-3 (U/ml)', 'CA19-9 (U/ml)', 'CA-125 (U/ml)',
    'TIMP-2 (pg/ml)', 'TGFa (pg/ml)', 'Sex_1', 'Leptin (pg/ml)', 'IL-8 (pg/ml)', 'IL-6 (pg/ml)',
    'AFP (pg/ml)', 'GDF15 (ng/ml)', 'Prolactin (pg/ml)', 'HGF (pg/ml)', 'CD44 (ng/ml)', 'Midkine (pg/ml)',
    'Thrombospondin-2 (pg/ml)', 'TIMP-1 (pg/ml)', 'HE4 (pg/ml)'
]
target_column = "Tumor type"

# Preparar datos reales
X_real = df_combined[important_features]
y_real = df_combined[target_column]

# Identificar clases minoritarias
class_counts = y_real.value_counts()
minority_classes = class_counts[class_counts < class_counts.median()].index

# Separar datos de clases minoritarias
X_minority = X_real[y_real.isin(minority_classes)]
y_minority = y_real[y_real.isin(minority_classes)]

# Entrenar el modelo CTGAN solo con las clases minoritarias
ctgan_params = {'epochs': 200}
model = CTGAN(**ctgan_params)
model.fit(X_minority)

# Generar datos sintéticos para las clases minoritarias
synthetic_data_minority = model.sample(len(X_minority))

# Asignar etiquetas correctas a los datos sintéticos generados
synthetic_data_minority[target_column] = np.random.choice(minority_classes, len(synthetic_data_minority))

# Preparar datos sintéticos generados
X_synthetic = synthetic_data_minority[important_features]
y_synthetic = synthetic_data_minority[target_column]

# Combinar datos reales y datos sintéticos generados
X_combined = pd.concat([X_real, X_synthetic], axis=0)
y_combined = pd.concat([y_real, y_synthetic], axis=0)

# Aplicar SMOTE para sobremuestrear las clases minoritarias en el conjunto combinado
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

# Definir los modelos individuales con regularización
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgbm_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42)
xgb_clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

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

# Generar curvas de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(
    voting_clf, X_resampled, y_resampled, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calcular medias y desviaciones estándar
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Graficar curvas de aprendizaje
plt.figure()
plt.title("Learning Curves with CTGAN and SMOTE")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.ylim((0.7, 1.01))
plt.grid()

# Graficar la curva de aprendizaje del conjunto de entrenamiento
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

# Graficar la curva de aprendizaje del conjunto de validación
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")

plt.legend(loc="best")
plt.show()
