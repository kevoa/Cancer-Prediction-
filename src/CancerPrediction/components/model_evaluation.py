import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import joblib
from CancerPrediction import logger
from CancerPrediction.utils.common import save_json
from CancerPrediction.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, y_true, y_pred):
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return precision, recall, f1

    def log_into_mlflow(self):
        test_data = pd.read_excel(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Seleccionar características importantes directamente de la configuración
        X_test = test_data[self.config.important_features]
        y_test = test_data[self.config.target_column]

        # Configurar MLflow
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_password

        with mlflow.start_run():
            # Realizar predicciones y evaluar el modelo en el conjunto de prueba
            y_pred = model.predict(X_test)

            # Calcular métricas
            precision, recall, f1 = self.eval_metrics(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Guardar métricas localmente
            scores = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist(),
                "classification_report": class_report
            }
            save_json(path=self.config.metric_file_name, data=scores)

            # Registrar métricas en MLflow
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_artifact(self.config.metric_file_name)
            
            for model_name, params in self.config.all_params.items():
                for param_name, param_value in params.items():
                    try:
                        mlflow.log_param(f"{model_name}_{param_name}", param_value)
                    except Exception as e:
                        logger.error(f"Error logging parameter {model_name}_{param_name}: {e}")

            # Registrar el modelo
            mlflow.sklearn.log_model(model, "model", registered_model_name="VotingClassifierModel")