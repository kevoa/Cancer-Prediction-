import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.label_encoder = joblib.load(Path('artifacts/data_transformation/Tumor type_label_encoder.joblib'))
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.joblib'))
        self.important_features = [
            'sFas (pg/ml)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'CA 15-3 (U/ml)', 'CA19-9 (U/ml)',
            'CA-125 (U/ml)', 'TIMP-2 (pg/ml)', 'TGFa (pg/ml)', 'Leptin (pg/ml)',
            'IL-8 (pg/ml)', 'IL-6 (pg/ml)', 'AFP (pg/ml)', 'GDF15 (ng/ml)', 'Prolactin (pg/ml)',
            'HGF (pg/ml)', 'CD44 (ng/ml)', 'Midkine (pg/ml)', 'Thrombospondin-2 (pg/ml)',
            'TIMP-1 (pg/ml)', 'HE4 (pg/ml)'
        ]

    def preprocess_input(self, input_data):
        data_dict = {feature: input_data[idx] for idx, feature in enumerate(self.important_features)}
        ordered_data = pd.DataFrame([data_dict])
        return self.preprocessor.transform(ordered_data)

    def predict(self, data):
        data = self.preprocess_input(data)
        print(f"Datos de entrada: {data}")
        prediction = self.model.predict(data)
        print(f"Predicción codificada: {prediction}")
        decoded_prediction = self.label_encoder.inverse_transform(prediction)
        print(f"Predicción decodificada: {decoded_prediction}")
        return decoded_prediction
