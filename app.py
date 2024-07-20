from flask import Flask, render_template, request
import numpy as np
from CancerPrediction.pipeline.prediction import PredictionPipeline
import os

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [
                'sFas (pg/ml)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'CA 15-3 (U/ml)', 'CA19-9 (U/ml)',
                'CA-125 (U/ml)', 'TIMP-2 (pg/ml)', 'TGFa (pg/ml)', 'Leptin (pg/ml)',
                'IL-8 (pg/ml)', 'IL-6 (pg/ml)', 'AFP (pg/ml)', 'GDF15 (ng/ml)', 'Prolactin (pg/ml)',
                'HGF (pg/ml)', 'CD44 (ng/ml)', 'Midkine (pg/ml)', 'Thrombospondin-2 (pg/ml)',
                'TIMP-1 (pg/ml)', 'HE4 (pg/ml)'
            ]
            
            input_data = []
            for feature in features:
                input_data.append(request.form[feature])
            
            data = np.array(input_data)

            obj = PredictionPipeline()
            prediction = obj.predict(data)

            return render_template('results.html', prediction=prediction[0])  

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
