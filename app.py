from flask import request, jsonify, Flask, render_template, send_file
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
app = Flask(__name__)

with open('LR.pkl', 'rb') as f:
    LR = pickle.load(f)

feature_names = ['sex', 'cp','trestbps','chol', 'restecg', 'thalach','exang', 'slope', 'ca', 'thal']

@app.route('/')
def start():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    features_df = pd.DataFrame(features, columns=feature_names)
    print(features_df)
    prediction = LR.predict(features_df)
    
    if prediction[0] == 0:
        return jsonify({'prediction': 'No Heart Disease'})
    else:
        return jsonify({'prediction': 'Heart Disease'})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/static/<filename>')
def send_image(filename):
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    app.run(debug=True)
