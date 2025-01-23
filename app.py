from flask import request,jsonify,Flask,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('LogisticRegression.pkl', 'rb') as f:
    LR = pickle.load(f)
with open('RandomForest.pkl', 'rb') as f:
    RF = pickle.load(f)
feature_names = ['age', 'sex', 'cp', 'restecg', 'thalach', 'slope', 'ca', 'thal']
@app.route('/')
def start():
    return render_template('index.html')
 

@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON data from the frontend
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Ensure it's a 2D array

    # Convert features into a DataFrame with correct column names
    features_df = pd.DataFrame(features, columns=feature_names)
    # Make prediction using the loaded model
    prediction = LR.predict(features_df)
    if(prediction[0]==0):
        return jsonify({'prediction': 'No Heart Disease'})
    else:
        return jsonify({'prediction': 'Heart Disease'})
    #return jsonify({'prediction': int(prediction[0])})  # Return the result as JSON

@app.route('/predict2', methods=['POST'])
def predict2():
    # Parse the JSON data from the frontend
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)  # Ensure it's a 2D array

    # Convert features into a DataFrame with correct column names
    features_df = pd.DataFrame(features, columns=feature_names)
    # Make prediction using the loaded model
    prediction = RF.predict(features_df)
    if(prediction[0]==0):
        return jsonify({'prediction': 'No Heart Disease'})
    else:
        return jsonify({'prediction': 'Heart Disease'})

if __name__ == '__main__':
    app.run(debug=True)