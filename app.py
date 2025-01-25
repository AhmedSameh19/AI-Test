from flask import request, jsonify, Flask, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained Logistic Regression model
with open('LR.pkl', 'rb') as f:
    LR = pickle.load(f)

# Feature names expected by the model
feature_names = ['age', 'sex', 'cp','trestbps','chol', 'restecg', 'thalach','exang', 'slope', 'ca', 'thal']

@app.route('/')
def start():
    return render_template('index.html')  # Render the updated index.html

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON data received from the frontend
    data = request.get_json(force=True)
    print(data)
    # Ensure features are in the correct format (2D array)
    features = np.array(data['features']).reshape(1, -1)
    
    # Convert features into a DataFrame with proper column names
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Make a prediction using the Logistic Regression model
    prediction = LR.predict(features_df)
    
    # Return prediction result as JSON
    if prediction[0] == 0:
        return jsonify({'prediction': 'No Heart Disease'})
    else:
        return jsonify({'prediction': 'Heart Disease'})

if __name__ == '__main__':
    app.run(debug=True)
