from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/fake_profile_model.pkl')

# Load model on startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

feature_order = [
    'userFollowerCount',
    'userFollowingCount',
    'userBiographyLength',
    'userMediaCount',
    'userHasProfilPic',
    'userIsPrivate',
    'usernameDigitCount',
    'usernameLength'
]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    try:
        features = [data.get(f, 0) for f in feature_order]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        return jsonify({'prediction': int(prediction), 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
