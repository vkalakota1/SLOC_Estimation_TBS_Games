from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load Trained Model
model = pickle.load(open('saved_model.pkl', 'rb'))

@app.route('/')
def home():
    return jsonify({"message": "SLOC Estimation API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = ['NRUL', 'MGOP', 'NPLY', 'ANIM', 'MAP', 'UNU', 'NP']

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        return jsonify({"Predicted_SLOC": round(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
