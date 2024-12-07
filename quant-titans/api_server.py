from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']  # Expect a list of feature values
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(port=5000)
