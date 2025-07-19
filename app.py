from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)

        input_data = [float(i) for i in data['input']]
        print("Converted to float:", input_data)

        prediction = model.predict([input_data])
        print("Model prediction:", prediction)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Could not get prediction'})
if __name__ == "__main__":
    app.run(debug=True)