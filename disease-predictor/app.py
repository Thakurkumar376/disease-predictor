from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the single PKL file
bundle = joblib.load('healthcare_model_bundle.pkl')

clf = bundle["decision_tree_model"]
le = bundle["label_encoder"]
symptoms_dict = bundle["symptoms_dict"]
precaution_dict = bundle["precaution_dict"]
description_dict = bundle["description_dict"]

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Web form prediction
@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms').lower().split(',')

    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        s = s.strip().replace(" ", "_")
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    prediction = clf.predict([input_vector])
    disease = le.inverse_transform(prediction)[0]

    description = description_dict.get(disease, "No description available.")
    precautions = precaution_dict.get(disease, [])

    return render_template(
        'index.html',
        prediction=disease,
        description=description,
        precautions=precautions
    )

# REST API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    symptoms = data.get("symptoms", [])

    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        s = s.strip().replace(" ", "_")
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    prediction = clf.predict([input_vector])
    disease = le.inverse_transform(prediction)[0]

    return jsonify({
        "predicted_disease": disease,
        "description": description_dict.get(disease, ""),
        "precautions": precaution_dict.get(disease, [])
    })

if __name__ == '__main__':
    app.run(debug=True)
