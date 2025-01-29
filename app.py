from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json
import pdfplumber
import re
import os

app = Flask(__name__)

# Load the ML model and column names
with open("models/crop_model.pkl", "rb") as f:
    final_model = pickle.load(f)

with open("models/columns.json", "r") as f:
    columns = json.load(f)["d_col"]

# Function to extract soil data from PDF
def extract_soil_data(pdf_path):
    data = {
        "n": None, "p": None, "k": None, "ph": None,
        "humidity": None, "rainfall": None, "temperature": None
    }

    bulk_density = 1.3  # g/cm³
    depth = 15  # cm

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

    patterns = {
        "n": r"Nitrogen\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "p": r"Phosphorus\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "k": r"Potassium\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "ph": r"pH\s*[:\-]?\s*([\d.]+)",
        "humidity": r"Humidity\s*[:\-]?\s*([\d.]+)%",
        "rainfall": r"Rainfall\s*[:\-]?\s*([\d.]+)\s*mm",
        "temperature": r"Temperature\s*[:\-]?\s*([\d.]+)\s*°?C"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2) if len(match.groups()) > 1 else None

            if unit == "%":  # Convert % to Kg/ha if necessary
                value = (value * bulk_density * depth) / 100

            data[key] = value

    return data

@app.route("/")
def home():
    return "🌾 Crop Prediction API is running! Send a POST request to /predict or /predict_from_pdf."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        input_df = input_df[columns]
        prediction = final_model.predict(input_df)
        return jsonify({"predicted_crop": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/predict_from_pdf", methods=["POST"])
def predict_from_pdf():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        pdf_path = os.path.join("uploads", file.filename)
        file.save(pdf_path)

        # Extract soil data
        extracted_data = extract_soil_data(pdf_path)

        # Replace missing values with default values (e.g., 0 or mean)
        for col in columns:
            if extracted_data[col] is None:
                extracted_data[col] = 0  # You can change 0 to a mean/median value if needed

        # Convert extracted data into a DataFrame
        input_df = pd.DataFrame([extracted_data])

        # Ensure columns match the model's expected format
        input_df = input_df[columns]

        # Predict the crop
        prediction = final_model.predict(input_df)

        # Remove the saved PDF after processing
        os.remove(pdf_path)

        return jsonify({"predicted_crop": prediction[0], "extracted_data": extracted_data})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Ensure uploads folder exists
    app.run(debug=True, port=5000)
