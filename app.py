from flask import Flask, request, jsonify, render_template
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
'''
# Function to extract soil data from PDF (robust version)
def extract_soil_data(pdf_path):
    data = {
        "n": None, "p": None, "k": None, "ph": None,
        "humidity": None, "rainfall": None, "temperature": None
    }

    bulk_density = 1.3  # g/cm³
    depth = 15  # cm

    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    text = text.replace("\n", " ")  # flatten text for easier matching

    # More flexible regex patterns
    patterns = {
        "n": r"(?:N\b|Nitrogen|Avail\.? Nitrogen)\s*[:\-]?\s*([\d.]+)\s*(%|kg/ha|ppm|mg/kg)?",
        "p": r"(?:P\b|Phosphorus|Avail\.? Phosphorus)\s*[:\-]?\s*([\d.]+)\s*(%|kg/ha|ppm|mg/kg)?",
        "k": r"(?:K\b|Potassium|Avail\.? Potassium)\s*[:\-]?\s*([\d.]+)\s*(%|kg/ha|ppm|mg/kg)?",
        "ph": r"(?:pH)\s*[:\-]?\s*([\d.]+)",
        "humidity": r"(?:Humidity|Moisture)\s*[:\-]?\s*([\d.]+)\s*%?",
        "rainfall": r"(?:Rainfall|Rain)\s*[:\-]?\s*([\d.]+)\s*(mm|cm)?",
        "temperature": r"(?:Temperature|Temp)\s*[:\-]?\s*([\d.]+)\s*(°?C|Celsius)?"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2).lower() if match.lastindex and match.group(2) else None

                # Convert % to approx kg/ha if possible
                if unit and "%" in unit:
                    value = (value * bulk_density * depth) / 100

                data[key] = value
            except Exception:
                data[key] = None  # fallback if parsing fails

    return data
'''

FIELD_MAP = {
    "n": ["Nitrogen", "N", "Total N", "Nitrogen (N)"],
    "p": ["Phosphorus", "P", "Available P", "P2O5"],
    "k": ["Potassium", "K", "K2O"],
    "ph": ["pH", "Soil pH", "Acidity"],
    "humidity": ["Humidity", "Moisture", "Water Content"],
    "rainfall": ["Rainfall", "Rain", "Precipitation"],
    "temperature": ["Temperature", "Temp"]
}

def extract_soil_data(pdf_path):
    """
    Generalized extractor for lawn/garden soil reports.

    Returns a dict with keys: n, p, k, ph, humidity, rainfall, temperature.
    - pH, P, K: averaged if found in tables
    - N: parsed from "Actual Nutrient Need" or via flexible regex
    - Humidity, Rainfall, Temperature: matched flexibly
    """
    data = {key: None for key in FIELD_MAP}

    ph_vals, p_vals, k_vals = [], [], []
    full_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            full_text_parts.append(txt)

            # Try to extract LABORATORY ANALYSIS tables
            tables = page.extract_tables() or []
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # Find the header row
                header_idx, header = None, None
                for idx, row in enumerate(table[:4]):
                    if not row:
                        continue
                    row_lc = [(c or "").strip().lower() for c in row]
                    if "sample" in row_lc and any("ph" in (c or "") for c in row_lc):
                        header_idx, header = idx, row
                        break
                if header is None:
                    continue

                # Normalize header -> build column map
                def norm(s): return re.sub(r"[^a-z]+", "", (s or "").lower())
                colmap = {}
                for i, cell in enumerate(header):
                    n = norm(cell or "")
                    if n == "ph":
                        colmap["ph"] = i
                    elif "phosphorus" in n or n == "p":
                        colmap["p"] = i
                    elif "potassium" in n or n == "k":
                        colmap["k"] = i

                # Collect numeric values from data rows
                for row in table[header_idx + 1:]:
                    def get_float(col_key):
                        try:
                            j = colmap[col_key]
                            val = (row[j] or "").replace(",", "").strip()
                            return float(val) if val != "" else None
                        except Exception:
                            return None

                    if "ph" in colmap:
                        v = get_float("ph")
                        if v is not None: ph_vals.append(v)
                    if "p" in colmap:
                        v = get_float("p")
                        if v is not None: p_vals.append(v)
                    if "k" in colmap:
                        v = get_float("k")
                        if v is not None: k_vals.append(v)

    # Combine text across pages
    full_text = "\n".join(full_text_parts)

    # Assign averages if we found any values
    if ph_vals: data["ph"] = round(sum(ph_vals) / len(ph_vals), 2)
    if p_vals:  data["p"]  = round(sum(p_vals)  / len(p_vals),  2)
    if k_vals:  data["k"]  = round(sum(k_vals)  / len(k_vals),  2)

    # Extract Nitrogen (N)
    n_val = None
    n_anchor = full_text.find("Nitrogen (N)")
    if n_anchor != -1:
        window = full_text[n_anchor:n_anchor + 600]
        for m in re.finditer(r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)", window):
            a, b, c = m.groups()
            if (a, b, c) == ("2", "5", "2"):
                continue
            if any("." in x for x in (a, b, c)):
                try:
                    n_val = float(a)
                    break
                except ValueError:
                    pass
    if n_val:
        data["n"] = n_val

    # Generic fallback: regex search for each field
    def find_field(field_key):
        variants = FIELD_MAP[field_key]
        pat = rf"(?:{'|'.join(map(re.escape, variants))})[^0-9\-]*([-+]?\d*\.?\d+)"
        m = re.search(pat, full_text, flags=re.IGNORECASE)
        return float(m.group(1)) if m else None

    for key in ["n", "humidity", "rainfall", "temperature"]:
        if data[key] is None:
            val = find_field(key)
            if val is not None:
                data[key] = val

    return data


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_from_pdf", methods=["POST"])
def predict_from_pdf():
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        pdf_path = os.path.join("output", file.filename)
        file.save(pdf_path)

        extracted_data = extract_soil_data(pdf_path)

        # Fill missing values with 0
        input_data = {col: extracted_data.get(col, 0) for col in columns}
        input_df = pd.DataFrame([input_data])[columns]

        prediction = final_model.predict(input_df)

        os.remove(pdf_path)

        return jsonify({
            "predicted_crop": prediction[0],
            "extracted_data": extracted_data
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    app.run(debug=True, port=5000)
