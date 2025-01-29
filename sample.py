import pdfplumber
import re

def extract_soil_data(pdf_path):
    # Initialize a dictionary to store the extracted values
    data = {"Nitrogen": None, "Phosphorus": None, "Potassium": None, "pH": None}

    # Set default values for bulk density (g/cm³) and depth (cm)
    bulk_density = 1.3  # g/cm³ (can vary based on soil type)
    depth = 15  # cm (typical for topsoil)

    # Open the PDF and extract the text
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    # Define the regular expressions to capture the relevant values (including units)
    patterns = {
        "Nitrogen": r"Nitrogen\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "Phosphorus": r"Phosphorus\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "Potassium": r"Potassium\s*[:\-]?\s*([\d.]+)(%|Kg/ha)?",
        "pH": r"pH\s*[:\-]?\s*([\d.]+)"
    }

    # Search for the patterns in the extracted text
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))  # Convert to float if a match is found
            unit = match.group(2) if len(match.groups()) > 1 else None  # Check if unit exists

            # If the unit is 'percentage', convert it to 'Kg/ha'
            if unit == "%":
                # Convert percentage to Kg/ha
                value = (value * bulk_density * depth) / 100

            data[key] = value

    return data

# Example usage
pdf_file = "soil_report_sample1.pdf"
extracted_data = extract_soil_data(pdf_file)
print("Extracted Soil Data:", extracted_data)
