from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import re

app = Flask(__name__)
CORS(app)

# Ollama config
OLLAMA_URL = "http://192.168.1.117:11434/api/generate"
MODEL_NAME = "optgpt:7b"

# Load dataset
DATASET_PATH = "65e18072-d042-4879-ad92-5d6e7ba0cdfd.csv"
df = pd.read_csv(DATASET_PATH)

# Define valid Hyderabad HV areas
VALID_HYDERABAD_HV_AREAS = [
    "CHANDANAGAR-HV", "KOMPALLY HV", "YAPRAL HV", "MIYAPUR HV", "MOULA - ALI (HV)",
    "NACHARAM HV CT METERS", "KAMALA NAGAR (HV)", "ASRAO NAGAR (HV)", "ATTIVELLY HV",
    "HV GACHIBOWLI", "HV WHITEFIELD",
    "GRC HV TARNAKA", "GRC HV SURYA NAGAR", "GRCHV CHUDIBAZAR", "GRCHV DATTACOLONY"
]

@app.route("/api/ev-plan", methods=["POST"])
def ev_plan():
    data = request.get_json()
    selected_area = data.get("area")
    num_chargers = data.get("num_chargers")

    if not selected_area or num_chargers is None:
        return jsonify({"error": "Area and number of chargers are required."}), 400

    selected_data = df[df["area"] == selected_area]
    if selected_data.empty:
        return jsonify({"error": "Area not found in the dataset."}), 404

    area_info = selected_data.iloc[0].to_dict()

    # Strict prompt to avoid intro
    prompt = f"""
Given only the data below, respond with ONLY the 4 sections listed, no introduction or extra explanation:

Area: {selected_area}
- Total Services: {area_info.get('totservices')}
- Billed Services: {area_info.get('billdservices')}
- Units: {area_info.get('units')}
- Load: {area_info.get('load')} kW
- New Chargers Requested: {num_chargers}

Respond using the exact structure below — do not add or remove sections:

1. **Current Load Status**  
- Is the area already overloaded? Use only the data above.

2. **Feasibility of Adding {num_chargers} Chargers**  
- Is it safe? Why or why not? Use the numbers.

3. **Alternative Suggestions (if needed)**  
- Suggest 2 areas ONLY from this list: {', '.join(VALID_HYDERABAD_HV_AREAS)}
- Justify based on load values only.

4. **Final Recommendation**  
- Bullet points only.
- NO introduction, NO summary, NO other content.
"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 10000,
            "top_p": 0.95,
            "n_predict": 6000,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        result = response.json()

        raw_response = result.get("response", "")

        # Post-process to remove anything before "1."
        match = re.search(r"(1\.\s\*\*Current Load Status\*\*.+)", raw_response, re.DOTALL)
        cleaned_response = match.group(1).strip() if match else raw_response.strip()

        return jsonify({"response": cleaned_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
