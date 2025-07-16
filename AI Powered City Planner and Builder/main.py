import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import PyPDF2
import tempfile
import json
import time

# Configure Streamlit page
st.set_page_config(page_title="AI Powered City Planner and Builder", layout="wide")

# Add logo to the top left
st.image("logoimage.png", width=200)  # Replace with your logo file path

# Hardcoded Gemini API Key
GEMINI_API_KEY = "YOUR GEMINI KEY"  # Replace with your actual Gemini API key

# Initialize Gemini API
@st.cache_resource
def init_gemini_api():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        return model
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {e}")
        return None

# Function to convert PDF to images
def pdf_to_images(pdf_path):
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path)
        return images
    except ImportError:
        return [Image.new('RGB', (800, 600), color='white')]

# Function to preprocess and fix JSON response
def preprocess_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            return None

# Function to normalize polygons
def normalize_polygons(polygons):
    normalized = []
    for polygon in polygons:
        bbox = polygon.get("approx_bbox", [])
        if isinstance(bbox, list) and all(isinstance(item, list) for item in bbox):
            bbox = bbox[0] if bbox else [0, 0, 0, 0]
        elif not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        normalized.append({
            "color": polygon.get("color", ""),
            "description": polygon.get("description", ""),
            "approx_bbox": bbox,
            "geo_coordinates": polygon.get("geo_coordinates", [])
        })
    return normalized

# Function to process document/image with Gemini API
def process_with_gemini(model, image, is_map=False, retries=3, delay=10):
    if not model:
        return None
    for attempt in range(retries):
        try:
            # Resize image to reduce token usage
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            # Convert image to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Prompt tailored to document or map
            if is_map:
                prompt = """
                Analyze the provided planning map. Extract:
                1. Text content (e.g., labels, place names) as an array of objects with 'text' and 'approx_bbox' ([x_min, y_min, x_max, y_max]).
                2. Annotations (e.g., color-coded zones) as an array of objects with 'type', 'description', 'approx_bbox' ([x_min, y_min, x_max, y_max]), and 'color'.
                3. Map features including:
                   - Polygons: Array of objects with 'color', 'description', 'approx_bbox' ([x_min, y_min, x_max, y_max]), and 'geo_coordinates' (array of [latitude, longitude] for each vertex if available).
                   - Addresses: Array of strings.
                   - Landmarks: Array of objects with 'name', 'type', and 'approx_location_bbox' ([x_min, y_min, x_max, y_max]).
                Determine the zone at the project location indicated on the map and include it as 'target_zone' in the output. If the location is within a buffer zone (e.g., near a water body), indicate this in 'target_zone' with 'buffer_zone': true.
                Ensure each 'approx_bbox' is a single [x_min, y_min, x_max, y_max] list, and 'geo_coordinates' contains exactly four [lat, lng] pairs if provided.
                Example:
                {
                  "text_content": [{"text": "Project Area", "approx_bbox": [10, 10, 50, 20]}],
                  "annotations": [{"type": "Shaded Area", "description": "Buffer Zone", "approx_bbox": [100, 100, 200, 200], "color": "Blue"}],
                  "map_features": {
                    "polygons": [{"color": "Blue", "description": "Buffer Zone", "approx_bbox": [100, 100, 200, 200], "geo_coordinates": [[0.0, 0.0], [0.0, 0.1], [-0.1, 0.1], [-0.1, 0.0]]}],
                    "addresses": ["Survey No. 123"],
                    "landmarks": [{"name": "Local Lake", "type": "Lake", "approx_location_bbox": [50, 50, 100, 100]}]
                  },
                  "target_zone": {"description": "Buffer Zone", "color": "Blue", "buffer_zone": true}
                }
                """
            else:
                prompt = """
                Analyze the provided proposal document. Extract:
                1. Text content (e.g., project title, location) as an array of objects with 'text' and 'approx_bbox' ([x_min, y_min, x_max, y_max]).
                2. Annotations (e.g., proposed uses) as an array of objects with 'type', 'description', 'approx_bbox' ([x_min, y_min, x_max, y_max]), and 'color' (if inferred).
                3. Map features including:
                   - Addresses: Array of strings.
                   - Landmarks: Array of objects with 'name', 'type', and 'approx_location_bbox' ([x_min, y_min, x_max, y_max]) if referenced.
                Example:
                {
                  "text_content": [{"text": "Sample Project", "approx_bbox": [50, 50, 200, 70]}],
                  "annotations": [{"type": "Proposed Use", "description": "50 villas", "approx_bbox": [50, 150, 200, 170], "color": "Purple"}],
                  "map_features": {
                    "addresses": ["Survey No. 123, Sample Area"],
                    "landmarks": [{"name": "Local Lake", "type": "Lake", "approx_location_bbox": [50, 250, 150, 270]}]
                  }
                }
                """
            
            response = model.generate_content([prompt, {"inline_data": {"mime_type": "image/png", "data": img_base64}}])
            extracted_data = preprocess_json_response(response.text)
            if extracted_data:
                extracted_data["map_features"]["polygons"] = normalize_polygons(extracted_data.get("map_features", {}).get("polygons", []))
                return extracted_data
            return None
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(delay)
                continue
            return None
    return None

# Function to get approval decision from Gemini
def get_approval_decision(model, document_data, map_data):
    if not model or not document_data or not map_data:
        st.error("Unable to determine approval due to missing data or API issues.")
        return
    
    # Combine document and map data into a single prompt for decision
    doc_text = json.dumps(document_data, indent=2)
    map_text = json.dumps(map_data, indent=2)
    prompt = f"""
    Based on the following proposal document data and map analysis, provide a decision on whether the project should be approved. Consider the proposed land use, the identified zones, features, and any buffer zone status. Provide a clear status (e.g., Approved, Rejected, Requires Review) and detailed reasons based on urban planning principles and environmental considerations.

    **Proposal Document Data:**
    {doc_text}

    **Map Analysis Data:**
    {map_text}

    Return the decision as a JSON object with this structure:
    {{
      "decision": {{
        "status": "string (e.g., Approved, Rejected, Requires Review)",
        "reasons": ["string", "string", ...]
      }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        decision_data = preprocess_json_response(response.text)
        if decision_data and "decision" in decision_data:
            st.subheader("Project Approval Decision")
            st.write(f"**Status:** {decision_data['decision']['status']}")
            st.write("**Reasons:**")
            for reason in decision_data['decision']['reasons']:
                st.write(f"- {reason}")
        else:
            st.error("Failed to parse approval decision.")
    except Exception as e:
        st.error("Failed to generate approval decision.")

# Main app
def main():
    st.title("AI Powered City Planner and Builder")
    st.write("Upload a proposal document (PDF) and reference map (image) to determine project approval.")

    # Initialize Gemini model
    model = init_gemini_api()
    if not model:
        return

    # File uploaders for document and map
    uploaded_files = st.file_uploader("Upload proposal document (PDF) and reference map (image)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        document_data = None
        map_data = None
        
        # Categorize uploaded files
        for file in uploaded_files:
            if file.type == "application/pdf":
                document_data = file
            elif file.type in ["image/png", "image/jpg", "image/jpeg"]:
                map_data = file

        if not document_data or not map_data:
            st.warning("Please upload both a PDF document and an image map to proceed.")
            return

        # Process document with spinner
        with st.spinner("Processing Proposal Document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(document_data.read())
                temp_file_path = temp_file.name
            images = pdf_to_images(temp_file_path)
            os.unlink(temp_file_path)
            document_extracted_data = process_with_gemini(model, images[0], is_map=False)
            if not document_extracted_data:
                st.error("Failed to process proposal document.")
                return

        # Process map with spinner
        with st.spinner("Processing Reference Map..."):
            map_image = Image.open(map_data).convert('RGB')
            map_extracted_data = process_with_gemini(model, map_image, is_map=True)
            if not map_extracted_data:
                st.error("Failed to process reference map.")
                return

        # Get and display approval decision
        get_approval_decision(model, document_extracted_data, map_extracted_data)

if __name__ == "__main__":
    main()