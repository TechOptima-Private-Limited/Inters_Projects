Urban EV Infrastructure Planning Tool
Overview
The Urban EV Infrastructure Planning Tool is a web-based application designed to assist urban planners, civil engineers, EV charging companies, and municipal boards (e.g., HMDA) in assessing the feasibility of adding electric vehicle (EV) chargers in specific areas of Hyderabad. The tool leverages power usage data and OptGPT analysis to evaluate grid capacity and provide data-driven recommendations.
Features

Area Selection: Users can select from a list of predefined Hyderabad HV areas.
Charger Feasibility Analysis: Evaluates whether adding a specified number of EV chargers is safe based on current load data.
Alternative Suggestions: Recommends alternative areas if the selected area is overloaded.
Interactive UI: Built with Streamlit for a user-friendly interface.
Backend Processing: Uses Flask and OptGPT (via Ollama) for data analysis and response generation.

Requirements
To run this application, ensure you have the following dependencies installed:

Python: Version 3.8 or higher
Python Packages:streamlit==1.29.0
flask==2.3.3
flask-cors==4.0.0
pandas==2.0.3
requests==2.31.0


Ollama: Configured to run the OptGPT model (optgpt:7b) locally at http://192.168.1.117:11434/api/generate
Dataset: A CSV file (65e18072-d042-4879-ad92-5d6e7ba0cdfd.csv) containing power usage data for Hyderabad HV areas.

Installation

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare the Dataset:

Place the dataset file (Telangana EV Consumption dataser 2025.csv) in the project root directory.
Ensure the CSV contains columns: area, totservices, billdservices, units, and load.


Set Up Ollama:

Install and configure Ollama on your local machine.
Ensure the OptGPT model (optgpt:7b) is available and accessible at http://192.168.1.117:11434/api/generate.
Update the OLLAMA_URL in the backend script if your Ollama instance runs on a different host or port.



Usage:

Run the Backend:

Start the Flask server:python backend.py


The backend will run on http://localhost:5000.


Run the Frontend:

In a separate terminal, start the Streamlit app:streamlit run frontend.py


The frontend will open in your default browser at http://localhost:8501.


Interact with the Tool:

Select an area from the dropdown menu.
Specify the number of EV chargers to add (1–20).
Click "Submit" to receive an OptGPT-generated analysis, including:
Current load status
Feasibility of adding chargers
Alternative area suggestions (if applicable)
Final recommendations

Notes

Ensure the backend server is running before starting the frontend.
The dataset file must match the expected format and column names.
The Ollama service must be running and accessible for the backend to function.
The tool is designed for Hyderabad HV areas listed in the backend script (VALID_HYDERABAD_HV_AREAS).

Troubleshooting

Backend Connection Error: Verify that the Flask server and Ollama service are running and accessible.
Dataset Not Found: Ensure the CSV file is in the project root and correctly named.
Dependency Issues: Use Python 3.8+ and verify all packages are installed correctly.


