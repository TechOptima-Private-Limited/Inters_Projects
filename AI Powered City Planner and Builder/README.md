The AI Powered City Planner and Builder is a Streamlit-based web application for evaluating the feasibility of proposed development projects in Hyderabad. By leveraging the Google Gemini API, the tool analyzes proposal documents (PDFs) and reference maps (images) to extract key information, assess land use compatibility, and provide data-driven project approval decisions based on urban planning principles and environmental considerations.

Features:

Document and Map Analysis: Processes proposal documents (PDFs) and reference maps (PNG/JPG) to extract text content, annotations, and map features like polygons, addresses, and landmarks.
Approval Decision: Generates a clear approval decision (Approved, Rejected, Requires Review) with detailed reasons based on urban planning and environmental factors.
Gemini API Integration: Uses the Gemini 2.5 Flash model for advanced content analysis and decision-making.
Polygon Normalization: Ensures consistent polygon data with bounding boxes and geo-coordinates for accurate map analysis.

Requirements:

Python: Version 3.8 or higher, Packages: streamlit==1.29.0, google-generativeai==0.5.0, opencv-python==4.8.0,pillow==10.0.0, pdf2image==1.16.0
Poppler: Required for pdf2image to convert PDFs to images. Install Poppler on your system:
Windows: Download from Poppler for Windows and add to PATH.
Linux: sudo apt-get install poppler-utils
MacOS: brew install poppler
Google Gemini API Key: Obtain an API key from Google Cloud and replace YOUR GEMINI KEY in the code with your key.

Installation

Clone the Repository:
git clone <repository-url>
cd ai-powered-city-planner

Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Install Poppler: Follow the instructions above to install Poppler for your operating system.
Verify installation: pdftoppm -v (should display Poppler version).

Configure Gemini API:

Replace GEMINI_API_KEY = "YOUR GEMINI KEY" in the code with your actual Google Gemini API key.
Ensure the API key has access to the gemini-2.5-flash model.

Usage:

Run the Application:
streamlit run app.py

The app will open in your default browser at http://localhost:8501.

Interact with the Tool:

Upload Files:
Upload a proposal document (PDF) and a reference map (PNG/JPG) using the file uploader.
Both files are required for analysis.

Processing:
The app converts the PDF to an image and processes both the document and map using the Gemini API.
It extracts text content, annotations, and map features (e.g., polygons, addresses, landmarks).

Approval Decision:
The tool generates a decision (Approved, Rejected, Requires Review) with detailed reasons based on the document and map data.
Results are displayed with a clear status and a list of reasons.

Expected File Formats:

Proposal Document (PDF):
Should contain project details (e.g., title, location, proposed land use).

Reference Map (PNG/JPG):
Should depict the project area with annotations (e.g., color-coded zones, labels).

Notes:

Gemini API: Requires a valid API key and access to the gemini-2.5-flash model. Update the GEMINI_API_KEY in the code if needed.
File Upload: Both a PDF document and an image map must be uploaded for the tool to proceed.
Image Processing: Images are resized to 512x512 pixels to optimize token usage with the Gemini API.

Troubleshooting:

Gemini API Errors: Verify the API key and ensure the gemini-2.5-flash model is accessible. Check for rate limits (429 errors) and retry after a delay.
PDF Processing Issues: Ensure Poppler is installed and added to your system PATH. Verify the PDF is not corrupted.
Image Processing Errors: Confirm the map image is in PNG, JPG, or JPEG format and not corrupted.
File Upload Issues: Ensure both a PDF and an image are uploaded. Check file sizes and formats.
Dependency Issues: Use Python 3.8+ and verify all required packages are installed correctly.
