The AI Powered Credit Analyzer is a Streamlit-based web application designed to assist financial analysts, credit officers, and risk managers in assessing the creditworthiness of companies. By integrating Qualitative Comparative Analysis (QCA) and multi-year financial trend analysis, the tool provides comprehensive credit evaluations, interactive visualizations, and exportable reports. It supports both automated data extraction from CSV files and manual data entry, leveraging OptGPT for generating detailed credit memos.

Features:

CSV Data Upload: Automatically detects and extracts financial data for multiple companies from uploaded CSV files.

Manual Data Entry: Allows users to input or edit financial data for specific companies and years.

Trend Analysis: Evaluates multi-year trends in revenue, profit, and key financial ratios (e.g., Current Ratio, Debt-to-Equity, Profit Margin).

Qualitative Comparative Analysis (QCA): Identifies conditions (e.g., high liquidity, low leverage) and combinations that predict strong credit profiles (requires data for at least 3 companies).

Comprehensive Credit Memos: Produces detailed reports with executive summaries, financial analysis, QCA insights, risk evaluations, and credit recommendations.

Analysis Assistant: Provides quick insights for specific analysis types (e.g., Growth Rate, Risk Assessment, QCA Insights).

Requirements:
Python: python==3.8+ Packages:streamlit==1.29.0,pandas==2.0.3,numpy==1.24.3,requests==2.31.0,pillow==10.0.0,plotly==5.15.0

Ollama: Configured to run the OptGPT model (optgpt:7b) locally at http://192.168.1.117:8006/api/generate.

Installation:

Clone the Repository:
git clone <repository-url>
cd ai-powered-credit-analyzer

Set Up a Virtual Environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Set Up Ollama:

Install and configure Ollama on your local machine.
Ensure the OptGPT model (optgpt:7b) is available and accessible at http://192.168.1.117:8006/api/generate.
Update the OLLAMA_URL in the script if your Ollama instance runs on a different host or port.

Run the Application:

streamlit run app.py

The app will open in your default browser at http://localhost:8501.

Interact with the Tool:

Upload a CSV file with financial data. Click Auto-Extract All Company Data to detect companies and extract metrics (e.g., Revenue, Net Profit, Total Debt).View detected companies and data extraction results.

Company Analysis Tab: Select a company to view raw data, generate credit memos, or visualize trends. Generate a Comprehensive Credit Memo for a detailed report with QCA insights (if applicable). Visualize financial trends with interactive charts.

Analysis Assistant Tab: Choose a company and analysis type (e.g., Growth Rate Analysis, QCA Insights) for quick insights.

Manual Data Entry Tab: Enter or edit financial data for a company and specific year. Save data and generate credit memos or visualize trends.

CSV File Format: 
Example structure:

Company,Year,Revenue,Net Profit,Total Debt,Equity,Current Assets,Current Liabilities,EBIT,Interest Expense,Total Assets
ABC Corp,2023,5000,500,1000,2000,1500,1000,600,50,3000
ABC Corp,2022,4500,400,900,1800,1400,900,550,45,2800
XYZ Inc,2023,3000,300,800,1500,1200,700,400,40,2000


Troubleshooting:

CSV Upload Errors: Ensure the CSV is UTF-8 encoded and includes clear column headers for company names and financial metrics.
API Connection Error: Verify that the Ollama service is running and the OLLAMA_URL and MODEL_NAME are correct.
No Companies Detected: Confirm that the CSV includes a column with company names or identifiable data in the first few columns.
Visualization Issues: Ensure at least 2 years of data are available for the selected company.
Dependency Issues: Use Python 3.8+ and verify all required packages are installed.