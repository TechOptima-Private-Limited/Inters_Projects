# Robinson's QCA Credit Analyzer

## Overview
The **Robinson's QCA Credit Analyzer** is an advanced financial analysis and credit assessment platform designed to evaluate the creditworthiness of companies using both algorithmic methods and qualitative insights. Built with Streamlit, this tool allows users to upload CSV files or manually enter financial data to perform detailed analyses, including growth trends, risk assessments, ratio evaluations, industry comparisons, credit recommendations, and Qualitative Comparative Analysis (QCA). The application integrates algorithmic computations with optional AI-driven synthesis for comprehensive credit memos.

- **Last Updated**: June 19, 2025
- **Current Time**: 06:00 PM IST
- **Developed by**: xAI

## Features
- **Data Input**: Upload CSV files or manually enter financial data for multiple companies and years.
- **Quick Analyses**:
  - Growth Rate Analysis
  - Risk Assessment
  - Ratio Analysis
  - Industry Comparison
  - Credit Recommendation
  - QCA Analysis (requires ≥3 companies)
- **Comprehensive Credit Memo**: Generates a detailed report with executive summary, financial trends, QCA insights, risk evaluation, and credit terms.
- **Visualizations**: Interactive charts for revenue, profit, and key ratios over time.
- **Exportable Reports**: Download analysis results as Markdown files.
- **Algorithmic-Driven**: Uses mathematical formulas (e.g., CAGR), statistical models (e.g., KMeans clustering), and rule-based scoring for deterministic outputs.

## Requirements
- **Python 3.8+**
- **Dependencies**:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `plotly`
  - `scikit-learn`
  - `requests`
  - `pillow`