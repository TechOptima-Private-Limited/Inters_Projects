import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from PIL import Image
from datetime import datetime
import plotly.express as px
from sklearn.cluster import KMeans
import warnings
from io import StringIO
import base64
from pathlib import Path
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OptGPT API setup
OLLAMA_URL = "http://192.168.1.117:11434/api/generate"
MODEL_NAME = "optgpt:7b"

# Streamlit page configuration
st.set_page_config(
    layout="wide",
    page_title="Robinson's QCA Credit Analyzer",
    initial_sidebar_state="expanded"
)

# Helper function to encode image for download
def get_image_download_link(img_path, filename):
    try:
        with open(img_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode()
        href = f'<a href="data:image/png;base64,{b64_string}" download="{filename}">Download Logo</a>'
        return href
    except FileNotFoundError:
        return None

# API call to OptGPT with retry mechanism
def call_optgpt(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            with st.spinner("🤖 OptGPT is processing your request..."):
                payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
                res = requests.post(OLLAMA_URL, json=payload, timeout=30)
                res.raise_for_status()
                data = res.json()
                raw_response = data.get("response", "")
                clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
                thought_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL | re.IGNORECASE)
                thought_process = thought_match.group(1).strip() if thought_match else None
                return clean_response, thought_process
        except requests.exceptions.RequestException as e:
            logger.error(f"API attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                return f"❌ API Error: Unable to connect after {max_retries} attempts", None
    return "❌ Unexpected Error", None

# Enhanced column matching with debug logging
def get_column_value(row, target_names, debug=False):
    if isinstance(target_names, str):
        target_names = [target_names]
    for target_name in target_names:
        normalized_target = target_name.replace(" ", "").replace("_", "").lower()
        for col in row.index:
            normalized_col = col.replace(" ", "").replace("_", "").lower()
            if (normalized_target in normalized_col or normalized_col in normalized_target or
                any(keyword in normalized_col for keyword in normalized_target.split())):
                value = row[col]
                matched_column = col
                if pd.isna(value) or value == '' or str(value).lower() in ['nan', 'null', 'none']:
                    if debug:
                        logger.debug(f"No valid data for {target_name} in column '{matched_column}'")
                    return 0
                try:
                    clean_value = str(value).replace(',', '').replace('₹', '').replace('Rs', '').replace('$', '').strip()
                    found_match = float(clean_value)
                    if debug:
                        logger.debug(f"Found {target_name}: {found_match} from column '{matched_column}'")
                    return found_match
                except ValueError:
                    logger.warning(f"Could not convert {value} to float for {target_name}")
                    return 0
    return 0

# Extract conclusion with improved regex
def extract_conclusion(response):
    match = re.search(r"Conclusion:\s*The company has a\s*(Good|Bad|Average)\s*credit profile\.", response, re.IGNORECASE)
    if match:
        status = match.group(1).lower()
        return {
            "good": "🟢 Good Credit Profile",
            "average": "🟡 Average Credit Profile",
            "bad": "🔴 Bad Credit Profile"
        }.get(status, "")
    return ""

class QCAAnalyzer:
    """Qualitative Comparative Analysis with Clustering"""
    
    def __init__(self):
        self.conditions = {}
        self.outcomes = {}
        self.truth_table = None
        self.qca_results = {}
    
    def calibrate_conditions(self, df_companies):
        calibrated_data = {}
        for company, years_data in df_companies.items():
            latest_year = max(years_data.keys())
            data = years_data[latest_year]
            current_ratio = data['current_assets'] / max(data['current_liabilities'], 0.1)
            debt_equity_ratio = data['total_debt'] / max(data['equity'], 0.1)
            profit_margin = (data['net_profit'] / max(data['revenue'], 0.1)) * 100
            roa = (data['net_profit'] / max(data['total_assets'], 0.1)) * 100
            interest_coverage = data['ebit'] / max(data['interest_expense'], 0.1)
            calibrated_data[company] = {
                'HIGH_LIQUIDITY': self._fuzzy_calibrate(current_ratio, 0.8, 1.2, 2.0),
                'LOW_LEVERAGE': self._fuzzy_calibrate(debt_equity_ratio, 2.0, 1.0, 0.3, inverse=True),
                'HIGH_PROFITABILITY': self._fuzzy_calibrate(profit_margin, 0, 5, 15),
                'HIGH_EFFICIENCY': self._fuzzy_calibrate(roa, 0, 3, 10),
                'STRONG_COVERAGE': self._fuzzy_calibrate(interest_coverage, 1, 2.5, 5),
                'REVENUE_GROWTH': self._calculate_growth_membership(years_data, 'revenue'),
                'STABLE_EARNINGS': self._calculate_stability_membership(years_data, 'net_profit')
            }
        return calibrated_data
    
    def _fuzzy_calibrate(self, value, threshold_out, threshold_cross, threshold_in, inverse=False):
        if inverse:
            if value <= threshold_in: return 1.0
            elif value >= threshold_out: return 0.0
            elif value <= threshold_cross: return 0.5 + (threshold_cross - value) / (2 * (threshold_cross - threshold_in))
            return 0.5 - (value - threshold_cross) / (2 * (threshold_out - threshold_cross))
        if value >= threshold_in: return 1.0
        elif value <= threshold_out: return 0.0
        elif value >= threshold_cross: return 0.5 + (value - threshold_cross) / (2 * (threshold_in - threshold_cross))
        return 0.5 - (threshold_cross - value) / (2 * (threshold_cross - threshold_out))
    
    def _calculate_growth_membership(self, years_data, metric):
        if len(years_data) < 2: return 0.5
        years = sorted(years_data.keys())
        values = [years_data[year][metric] for year in years]
        n_years = len(values) - 1
        if values[0] > 0:
            cagr = ((values[-1] / values[0]) ** (1/n_years) - 1) * 100
        else:
            cagr = 0
        return self._fuzzy_calibrate(cagr, -5, 5, 15)
    
    def _calculate_stability_membership(self, years_data, metric):
        if len(years_data) < 2: return 0.5
        values = [years_data[year][metric] for year in sorted(years_data.keys())]
        if np.mean(values) == 0: return 0.0
        cv = np.std(values) / abs(np.mean(values))
        return self._fuzzy_calibrate(cv, 1.0, 0.5, 0.2, inverse=True)
    
    def create_truth_table(self, calibrated_data, outcome_data):
        companies = list(calibrated_data.keys())
        conditions = list(calibrated_data[companies[0]].keys())
        truth_table_data = []
        for company in companies:
            row = {'Company': company}
            for condition in conditions:
                row[condition] = 1 if calibrated_data[company][condition] > 0.5 else 0
            row['GOOD_CREDIT'] = outcome_data.get(company, 0)
            truth_table_data.append(row)
        self.truth_table = pd.DataFrame(truth_table_data)
        return self.truth_table
    
    def analyze_sufficiency(self, outcome='GOOD_CREDIT'):
        if self.truth_table is None: return None
        conditions = [col for col in self.truth_table.columns if col not in ['Company', outcome]]
        results = {}
        for condition in conditions:
            positive_cases = self.truth_table[self.truth_table[condition] == 1]
            consistency = positive_cases[outcome].mean() if len(positive_cases) > 0 else 0
            coverage = len(positive_cases[positive_cases[outcome] == 1]) / max(self.truth_table[outcome].sum(), 1) if len(positive_cases) > 0 else 0
            results[condition] = {'consistency': consistency, 'coverage': coverage, 'cases': len(positive_cases)}
        
        combination_results = {}
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                combo_name = f"{cond1} * {cond2}"
                combo_cases = self.truth_table[(self.truth_table[cond1] == 1) & (self.truth_table[cond2] == 1)]
                consistency = combo_cases[outcome].mean() if len(combo_cases) > 0 else 0
                coverage = len(combo_cases[combo_cases[outcome] == 1]) / max(self.truth_table[outcome].sum(), 1) if len(combo_cases) > 0 else 0
                combination_results[combo_name] = {'consistency': consistency, 'coverage': coverage, 'cases': len(combo_cases)}
        
        # Add clustering for enhanced insights
        condition_values = self.truth_table[conditions].values
        kmeans = KMeans(n_clusters=min(3, len(condition_values)), random_state=42).fit(condition_values)
        self.truth_table['Cluster'] = kmeans.labels_
        
        return {'single_conditions': results, 'combinations': combination_results, 'clusters': self.truth_table.groupby('Cluster').apply(lambda x: x['Company'].tolist())}
    
    def generate_qca_report(self, sufficiency_results):
        report = "# 🔍 QUALITATIVE COMPARATIVE ANALYSIS REPORT\n\n"
        report += "## 📊 Single Condition Analysis\n\n"
        report += "| Condition | Consistency | Coverage | Cases |\n"
        report += "|-----------|-------------|----------|---------|\n"
        single_results = sufficiency_results['single_conditions']
        for condition, metrics in sorted(single_results.items(), key=lambda x: x[1]['consistency'], reverse=True):
            report += f"| {condition} | {metrics['consistency']:.3f} | {metrics['coverage']:.3f} | {metrics['cases']} |\n"
        
        best_single = max(single_results.items(), key=lambda x: x[1]['consistency'])
        report += f"\n**🏆 Best Single Condition:** {best_single[0]} (Consistency: {best_single[1]['consistency']:.3f})\n\n"
        
        report += "## 🔗 Combination Analysis\n\n"
        report += "| Combination | Consistency | Coverage | Cases |\n"
        report += "|-------------|-------------|----------|---------|\n"
        combo_results = sufficiency_results['combinations']
        for combo, metrics in sorted(combo_results.items(), key=lambda x: x[1]['consistency'], reverse=True)[:10]:
            report += f"| {combo} | {metrics['consistency']:.3f} | {metrics['coverage']:.3f} | {metrics['cases']} |\n"
        
        if combo_results:
            best_combo = max(combo_results.items(), key=lambda x: x[1]['consistency'])
            report += f"\n**🏆 Best Combination:** {best_combo[0]} (Consistency: {best_combo[1]['consistency']:.3f})\n\n"
        
        report += "## 📊 Cluster Analysis\n\n"
        clusters = sufficiency_results['clusters']
        for cluster, companies in clusters.items():
            report += f"**Cluster {cluster}:** {', '.join(companies)}\n"
        
        report += "## 💡 QCA Insights\n\n"
        high_consistency = [cond for cond, metrics in single_results.items() if metrics['consistency'] > 0.8]
        if high_consistency:
            report += f"**🎯 Highly Consistent Conditions:** {', '.join(high_consistency)}\n\n"
        high_coverage = [cond for cond, metrics in single_results.items() if metrics['coverage'] > 0.6]
        if high_coverage:
            report += f"**📈 High Coverage Conditions:** {', '.join(high_coverage)}\n\n"
        
        return report

class FinancialHistoryManager:
    """Enhanced class to manage multi-year financial data with algorithmic analysis"""
    
    def __init__(self):
        self.company_data = {}
        self.csv_companies = []
        self.qca_analyzer = QCAAnalyzer()
        self.industry_benchmarks = {
            'current_ratio': 1.5,
            'debt_equity_ratio': 1.0,
            'profit_margin': 10.0
        }
    
    def detect_companies_in_csv(self, df):
        companies = []
        company_columns = [col for col in df.columns if any(keyword in str(col).lower() for keyword in ['company', 'name', 'firm', 'entity', 'client'])]
        if company_columns:
            company_col = company_columns[0]
            unique_companies = df[company_col].dropna().unique()
            for company in unique_companies:
                company_name = str(company).strip()
                if company_name and company_name.lower() not in ['nan', 'none', '', 'null'] and not company_name.replace('.', '').replace('-', '').isdigit():
                    company_name = re.sub(r'\s+', ' ', company_name).split('(')[0].strip()
                    companies.append(company_name)
        else:
            for idx, row in df.iterrows():
                for col in df.columns[:3]:
                    cell_value = str(row[col]).strip()
                    if cell_value and cell_value.lower() not in ['nan', 'none', '', 'null'] and not cell_value.replace('.', '').replace('-', '').isdigit() and len(cell_value) > 2:
                        clean_name = re.sub(r'\s+', ' ', cell_value).split('(')[0].strip()
                        if clean_name not in companies:
                            companies.append(clean_name)
                        break
                if len(companies) >= 10:
                    break
        self.csv_companies = companies
        return companies
    
    def extract_single_company_data(self, df, company_name):
        company_rows = pd.DataFrame()
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['company', 'name', 'firm', 'entity']):
                matches = df[df[col].astype(str).str.contains(company_name, case=False, na=False)]
                if not matches.empty:
                    company_rows = matches
                    break
        if company_rows.empty:
            mask = df.astype(str).apply(lambda x: x.str.contains(company_name, case=False, na=False)).any(axis=1)
            company_rows = df[mask]
        if company_rows.empty:
            return None
        financial_data = {}
        for _, row in company_rows.iterrows():
            year = self._extract_year_from_row(row) or datetime.now().year
            year_str = str(year)
            year_data = {
                'year': year,
                'revenue': get_column_value(row, ['revenue', 'sales', 'turnover', 'total revenue', 'gross revenue']),
                'net_profit': get_column_value(row, ['net profit', 'profit after tax', 'pat', 'net income', 'profit']),
                'ebit': get_column_value(row, ['ebit', 'operating profit', 'earnings before interest and tax', 'operating income']),
                'total_debt': get_column_value(row, ['total debt', 'debt', 'borrowings', 'loans', 'liabilities']),
                'equity': get_column_value(row, ['equity', 'shareholders equity', 'net worth', 'share capital', 'owners equity']),
                'current_assets': get_column_value(row, ['current assets', 'ca', 'liquid assets', 'short term assets']),
                'current_liabilities': get_column_value(row, ['current liabilities', 'cl', 'short term liabilities']),
                'interest_expense': get_column_value(row, ['interest expense', 'interest cost', 'finance cost', 'interest']),
                'total_assets': get_column_value(row, ['total assets', 'assets']),
                'inventory': get_column_value(row, ['inventory', 'stock', 'stocks'])
            }
            if any(year_data[key] > 0 for key in ['revenue', 'total_assets', 'equity']) or year_data['net_profit'] != 0:
                financial_data[year_str] = year_data
        if financial_data:
            self.company_data[company_name] = financial_data
            return financial_data
        return None
    
    def _extract_year_from_row(self, row):
        for col in row.index:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['year', 'fy', 'financial', 'period', 'date']):
                year_value = row[col]
                if pd.notna(year_value):
                    year_match = re.search(r'(20\d{2}|19\d{2})', str(year_value))
                    if year_match:
                        return int(year_match.group(1))
                    try:
                        year_val = float(year_value)
                        if 1900 <= year_val <= 2100:
                            return int(year_val)
                    except:
                        pass
        for value in row.values:
            if pd.notna(value):
                year_match = re.search(r'(20\d{2}|19\d{2})', str(value))
                if year_match:
                    return int(year_match.group(1))
        return None
    
    def get_company_history(self, company_name):
        return self.company_data.get(company_name, {})
    
    def calculate_financial_ratios(self, company_name):
        history = self.get_company_history(company_name)
        if not history:
            return None
        ratios_by_year = {}
        for year, data in history.items():
            ratios = {}
            ratios['current_ratio'] = round(data['current_assets'] / max(data['current_liabilities'], 0.1), 2)
            ratios['debt_equity_ratio'] = round(data['total_debt'] / max(data['equity'], 0.1), 2)
            ratios['profit_margin'] = round((data['net_profit'] / max(data['revenue'], 0.1)) * 100, 2)
            ratios['roa'] = round((data['net_profit'] / max(data['total_debt'] + data['equity'], 0.1)) * 100, 2)
            ratios['interest_coverage'] = round(data['ebit'] / max(data['interest_expense'], 0.1), 2)
            ratios_by_year[year] = ratios
        return ratios_by_year
    
    def generate_trend_analysis(self, company_name):
        history = self.get_company_history(company_name)
        ratios = self.calculate_financial_ratios(company_name)
        if not history or not ratios:
            return "No historical data available for trend analysis."
        years = sorted(history.keys())
        analysis = StringIO()
        analysis.write(f"\n{'='*80}\n")
        analysis.write(f"FINANCIAL TREND ANALYSIS FOR {company_name.upper()}\n")
        analysis.write(f"{'='*80}\n")
        analysis.write(f"\n📊 FINANCIAL PERFORMANCE OVER TIME:\n")
        analysis.write(f"{'-'*80}\n")
        analysis.write(f"{'Metric':<25} | " + " | ".join([f"{year:>10}" for year in years]) + " | Trend\n")
        analysis.write(f"{'-'*80}\n")
        revenues = [history[year]['revenue'] for year in years]
        revenue_trend = self._calculate_trend(revenues)
        analysis.write(f"{'Revenue (₹Cr)':<25} | " + " | ".join([f"{history[year]['revenue']:>10.1f}" for year in years]) + f" | {revenue_trend}\n")
        profits = [history[year]['net_profit'] for year in years]
        profit_trend = self._calculate_trend(profits)
        analysis.write(f"{'Net Profit (₹Cr)':<25} | " + " | ".join([f"{history[year]['net_profit']:>10.1f}" for year in years]) + f" | {profit_trend}\n")
        analysis.write(f"\n📈 FINANCIAL RATIOS TREND:\n")
        analysis.write(f"{'-'*80}\n")
        cr_values = [ratios[year]['current_ratio'] for year in years]
        cr_trend = self._calculate_trend(cr_values)
        analysis.write(f"{'Current Ratio':<25} | " + " | ".join([f"{ratios[year]['current_ratio']:>10.2f}" for year in years]) + f" | {cr_trend}\n")
        de_values = [ratios[year]['debt_equity_ratio'] for year in years]
        de_trend = self._calculate_trend(de_values)
        analysis.write(f"{'Debt/Equity Ratio':<25} | " + " | ".join([f"{ratios[year]['debt_equity_ratio']:>10.2f}" for year in years]) + f" | {de_trend}\n")
        pm_values = [ratios[year]['profit_margin'] for year in years]
        pm_trend = self._calculate_trend(pm_values)
        analysis.write(f"{'Profit Margin (%)':<25} | " + " | ".join([f"{ratios[year]['profit_margin']:>10.2f}" for year in years]) + f" | {pm_trend}\n")
        analysis.write(f"{'='*80}\n")
        return analysis.getvalue()
    
    def _calculate_trend(self, values):
        if len(values) < 2:
            return "→ Insufficient Data"
        start_val = values[0] if values[0] != 0 else 0.01
        end_val = values[-1]
        if end_val > start_val * 1.1:
            pct_change = ((end_val - start_val) / abs(start_val)) * 100
            return f"↗ Growing (+{pct_change:.1f}%)"
        elif end_val < start_val * 0.9:
            pct_change = ((start_val - end_val) / abs(start_val)) * 100
            return f"↘ Declining (-{pct_change:.1f}%)"
        return "→ Stable"
    
    def create_visualizations(self, company_name):
        history = self.get_company_history(company_name)
        ratios = self.calculate_financial_ratios(company_name)
        if not history:
            return None, None
        years = sorted(history.keys())
        financial_data = [{'Year': year, 'Revenue': history[year]['revenue'], 'Net_Profit': history[year]['net_profit'], 'EBIT': history[year]['ebit']} for year in years]
        ratio_data = [{'Year': year, 'Current_Ratio': ratios[year]['current_ratio'], 'Debt_Equity_Ratio': ratios[year]['debt_equity_ratio'], 'Profit_Margin': ratios[year]['profit_margin']} for year in years]
        return financial_data, ratio_data
    
    def perform_qca_analysis(self):
        if len(self.company_data) < 3:
            return "Need at least 3 companies for meaningful QCA analysis"
        calibrated_data = self.qca_analyzer.calibrate_conditions(self.company_data)
        outcome_data = {}
        for company, years_data in self.company_data.items():
            latest_year = max(years_data.keys())
            data = years_data[latest_year]
            score = 0
            if data['current_assets'] / max(data['current_liabilities'], 0.1) > 1.2: score += 1
            if data['total_debt'] / max(data['equity'], 0.1) < 1.0: score += 1
            if (data['net_profit'] / max(data['revenue'], 0.1)) * 100 > 5: score += 1
            if data['ebit'] / max(data['interest_expense'], 0.1) > 2.5: score += 1
            outcome_data[company] = 1 if score >= 3 else 0
        truth_table = self.qca_analyzer.create_truth_table(calibrated_data, outcome_data)
        sufficiency_results = self.qca_analyzer.analyze_sufficiency()
        qca_report = self.qca_analyzer.generate_qca_report(sufficiency_results)
        return {'calibrated_data': calibrated_data, 'truth_table': truth_table, 'sufficiency_results': sufficiency_results, 'qca_report': qca_report}
    
    def perform_growth_analysis(self, company_name):
        history = self.get_company_history(company_name)
        if len(history) < 2:
            return "Insufficient data for growth analysis (need at least 2 years)."
        years = sorted(history.keys())
        revenues = [history[year]['revenue'] for year in years]
        profits = [history[year]['net_profit'] for year in years]
        n_years = len(years) - 1
        revenue_cagr = ((revenues[-1] / revenues[0]) ** (1/n_years) - 1) * 100 if revenues[0] > 0 else 0
        profit_cagr = ((profits[-1] / profits[0]) ** (1/n_years) - 1) * 100 if profits[0] > 0 else 0
        pattern = "Consistent" if np.std([self._calculate_trend([revenues[i], revenues[i+1]]) for i in range(len(revenues)-1)]) < 0.1 else "Volatile"
        return f"Growth Rate Analysis for {company_name}:\n- Revenue CAGR: {revenue_cagr:.1f}%\n- Profit CAGR: {profit_cagr:.1f}%\n- Growth Pattern: {pattern}"
    
    def perform_risk_assessment(self, company_name):
        history = self.get_company_history(company_name)
        if not history:
            return "No data available for risk assessment."
        latest_year = max(history.keys())
        data = history[latest_year]
        ratios = self.calculate_financial_ratios(company_name)[latest_year]
        risks = []
        if ratios['debt_equity_ratio'] > 1.5:
            risks.append("High Debt-to-Equity Ratio (>1.5)")
        if ratios['current_ratio'] < 1.0:
            risks.append("Low Current Ratio (<1.0)")
        if (data['net_profit'] / max(data['revenue'], 0.1)) * 100 < 0:
            risks.append("Negative Profit Margin")
        stability = self._calculate_trend([history[year]['net_profit'] for year in sorted(history.keys())])
        if "Declining" in stability:
            risks.append("Declining Profit Trend")
        return f"Risk Assessment for {company_name}:\n- Risks: {', '.join(risks) if risks else 'None'}\n- Stability: {stability}"
    
    def perform_ratio_analysis(self, company_name):
        ratios = self.calculate_financial_ratios(company_name)
        if not ratios:
            return "No data available for ratio analysis."
        latest_year = max(ratios.keys())
        data = ratios[latest_year]
        analysis = f"Ratio Analysis for {company_name} ({latest_year}):\n"
        analysis += f"- Current Ratio: {data['current_ratio']:.2f} ({'Strong' if data['current_ratio'] > 1.5 else 'Weak'})\n"
        analysis += f"- Debt/Equity Ratio: {data['debt_equity_ratio']:.2f} ({'Healthy' if data['debt_equity_ratio'] < 1.0 else 'High'})\n"
        analysis += f"- Profit Margin: {data['profit_margin']:.2f}% ({'Good' if data['profit_margin'] > 10.0 else 'Low'})\n"
        return analysis
    
    def perform_industry_comparison(self, company_name):
        ratios = self.calculate_financial_ratios(company_name)
        if not ratios:
            return "No data available for industry comparison."
        latest_year = max(ratios.keys())
        data = ratios[latest_year]
        comparison = f"Industry Comparison for {company_name} ({latest_year}):\n"
        comparison += f"- Current Ratio: {data['current_ratio']:.2f} vs Industry {self.industry_benchmarks['current_ratio']:.2f} ({'Above' if data['current_ratio'] > self.industry_benchmarks['current_ratio'] else 'Below'})\n"
        comparison += f"- Debt/Equity Ratio: {data['debt_equity_ratio']:.2f} vs Industry {self.industry_benchmarks['debt_equity_ratio']:.2f} ({'Better' if data['debt_equity_ratio'] < self.industry_benchmarks['debt_equity_ratio'] else 'Worse'})\n"
        comparison += f"- Profit Margin: {data['profit_margin']:.2f}% vs Industry {self.industry_benchmarks['profit_margin']:.2f}% ({'Better' if data['profit_margin'] > self.industry_benchmarks['profit_margin'] else 'Worse'})\n"
        return comparison
    
    def perform_credit_recommendation(self, company_name):
        history = self.get_company_history(company_name)
        if not history:
            return "No data available for credit recommendation."
        latest_year = max(history.keys())
        data = history[latest_year]
        ratios = self.calculate_financial_ratios(company_name)[latest_year]
        score = 0
        if ratios['current_ratio'] > 1.2: score += 1
        if ratios['debt_equity_ratio'] < 1.0: score += 1
        if data['net_profit'] / max(data['revenue'], 0.1) * 100 > 5: score += 1
        if ratios['interest_coverage'] > 2.5: score += 1
        trend = self._calculate_trend([history[year]['net_profit'] for year in sorted(history.keys())])
        if "Growing" in trend: score += 1
        recommendation = "Approve" if score >= 4 else "Conditional" if score >= 2 else "Reject"
        return f"Credit Recommendation for {company_name}:\n- Score: {score}/5\n- Recommendation: {recommendation}\n- Rationale: Based on liquidity, leverage, profitability, coverage, and profit trend"

@st.cache_resource
def get_history_manager():
    return FinancialHistoryManager()

history_manager = get_history_manager()

# Streamlit App Layout
st.markdown("<h1 style='text-align: center;'>🏦 Robinson's QCA Credit Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Advanced Financial Analysis and Credit Assessment Platform</p>", unsafe_allow_html=True)

left, center, right = st.columns([1, 2, 1])

# LEFT PANEL
with left:
    logo_path = Path("image.png")
    if logo_path.exists():
        st.image(str(logo_path), width=200)
    else:
        st.markdown("<h3 style='text-align: center;'>Logo</h3>", unsafe_allow_html=True)
    
    st.subheader("📁 Data Input")
    uploaded_file = st.file_uploader("Upload CSV with financial data", type=["csv"], key="file_uploader")
    
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            st.success("✅ CSV file uploaded successfully!")
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
            with st.expander("📋 Available Columns"):
                st.write(", ".join(df.columns.tolist()))
            
            detected_companies = history_manager.detect_companies_in_csv(df)
            if detected_companies:
                st.markdown(f"🏢 **Detected Companies ({len(detected_companies)}):**")
                for i, company in enumerate(detected_companies[:5], 1):
                    st.write(f"{i}. {company}")
                if len(detected_companies) > 5:
                    st.write(f"... and {len(detected_companies) - 5} more")
                
                if st.button("🚀 Auto-Extract All Company Data", type="primary"):
                    success_count = 0
                    with st.spinner("Extracting financial data..."):
                        for company in detected_companies:
                            extracted_data = history_manager.extract_single_company_data(df, company)
                            if extracted_data:
                                success_count += 1
                                st.success(f"✅ {company}: {len(extracted_data)} years of data")
                            else:
                                st.warning(f"⚠️ {company}: No financial data found")
                    
                    if success_count > 0:
                        st.balloons()
                        st.success(f"🎉 Successfully extracted data for {success_count}/{len(detected_companies)} companies!")
                        st.session_state['csv_extracted'] = True
                        st.rerun()
                    else:
                        st.error("❌ No financial data could be extracted. Please check your CSV format.")
            else:
                st.warning("⚠️ No companies detected in CSV. Ensure your CSV includes:")
                st.markdown("""
                - A column with company names (e.g., 'Company', 'Name')
                - Financial data columns (e.g., Revenue, Profit, Assets)
                - Clear year indicators
                """)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.markdown("**Common issues:**")
            st.markdown("- File encoding problems (try saving as UTF-8)")
            st.markdown("- Malformed CSV structure")
            st.markdown("- Special characters in headers")
    
    st.subheader("🛠 Advanced Features")
    st.markdown("""
    - 📈 Multi-Year Financial Tracking
    - 📊 Interactive Visualizations
    - 🔍 Comprehensive Ratio Analysis
    - 📋 Historical Credit Assessment
    - 🎯 Predictive Risk Modeling
    - 📥 Exportable Reports
    """)

# CENTER PANEL
with center:
    st.subheader("🏦 Company Analysis")
    csv_extracted = st.session_state.get('csv_extracted', False)
    available_companies = list(history_manager.company_data.keys())
    
    if uploaded_file and csv_extracted and available_companies:
        st.success(f"📊 Analysis Ready - {len(available_companies)} companies available")
        selected_company = st.selectbox("Select Company to Analyze", available_companies, key="company_select")
        
        if selected_company:
            company_history = history_manager.get_company_history(selected_company)
            available_years = sorted(company_history.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 **Available Years:** {', '.join(available_years)}")
            with col2:
                st.info(f"📈 **Data Points:** {len(available_years)} years")
            
            with st.expander("📋 View Raw Financial Data"):
                for year in available_years:
                    year_data = company_history[year]
                    st.write(f"**{year}:**")
                    st.json({
                        "Revenue": f"₹{year_data['revenue']:.1f} Cr",
                        "Net Profit": f"₹{year_data['net_profit']:.1f} Cr",
                        "Total Debt": f"₹{year_data['total_debt']:.1f} Cr",
                        "Equity": f"₹{year_data['equity']:.1f} Cr"
                    })
            
            if st.button("🔍 Generate Comprehensive Credit Memo", type="primary"):
                trend_analysis = history_manager.generate_trend_analysis(selected_company)
                qca_section = ""
                if len(history_manager.company_data) >= 3:
                    with st.spinner("Performing QCA analysis..."):
                        qca_results = history_manager.perform_qca_analysis()
                        if isinstance(qca_results, dict):
                            qca_section = f"""
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
{qca_results['qca_report']}
This QCA analysis compares {selected_company} with {len(history_manager.company_data)-1} other companies to identify conditions leading to good credit profiles.
"""
                        else:
                            qca_section = f"""
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
{qca_results}
"""
                else:
                    qca_section = """
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
Insufficient data for QCA (need at least 3 companies). Analysis based on trend data only.
"""
                
                prompt = f"""
You are a senior credit analyst. Generate a comprehensive CREDIT APPROVAL MEMORANDUM for {selected_company}.

HISTORICAL FINANCIAL DATA:
{trend_analysis}

{qca_section}

<think>
1. Analyze multi-year financial trends
2. Assess credit risk based on trends
3. Evaluate ratio stability
4. Incorporate QCA findings to identify key conditions for creditworthiness
5. Determine appropriate credit rating
</think>

Generate a professional credit memo with:
1. Executive Summary
2. Historical Performance Analysis
3. Financial Ratio Assessment
4. QCA Insights
5. Risk Evaluation
6. Credit Recommendation
7. Proposed Terms & Conditions

Conclusion: The company has a [Good/Average/Bad] credit profile.
"""
                
                st.markdown("### 📄 Credit Analysis Report")
                with st.spinner("Generating comprehensive analysis..."):
                    response_text, thought = call_optgpt(prompt)
                    st.markdown(response_text, unsafe_allow_html=True)
                    conclusion = extract_conclusion(response_text)
                    if conclusion:
                        st.markdown(f"### {conclusion}")
                    if thought:
                        with st.expander("🧠 Show Analysis Process"):
                            st.markdown(thought, unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Report",
                    data=response_text,
                    file_name=f"{selected_company}_Credit_Analysis_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
            if st.button("📈 Show Visualizations"):
                financial_data, ratio_data = history_manager.create_visualizations(selected_company)
                if financial_data:
                    st.markdown("### 📊 Financial Performance Charts")
                    fig1 = px.line(financial_data, x='Year', y=['Revenue', 'Net_Profit'],
                                   title=f"{selected_company} - Revenue & Profit Trend",
                                   labels={'value': 'Amount (₹ Cr)', 'variable': 'Metric'},
                                   template="plotly_white")
                    st.plotly_chart(fig1, use_container_width=True)
                    fig2 = px.line(ratio_data, x='Year', y=['Current_Ratio', 'Profit_Margin'],
                                   title=f"{selected_company} - Key Ratios Trend",
                                   labels={'value': 'Ratio/Percentage', 'variable': 'Metric'},
                                   template="plotly_white")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No visualization data available")
    
    else:
        st.info("📝 Upload a CSV file or use manual entry mode below")
        st.subheader("✍️ Manual Financial Data Entry")
        company_name = st.text_input("Company Name", key="manual_company_name")
        
        if company_name:
            existing_data = history_manager.get_company_history(company_name)
            if existing_data:
                st.info(f"📊 Company exists with {len(existing_data)} years of data")
                with st.expander("📋 View Existing Data"):
                    for year, data in sorted(existing_data.items()):
                        st.write(f"**{year}:** Revenue: ₹{data['revenue']} Cr, Profit: ₹{data['net_profit']} Cr")
            
            st.markdown("💡 **Tip:** Enter data for at least 3 companies to enable QCA Analysis.")
            if history_manager.company_data:
                st.write(f"📊 Saved companies: {', '.join(history_manager.company_data.keys())}")
            
            col_year1, col_year2 = st.columns(2)
            with col_year1:
                entry_mode = st.radio("Entry Mode", ["Add New Year", "Edit Existing Year"], key="entry_mode")
            with col_year2:
                if entry_mode == "Add New Year":
                    year = st.number_input("Financial Year", min_value=1990, max_value=2030, value=datetime.now().year, key="new_year")
                else:
                    if existing_data:
                        year = st.selectbox("Select Year to Edit", sorted(existing_data.keys(), reverse=True), key="edit_year")
                    else:
                        st.warning("No existing data to edit")
                        year = datetime.now().year
            
            default_values = existing_data.get(str(year), {}) if entry_mode == "Edit Existing Year" and existing_data else {}
            
            col1, col2, col3 = st.columns(3)
            with col1:
                revenue = st.number_input("Revenue (₹ Cr)", value=default_values.get('revenue', 0.0), key="revenue")
                net_profit = st.number_input("Net Profit (₹ Cr)", value=default_values.get('net_profit', 0.0), key="net_profit")
                ebit = st.number_input("EBIT (₹ Cr)", value=default_values.get('ebit', 0.0), key="ebit")
            with col2:
                total_debt = st.number_input("Total Debt (₹ Cr)", value=default_values.get('total_debt', 0.0), key="total_debt")
                equity = st.number_input("Equity (₹ Cr)", value=default_values.get('equity', 0.0), key="equity")
                interest_expense = st.number_input("Interest Expense (₹ Cr)", value=default_values.get('interest_expense', 0.0), key="interest_expense")
            with col3:
                current_assets = st.number_input("Current Assets (₹ Cr)", value=default_values.get('current_assets', 0.0), key="current_assets")
                current_liabilities = st.number_input("Current Liabilities (₹ Cr)", value=default_values.get('current_liabilities', 0.0), key="current_liabilities")
                total_assets = st.number_input("Total Assets (₹ Cr)", value=default_values.get('total_assets', 0.0), key="total_assets")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("💾 Save Data", key="save_data"):
                    manual_data = {
                        'year': year,
                        'revenue': revenue,
                        'net_profit': net_profit,
                        'ebit': ebit,
                        'total_debt': total_debt,
                        'equity': equity,
                        'interest_expense': interest_expense,
                        'current_assets': current_assets,
                        'current_liabilities': current_liabilities,
                        'total_assets': total_assets,
                        'inventory': 0
                    }
                    if company_name not in history_manager.company_data:
                        history_manager.company_data[company_name] = {}
                    history_manager.company_data[company_name][str(year)] = manual_data
                    st.success(f"✅ Data saved for {company_name} - Year {year}")
                    st.rerun()
            
            with col_btn2:
                if st.button("🔍 Generate Comprehensive Credit Memo", key="analyze_manual"):
                    if company_name in history_manager.company_data:
                        trend_analysis = history_manager.generate_trend_analysis(company_name)
                        qca_section = ""
                        if len(history_manager.company_data) >= 3:
                            with st.spinner("Performing QCA analysis..."):
                                qca_results = history_manager.perform_qca_analysis()
                                if isinstance(qca_results, dict):
                                    qca_section = f"""
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
{qca_results['qca_report']}
This QCA analysis compares {company_name} with {len(history_manager.company_data)-1} other companies to identify conditions leading to good credit profiles.
"""
                                else:
                                    qca_section = f"""
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
{qca_results}
"""
                        else:
                            qca_section = """
QUALITATIVE COMPARATIVE ANALYSIS (QCA) RESULTS:
Insufficient data for QCA (need at least 3 companies). Analysis based on trend data only.
"""
                        
                        prompt = f"""
You are a senior credit analyst. Generate a comprehensive CREDIT APPROVAL MEMORANDUM for {company_name}.

HISTORICAL FINANCIAL DATA:
{trend_analysis}

{qca_section}

<think>
1. Analyze multi-year financial trends
2. Assess credit risk based on trends
3. Evaluate ratio stability
4. Incorporate QCA findings to identify key conditions for creditworthiness
5. Determine appropriate credit rating
</think>

Generate a professional credit memo with:
1. Executive Summary
2. Historical Performance Analysis
3. Financial Ratio Assessment
4. QCA Insights
5. Risk Evaluation
6. Credit Recommendation
7. Proposed Terms & Conditions

Conclusion: The company has a [Good/Average/Bad] credit profile.
"""
                        
                        st.markdown("### 📄 Credit Analysis Report")
                        with st.spinner("Generating comprehensive analysis..."):
                            response_text, thought = call_optgpt(prompt)
                            st.markdown(response_text, unsafe_allow_html=True)
                            conclusion = extract_conclusion(response_text)
                            if conclusion:
                                st.markdown(f"### {conclusion}")
                            if thought:
                                with st.expander("🧠 Show Analysis Process"):
                                    st.markdown(thought, unsafe_allow_html=True)
                        
                        st.download_button(
                            label="📥 Download Report",
                            data=response_text,
                            file_name=f"{company_name}_Credit_Analysis_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown",
                            key="download_manual_credit_memo"
                        )
                    else:
                        st.error("❌ Please save financial data for the company before analyzing.")
            
            # Quick Analysis Options
            st.markdown("### Quick Analysis Options")
            st.markdown("💡 **Note:** For a full credit memo including QCA, use 'Generate Comprehensive Credit Memo' above.")
            if available_companies:
                query_company = st.selectbox("Select Company", available_companies, key="center_query_company")
                query_type = st.selectbox("Analysis Type", [
                    "Growth Rate Analysis",
                    "Risk Assessment",
                    "Ratio Analysis",
                    "Industry Comparison",
                    "Credit Recommendation",
                    "QCA Analysis"
                ], key="center_query_type")
                
                if query_type == "QCA Analysis" and len(history_manager.company_data) < 3:
                    st.warning(f"⚠️ QCA Analysis requires at least 3 companies. Currently saved: {len(history_manager.company_data)} ({', '.join(history_manager.company_data.keys())})")
                
                if st.button("🔍 Perform Quick Analysis", key="center_quick_analysis"):
                    if query_company in history_manager.company_data:
                        if query_type == "QCA Analysis":
                            if len(history_manager.company_data) >= 3:
                                with st.spinner("Performing QCA analysis..."):
                                    qca_results = history_manager.perform_qca_analysis()
                                    if isinstance(qca_results, dict):
                                        st.markdown("### 📊 QCA Analysis Results")
                                        st.markdown(qca_results['qca_report'], unsafe_allow_html=True)
                                        st.download_button(
                                            label="📥 Download QCA Report",
                                            data=qca_results['qca_report'],
                                            file_name=f"{query_company}_QCA_Analysis_{datetime.now().strftime('%Y%m%d')}.md",
                                            mime="text/markdown",
                                            key="download_qca_report"
                                        )
                                    else:
                                        st.error(f"❌ QCA Analysis Error: {qca_results}")
                            else:
                                st.error("❌ Insufficient data for QCA Analysis. Please save data for at least 3 companies.")
                        else:
                            analysis_funcs = {
                                "Growth Rate Analysis": history_manager.perform_growth_analysis,
                                "Risk Assessment": history_manager.perform_risk_assessment,
                                "Ratio Analysis": history_manager.perform_ratio_analysis,
                                "Industry Comparison": history_manager.perform_industry_comparison,
                                "Credit Recommendation": history_manager.perform_credit_recommendation
                            }
                            with st.spinner(f"Performing {query_type}..."):
                                result = analysis_funcs[query_type](query_company)
                                st.markdown(f"### 📄 {query_type} Result")
                                st.markdown(result, unsafe_allow_html=True)
                                st.download_button(
                                    label=f"📥 Download {query_type} Report",
                                    data=result,
                                    file_name=f"{query_company}_{query_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown",
                                    key=f"download_{query_type.lower().replace(' ', '_')}_report"
                                )
                    else:
                        st.error("❌ Selected company has no saved data.")
            
            if company_name in history_manager.company_data and len(history_manager.company_data[company_name]) > 1:
                st.markdown("---")
                st.subheader("📊 Multi-Year Actions")
                col_act1, col_act2 = st.columns(2)
                with col_act1:
                    if st.button("📈 Show Trends", key="show_trends"):
                        financial_data, ratio_data = history_manager.create_visualizations(company_name)
                        if financial_data:
                            fig1 = px.line(financial_data, x='Year', y=['Revenue', 'Net_Profit'],
                                           title=f"{company_name} - Revenue & Profit Trend",
                                           labels={'value': 'Amount (₹ Cr)', 'variable': 'Metric'},
                                           template="plotly_white")
                            st.plotly_chart(fig1, use_container_width=True)
                with col_act2:
                    if st.button("📋 View Summary", key="view_summary"):
                        trend_summary = history_manager.generate_trend_analysis(company_name)
                        with st.expander("📊 Financial Trend Summary", expanded=True):
                            st.text(trend_summary)

# RIGHT PANEL
with right:
    st.subheader("💬 Analysis Assistant")
    st.markdown("### Coming Soon: AI-Driven Insights")
    st.info("Stay tuned for advanced AI-powered analysis features to assist with your credit decisions!")
    
    st.subheader("💡 Analysis Guidelines")
    st.markdown("""
    **Key Credit Indicators to Monitor:**
    - 📈 **Revenue Growth Consistency**: Stable or increasing revenue trends
    - 💰 **Profit Margin Stability**: Consistent profitability
    - ⚖️ **Debt-to-Equity Ratio**: Low leverage indicates financial health
    - 💧 **Current Ratio**: Liquidity above 1.2 is desirable
    - 🔄 **Cash Flow Patterns**: Positive operating cash flows
    
    **Potential Red Flags:**
    - 📉 Declining revenue trends
    - ❌ Negative or volatile profit margins
    - ⚠️ High debt-to-equity ratios (>1.5)
    - 🚨 Poor liquidity (current ratio <1)
    - 🔶 Inconsistent financial performance
    """)