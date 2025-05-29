# parserb.py
import pdfplumber
import re

def extract_resume_data(file):
    try:
        # Extract text from PDF
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return {
            "name": "Error",
            "email": f"Error reading PDF: {e}",
            "technical_skills": [],
            "soft_skills": []
        }

    # Clean and split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # --- Extract Name ---
    name = "Not found"
    for line in lines[:10]:
        if len(line.split()) in [2, 3] and all(word[0].isupper() for word in line.split()):
            name = line
            break

    # --- Extract Email ---
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else "Not found"

    # --- Define Skills ---
    technical_keywords = {
        "python", "java", "c++", "c", "sql", "mysql", "mongodb", "postgresql", "excel", "power bi",
        "html", "css", "javascript", "typescript", "react", "node", "flask", "django",
        "pandas", "numpy", "matplotlib", "seaborn", "tensorflow", "keras", "scikit-learn"
    }

    soft_keywords = {
        "communication", "teamwork", "leadership", "adaptability", "problem solving",
        "critical thinking", "time management", "creativity", "work ethic", "interpersonal skills"
    }

    found_technical_skills = set()
    found_soft_skills = set()

    for line in lines:
        line_lower = line.lower()
        for tech in technical_keywords:
            if re.search(rf'\b{re.escape(tech)}\b', line_lower):
                found_technical_skills.add(tech)
        for soft in soft_keywords:
            if re.search(rf'\b{re.escape(soft)}\b', line_lower):
                found_soft_skills.add(soft)

    return {
        "name": name,
        "email": email,
        "technical_skills": sorted(found_technical_skills),
        "soft_skills": sorted(found_soft_skills)
    }
