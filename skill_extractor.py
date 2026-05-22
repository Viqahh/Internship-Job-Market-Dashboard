"""NLP-style skill extraction for internship job descriptions.

This module keeps the first version intentionally explainable:
- clean the text
- search for known skill phrases
- return semicolon-separated lists that are easy to analyze in pandas, SQLite,
  Streamlit, and Tableau
"""

from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Set


STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "will",
    "you",
    "your",
}


# The keys are display names. The values are phrases that may appear in job ads.
SKILL_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "Programming": {
        "Python": ["python", "python programming"],
        "SQL": ["sql", "structured query language"],
        "Java": ["java", "java programming"],
        "JavaScript": ["javascript", "java script", "js"],
        "TypeScript": ["typescript", "type script"],
        "C++": ["c++", "cpp"],
        "C#": ["c#", "c sharp"],
        "PHP": ["php"],
        "R": ["r", "r programming", "r language"],
    },
    "Software Engineering": {
        "HTML/CSS": ["html", "css", "html css", "web page"],
        "React": ["react", "reactjs", "react js"],
        "Node.js": ["node js", "nodejs", "node"],
        "Express.js": ["express js", "expressjs", "express"],
        "REST API": ["rest api", "restful api", "api integration"],
        "Git": ["git", "version control"],
        "GitHub": ["github"],
        "Docker": ["docker", "container"],
        "Agile": ["agile", "scrum", "sprint"],
        "software testing": ["software testing", "unit testing", "qa testing", "test cases"],
        "OOP": ["oop", "object oriented programming", "object oriented"],
        "Flutter": ["flutter", "dart"],
        "React Native": ["react native"],
    },
    "Data Analysis": {
        "Excel": ["excel", "microsoft excel", "spreadsheet", "spreadsheets"],
        "pandas": ["pandas", "python pandas"],
        "NumPy": ["numpy"],
        "data cleaning": ["data cleaning", "data cleansing", "clean data"],
        "data visualization": ["data visualization", "data visualisation", "visualization", "visualisation"],
        "reporting": ["reporting", "reports", "weekly report", "monthly report"],
        "dashboard": ["dashboard", "dashboards", "dashboarding"],
        "statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    },
    "BI Tools": {
        "Tableau": ["tableau", "tableau public"],
        "Power BI": ["power bi", "powerbi", "pbi"],
        "Looker": ["looker", "looker studio"],
        "Google Data Studio": ["google data studio", "data studio"],
    },
    "Machine Learning / AI": {
        "machine learning": ["machine learning", "ml model", "ml models", "predictive model", "predictive modeling"],
        "deep learning": ["deep learning", "neural network", "neural networks"],
        "NLP": ["nlp", "natural language processing", "text analytics", "text mining"],
        "computer vision": ["computer vision", "image recognition", "object detection"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "TensorFlow": ["tensorflow", "tensor flow"],
        "PyTorch": ["pytorch", "torch"],
    },
    "Database / Cloud": {
        "MySQL": ["mysql"],
        "PostgreSQL": ["postgresql", "postgres"],
        "MongoDB": ["mongodb", "mongo db"],
        "SQLite": ["sqlite", "sqlite3"],
        "Firebase": ["firebase", "firestore"],
        "AWS": ["aws", "amazon web services"],
        "Google Cloud": ["google cloud", "gcp"],
        "Azure": ["azure", "microsoft azure"],
    },
    "Information Systems": {
        "requirements analysis": ["requirements analysis", "business requirements", "system requirements"],
        "system analysis": ["system analysis", "systems analysis", "system analyst"],
        "UAT": ["uat", "user acceptance testing"],
        "ERP": ["erp", "enterprise resource planning", "sap"],
        "CRM": ["crm", "salesforce", "hubspot"],
        "business process": ["business process", "process mapping", "workflow"],
        "IT documentation": ["it documentation", "technical documentation", "system documentation"],
    },
    "Cybersecurity / IT": {
        "Linux": ["linux", "ubuntu"],
        "networking": ["networking", "tcp ip", "dns", "routing"],
        "cybersecurity": ["cybersecurity", "cyber security", "security monitoring"],
        "vulnerability assessment": ["vulnerability assessment", "vulnerability scanning"],
        "incident response": ["incident response", "security incident"],
        "troubleshooting": ["troubleshooting", "technical support", "it support"],
        "firewall": ["firewall", "network firewall"],
    },
    "Multimedia / UIUX": {
        "UI/UX design": ["ui ux", "ux ui", "user experience", "user interface"],
        "Figma": ["figma"],
        "wireframing": ["wireframe", "wireframing"],
        "prototyping": ["prototype", "prototyping"],
        "Adobe Photoshop": ["photoshop", "adobe photoshop"],
        "Adobe Illustrator": ["illustrator", "adobe illustrator"],
        "Adobe Premiere Pro": ["premiere pro", "adobe premiere"],
        "video editing": ["video editing", "video editor"],
        "motion graphics": ["motion graphics", "after effects"],
        "Blender": ["blender"],
        "Unity": ["unity", "unity3d"],
        "game development": ["game development", "game design", "gameplay"],
    },
    "Soft Skills": {
        "communication": ["communication", "communicate", "communicating"],
        "problem solving": ["problem solving", "problem-solving", "solve problems"],
        "teamwork": ["teamwork", "team player", "collaboration", "collaborate"],
        "analytical thinking": ["analytical thinking", "analytical mindset", "critical thinking"],
        "stakeholder management": ["stakeholder management", "stakeholders", "stakeholder"],
        "presentation": ["presentation", "present findings", "present insights"],
        "documentation": ["documentation", "document requirements", "write documentation"],
    },
}


TOOL_NAMES: Set[str] = {
    "Excel",
    "pandas",
    "Tableau",
    "Power BI",
    "Looker",
    "Google Data Studio",
    "NumPy",
    "React",
    "Node.js",
    "Express.js",
    "Git",
    "GitHub",
    "Docker",
    "Flutter",
    "React Native",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "SQLite",
    "Firebase",
    "AWS",
    "Google Cloud",
    "Azure",
    "Linux",
    "Figma",
    "Adobe Photoshop",
    "Adobe Illustrator",
    "Adobe Premiere Pro",
    "Blender",
    "Unity",
}


def preprocess_text(text: object, remove_stopwords: bool = False) -> str:
    """Clean text before NLP analysis.

    Steps:
    1. Convert to lowercase so Python and python match the same way.
    2. Replace punctuation with spaces so "SQL,Python" becomes two words.
    3. Remove extra spaces so matching becomes more reliable.
    4. Optionally remove common stopwords for word-frequency analysis.
    """

    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    punctuation_table = str.maketrans({char: " " for char in string.punctuation})
    text = text.translate(punctuation_table)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        tokens = [word for word in text.split() if word not in STOPWORDS]
        text = " ".join(tokens)

    return text


def tokenize_text(text: object, remove_stopwords: bool = True) -> List[str]:
    """Return a simple list of cleaned words from text."""

    clean_text = preprocess_text(text, remove_stopwords=remove_stopwords)
    return clean_text.split() if clean_text else []


def _phrase_exists(clean_text: str, phrase: str) -> bool:
    """Check whether a single-word or multi-word phrase exists in clean text."""

    clean_phrase = preprocess_text(phrase)
    if not clean_phrase:
        return False

    parts = clean_phrase.split()
    pattern = r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, parts)) + r"(?![a-z0-9])"
    return re.search(pattern, clean_text) is not None


def extract_skills_by_category(text: object) -> Dict[str, List[str]]:
    """Extract skills grouped by category from one job description."""

    clean_text = preprocess_text(text)
    results: Dict[str, List[str]] = {}

    for category, skills in SKILL_KEYWORDS.items():
        found_skills = []
        for display_name, variants in skills.items():
            if any(_phrase_exists(clean_text, variant) for variant in variants):
                found_skills.append(display_name)
        results[category] = sorted(found_skills)

    return results


def flatten_skills(skills_by_category: Dict[str, Iterable[str]]) -> List[str]:
    """Convert grouped skills into one sorted unique list."""

    all_skills = set()
    for skills in skills_by_category.values():
        all_skills.update(skills)
    return sorted(all_skills)


def extract_all_skills(text: object) -> List[str]:
    """Extract every known technical/tool/soft skill from text."""

    return flatten_skills(extract_skills_by_category(text))


def extract_technical_skills(text: object) -> List[str]:
    """Extract technical skills, excluding soft skills."""

    skills_by_category = extract_skills_by_category(text)
    technical = []
    for category, skills in skills_by_category.items():
        if category != "Soft Skills":
            technical.extend(skills)
    return sorted(set(technical))


def extract_soft_skills(text: object) -> List[str]:
    """Extract only soft skills."""

    return extract_skills_by_category(text).get("Soft Skills", [])


def extract_tools_mentioned(text: object) -> List[str]:
    """Extract tools/platforms/libraries from the known skill list."""

    all_skills = extract_all_skills(text)
    return sorted(skill for skill in all_skills if skill in TOOL_NAMES)


def skills_to_string(skills: Iterable[str]) -> str:
    """Store a list as a consistent semicolon-separated string."""

    return "; ".join(sorted(set(skill for skill in skills if skill)))


def string_to_skills(value: object) -> List[str]:
    """Read a semicolon-separated skill string back into a Python list."""

    if value is None:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]
