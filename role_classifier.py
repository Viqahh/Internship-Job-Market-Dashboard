"""Simple rule-based role classification for internship job postings."""

from __future__ import annotations

import re

from skill_extractor import preprocess_text


ROLE_CATEGORIES = [
    "AI / Machine Learning Intern",
    "Data Science Intern",
    "Data Analyst Intern",
    "Business Analyst Intern",
    "Product Analyst Intern",
    "Information Systems Intern",
    "Software Engineering Intern",
    "Cybersecurity / IT Intern",
    "Multimedia / UIUX Intern",
    "Other",
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Return True when a keyword phrase appears as a full phrase.

    This avoids false matches such as `uat` inside `evaluate`.
    """

    for keyword in keywords:
        clean_keyword = preprocess_text(keyword)
        if not clean_keyword:
            continue
        pattern = r"(?<![a-z0-9])" + r"\s+".join(map(re.escape, clean_keyword.split())) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def classify_role(job_title: object, job_description: object = "") -> str:
    """Classify a job posting using title first, then description.

    Rule-based classification is easy to explain in an interview. It is not
    perfect, but it is transparent and useful for a portfolio project.
    """

    title = preprocess_text(job_title)
    description = preprocess_text(job_description)
    combined = f"{title} {description}"

    # Title rules are checked first because they are the strongest signal.
    if _contains_any(
        title,
        [
            "ui ux",
            "ux ui",
            "user experience",
            "user interface",
            "graphic design",
            "multimedia",
            "creative media",
            "motion graphics",
            "video editor",
            "game artist",
            "3d artist",
            "game design",
            "digital media",
        ],
    ):
        return "Multimedia / UIUX Intern"

    if _contains_any(
        title,
        [
            "cybersecurity",
            "cyber security",
            "security analyst",
            "soc analyst",
            "network intern",
            "it support",
            "technical support",
            "system administrator",
        ],
    ):
        return "Cybersecurity / IT Intern"

    if _contains_any(
        title,
        [
            "software engineer",
            "software developer",
            "frontend",
            "front end",
            "backend",
            "back end",
            "full stack",
            "fullstack",
            "web developer",
            "mobile developer",
            "mobile app developer",
            "application developer",
            "qa engineer",
            "quality assurance",
            "devops",
            "cloud engineer",
            "site reliability",
        ],
    ):
        return "Software Engineering Intern"

    if _contains_any(
        title,
        [
            "information system",
            "information systems",
            "system analyst",
            "systems analyst",
            "erp",
            "crm",
            "business systems",
            "functional analyst",
            "implementation consultant",
            "digital transformation",
        ],
    ):
        return "Information Systems Intern"

    if _contains_any(title, ["business analyst", "bi analyst", "business intelligence"]):
        return "Business Analyst Intern"

    if _contains_any(title, ["product analyst", "product data", "growth analyst", "product intern", "product operations analyst"]):
        return "Product Analyst Intern"

    if _contains_any(title, ["data science", "data scientist"]):
        return "Data Science Intern"

    if _contains_any(title, ["data analyst", "analytics intern", "reporting analyst"]):
        return "Data Analyst Intern"

    if _contains_any(
        title,
        [
            "ai intern",
            "ai research",
            "ai engineering",
            "ai model",
            "ai analytics",
            "ai data",
            "junior ai",
            "artificial intelligence",
            "machine learning",
            "ml intern",
            "computer vision",
            "nlp",
        ],
    ):
        return "AI / Machine Learning Intern"

    if _contains_any(title, ["data engineer", "data engineering", "analytics engineer", "database", "cloud data"]):
        return "Software Engineering Intern"

    # Description rules are fallback rules for unclear titles.
    if _contains_any(combined, ["figma", "wireframe", "prototype", "adobe", "photoshop", "illustrator", "premiere pro", "blender", "unity"]):
        return "Multimedia / UIUX Intern"

    if _contains_any(combined, ["vulnerability", "network security", "firewall", "linux", "ticketing", "troubleshooting", "incident response"]):
        return "Cybersecurity / IT Intern"

    if _contains_any(combined, ["react", "javascript", "typescript", "node js", "api", "git", "github", "unit testing", "sprint", "agile"]):
        return "Software Engineering Intern"

    if _contains_any(combined, ["uat", "user acceptance testing", "erp", "crm", "business process", "system requirements", "workflow"]):
        return "Information Systems Intern"

    if _contains_any(combined, ["business requirements", "stakeholder", "business intelligence"]):
        return "Business Analyst Intern"

    if _contains_any(combined, ["product metrics", "a b test", "user behavior", "customer journey"]):
        return "Product Analyst Intern"

    if _contains_any(combined, ["statistical analysis", "predictive modeling", "feature engineering", "model evaluation", "pandas", "scikit learn"]):
        return "Data Science Intern"

    if _contains_any(combined, ["machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch"]):
        return "AI / Machine Learning Intern"

    if _contains_any(combined, ["dashboard", "reporting", "excel", "tableau", "power bi", "data visualization"]):
        return "Data Analyst Intern"

    if _contains_any(combined, ["etl", "backend", "database", "pipeline"]):
        return "Software Engineering Intern"

    return "Other"
