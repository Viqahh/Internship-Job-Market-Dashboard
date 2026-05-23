"""Skill gap recommendation logic for internship preparation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List

import pandas as pd

from skill_extractor import string_to_skills


DEFAULT_CURRENT_SKILLS = [
    "Java",
    "SQL",
    "Python",
    "basic machine learning",
    "basic data analysis",
    "Firebase",
    "Streamlit",
]


SKILL_ALIASES = {
    "basic machine learning": "machine learning",
    "ml": "machine learning",
    "basic data analysis": "basic data analysis",
    "data analysis": "basic data analysis",
    "streamlit": "Streamlit",
    "python": "Python",
    "sql": "SQL",
    "java": "Java",
    "excel": "Excel",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "pandas": "pandas",
    "java script": "JavaScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "react": "React",
    "reactjs": "React",
    "react js": "React",
    "html": "HTML/CSS",
    "css": "HTML/CSS",
    "figma": "Figma",
    "firebase": "Firebase",
}


ACTION_LIBRARY = {
    "Tableau": "Build one Tableau Public dashboard using the exported CSV and write 3 business insights.",
    "Power BI": "Learn the basic chart/filter concepts so you can discuss BI tools even on a MacBook.",
    "Excel": "Practice pivot tables, lookup formulas, and cleaning messy internship datasets.",
    "pandas": "Practice filtering, grouping, missing-value handling, and text cleaning in pandas.",
    "data cleaning": "Create before/after cleaning examples and explain your decisions clearly.",
    "data visualization": "Practice choosing chart types and writing insight captions, not just plotting charts.",
    "reporting": "Write short weekly-style summaries with findings, impact, and next steps.",
    "dashboard": "Improve dashboard storytelling: KPI row, filters, chart hierarchy, and insight section.",
    "SQL": "Strengthen joins, GROUP BY, HAVING, subqueries, and window functions.",
    "Python": "Keep building small analytics scripts that turn raw data into business-ready outputs.",
    "machine learning": "Review train/test split, evaluation metrics, and simple classification/regression examples.",
    "JavaScript": "Build small interactive web features and practice DOM/API basics.",
    "TypeScript": "Learn typed JavaScript basics and apply it to safer frontend code.",
    "React": "Build a small component-based interface with state, props, and form input.",
    "Node.js": "Create a simple backend API and connect it to a frontend or database.",
    "REST API": "Practice sending, receiving, and documenting API requests.",
    "Git": "Practice branching, commits, pull requests, and clean project history.",
    "GitHub": "Publish projects with clear README files and organized commits.",
    "Docker": "Learn basic containers and how to run an app consistently.",
    "HTML/CSS": "Practice responsive layouts, clean spacing, and accessible UI structure.",
    "UI/UX design": "Practice user flows, wireframes, and usability-focused interface decisions.",
    "Figma": "Create wireframes and clickable prototypes for app ideas.",
    "Adobe Photoshop": "Practice image editing and design asset preparation.",
    "Adobe Illustrator": "Practice vector assets, icons, and simple branding elements.",
    "Adobe Premiere Pro": "Practice editing short demo videos for portfolio work.",
    "Unity": "Build a small interactive scene or game prototype.",
    "Linux": "Practice terminal commands, permissions, logs, and troubleshooting.",
    "cybersecurity": "Learn basic security concepts, common vulnerabilities, and monitoring workflows.",
    "requirements analysis": "Practice turning user needs into clear system requirements.",
    "system analysis": "Practice mapping system workflows, actors, data, and process rules.",
    "UAT": "Practice writing user acceptance test cases and checking expected behavior.",
    "communication": "Practice explaining technical analysis in simple business language.",
    "stakeholder management": "Practice translating business questions into metrics and dashboard requirements.",
    "presentation": "Prepare a 1-minute project walkthrough and a 3-slide insight summary.",
}


LEARNING_FOCUS_LIBRARY = {
    "dashboard": "Learn how to design KPI cards, filters, chart hierarchy, and an insight section.",
    "reporting": "Learn how to turn analysis results into short business summaries.",
    "data cleaning": "Learn how to handle missing values, duplicates, inconsistent text, and messy CSV files.",
    "Tableau": "Learn how to build a Tableau Public dashboard from a cleaned CSV.",
    "data visualization": "Learn how to choose the right chart and explain the message clearly.",
    "SQL": "Learn joins, GROUP BY, aggregation, subqueries, and window functions.",
    "Excel": "Learn pivot tables, lookup formulas, sorting, filtering, and quick business reports.",
    "Power BI": "Learn the basic BI dashboard concepts so you can discuss it even if you use Tableau on Mac.",
    "pandas": "Learn filtering, grouping, cleaning, and transforming data in Python.",
    "Python": "Learn how to automate data cleaning, analysis, and chart generation.",
    "JavaScript": "Learn web interactivity, API calls, and frontend logic.",
    "React": "Learn components, props, state, events, and form handling.",
    "Node.js": "Learn how backend routes, APIs, and simple server logic work.",
    "REST API": "Learn how applications send and receive data between systems.",
    "Git": "Learn version control, branching, and clean commits.",
    "GitHub": "Learn how to present portfolio projects professionally online.",
    "HTML/CSS": "Learn responsive layouts and clean frontend structure.",
    "UI/UX design": "Learn user flows, wireframes, accessibility, and usability basics.",
    "Figma": "Learn wireframing, prototyping, and interface design handoff.",
    "Linux": "Learn terminal basics, file permissions, and troubleshooting commands.",
    "cybersecurity": "Learn basic security risks, vulnerability concepts, and monitoring.",
    "requirements analysis": "Learn how to gather and write clear system requirements.",
    "system analysis": "Learn how to map business processes into system workflows.",
    "communication": "Learn how to explain insights in simple business language.",
    "presentation": "Learn how to present findings, recommendations, and next steps.",
    "stakeholder management": "Learn how to translate business questions into metrics and dashboard requirements.",
}


def normalize_current_skills(current_skills: Iterable[str]) -> set[str]:
    """Normalize user-entered skills so matching is more forgiving."""

    normalized = set()
    for skill in current_skills:
        clean_skill = str(skill).strip()
        if not clean_skill:
            continue
        normalized.add(SKILL_ALIASES.get(clean_skill.lower(), clean_skill))
    return normalized


def count_market_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Count demand frequency across extracted skills, tools, and soft skills.

    Each skill is counted at most once per job posting. This prevents tools such
    as Excel or Tableau from being counted twice when they appear in both the
    technical-skill column and the tool column.
    """

    counter: Counter[str] = Counter()
    skill_area = {}

    for _, row in df.iterrows():
        posting_skills = set()
        for column, area in [
            ("extracted_technical_skills", "Technical"),
            ("extracted_tools", "Tool"),
            ("extracted_soft_skills", "Soft Skill"),
        ]:
            if column in df.columns:
                for skill in string_to_skills(row.get(column, "")):
                    posting_skills.add(skill)
                    skill_area.setdefault(skill, area)
        counter.update(posting_skills)

    total_postings = max(len(df), 1)
    rows = []
    for skill, count in counter.most_common():
        rows.append(
            {
                "skill": skill,
                "skill_area": skill_area.get(skill, "Skill"),
                "demand_count": count,
                "demand_percent": round((count / total_postings) * 100, 1),
            }
        )

    return pd.DataFrame(rows)


def build_recommendations(
    df: pd.DataFrame,
    current_skills: Iterable[str] = DEFAULT_CURRENT_SKILLS,
    top_n: int = 10,
) -> pd.DataFrame:
    """Recommend high-demand skills that are missing from the current skill set."""

    demand_df = count_market_skills(df)
    if demand_df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "skill",
                "demand_count",
                "demand_percent",
                "priority",
                "why_it_matters",
                "suggested_action",
            ]
        )

    owned_skills = normalize_current_skills(current_skills)
    missing_df = demand_df[~demand_df["skill"].isin(owned_skills)].copy()

    priority_boost = {
        "dashboard": 18,
        "data cleaning": 16,
        "data visualization": 15,
        "Tableau": 25,
        "Power BI": 20,
        "pandas": 22,
        "JavaScript": 18,
        "TypeScript": 14,
        "React": 18,
        "Node.js": 16,
        "REST API": 14,
        "Git": 14,
        "GitHub": 12,
        "HTML/CSS": 12,
        "Figma": 16,
        "UI/UX design": 15,
        "Linux": 10,
        "cybersecurity": 10,
        "requirements analysis": 10,
        "system analysis": 10,
        "Excel": 5,
        "reporting": 8,
        "communication": 3,
        "presentation": 3,
        "stakeholder management": 3,
    }

    area_weight = {
        "Tool": 1.45,
        "Technical": 1.25,
        "Soft Skill": 0.25,
        "Skill": 1.0,
    }

    missing_df["priority_score"] = missing_df.apply(
        lambda row: (row["demand_count"] * area_weight.get(row.get("skill_area", "Skill"), 1.0))
        + priority_boost.get(row["skill"], 0),
        axis=1,
    )
    missing_df = missing_df.sort_values(
        ["priority_score", "demand_count", "skill"], ascending=[False, False, True]
    ).head(top_n)

    rows: List[dict] = []
    for rank, (_, row) in enumerate(missing_df.iterrows(), start=1):
        skill = row["skill"]
        rows.append(
            {
                "rank": rank,
                "skill": skill,
                "skill_area": row.get("skill_area", "Skill"),
                "demand_count": int(row["demand_count"]),
                "demand_percent": row["demand_percent"],
                "priority": "High" if rank <= 5 else "Medium",
                "why_it_matters": f"Appears in {row['demand_percent']}% of postings in this dataset.",
                "learning_focus": LEARNING_FOCUS_LIBRARY.get(
                    skill,
                    f"Learn the basics of {skill} and show it in a small portfolio task.",
                ),
                "suggested_action": ACTION_LIBRARY.get(
                    skill,
                    f"Build a small project task that uses {skill} and explain the result clearly.",
                ),
            }
        )

    return pd.DataFrame(rows)


def build_learning_plan(recommendations_df: pd.DataFrame) -> List[str]:
    """Create a beginner-friendly learning order from recommendation rows.

    The order follows the actual recommendation table, so when the user edits
    their current skills, the learning order changes too.
    """

    if recommendations_df.empty:
        return []

    plan = []
    for step, (_, row) in enumerate(recommendations_df.head(5).iterrows(), start=1):
        skill = row["skill"]
        focus = row.get("learning_focus", ACTION_LIBRARY.get(skill, f"Practice {skill}."))
        plan.append(
            f"Step {step}: {skill} - {focus} Market demand: {row['demand_percent']}% of postings."
        )

    return plan
