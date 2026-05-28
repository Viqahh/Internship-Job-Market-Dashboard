"""Convert a real public tech jobs dataset into this project format.

Source dataset:
Hugging Face - Tech Job Postings Dataset (2026)
https://huggingface.co/datasets/Sundaydream/tech-jobs-dataset-2026

The source JSON contains real job titles, companies, locations, workplace
types, employment types, salary ranges, and descriptions. It does not include
posting date, so that field is marked as not provided.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from role_classifier import classify_role
from skill_extractor import (
    extract_soft_skills,
    extract_technical_skills,
    extract_tools_mentioned,
    skills_to_string,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_ROOT / "data" / "source_hf_tech_jobs_2026.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw_job_postings.csv"


TARGET_COLUMNS = [
    "job_id",
    "job_title",
    "company",
    "location",
    "work_mode",
    "job_category",
    "job_description",
    "required_skills",
    "soft_skills",
    "tools_mentioned",
    "internship_duration",
    "date_posted",
    "source_platform",
]


MAX_ROWS = 300
MAX_ROWS_PER_CATEGORY = 120
MIN_DESCRIPTION_LENGTH = 80
EARLY_CAREER_PATTERN = re.compile(
    r"\b(intern|internship|co-op|co op|student|graduate|entry level|new grad|trainee|summer 2026)\b",
    re.IGNORECASE,
)
DURATION_PATTERN = re.compile(
    r"\b(\d{1,2}\s*[- ]?\s*(?:week|weeks|month|months)|summer\s+\d{4}|fall\s+\d{4}|spring\s+\d{4})\b",
    re.IGNORECASE,
)


def clean_text(value: object) -> str:
    """Keep text readable in CSV cells."""

    return " ".join(str(value or "").replace("\n", " ").split())


def is_early_career_posting(row: dict[str, object]) -> bool:
    """Keep internship and early-career postings from the larger tech dataset."""

    employment_type = clean_text(row.get("employment_type")).lower()
    title = clean_text(row.get("title"))
    return employment_type == "internship" or bool(EARLY_CAREER_PATTERN.search(title))


def normalize_work_mode(value: object) -> str:
    """Convert source workplace labels into dashboard-friendly values."""

    text = clean_text(value).lower()
    if text == "remote":
        return "Remote"
    if text == "hybrid":
        return "Hybrid"
    if text in {"on-site", "onsite", "on site"}:
        return "On-site"
    return "Not specified"


def extract_duration(title: str, description: str) -> str:
    """Pull a simple internship duration/term if the posting mentions one."""

    match = DURATION_PATTERN.search(f"{title} {description}")
    if not match:
        return "Not specified"
    return match.group(1).strip().title()


def convert_source_dataset() -> list[dict[str, str]]:
    """Read the public source JSON and return rows in this project's schema."""

    rows: list[dict[str, str]] = []
    category_counts: dict[str, int] = defaultdict(int)

    source_rows = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    for source_row in source_rows:
        if not is_early_career_posting(source_row):
            continue

        title = clean_text(source_row.get("title"))
        description = clean_text(source_row.get("description"))

        if not title or len(description) < MIN_DESCRIPTION_LENGTH:
            continue

        category = classify_role(title, description)
        if category == "Other":
            continue

        if category_counts[category] >= MAX_ROWS_PER_CATEGORY:
            continue

        combined_text = f"{title} {description}"
        technical_skills = extract_technical_skills(combined_text)
        tools = extract_tools_mentioned(combined_text)
        soft_skills = extract_soft_skills(combined_text)

        if not technical_skills and not tools:
            continue

        category_counts[category] += 1
        job_number = len(rows) + 1

        rows.append(
            {
                "job_id": f"REAL{job_number:04d}",
                "job_title": title,
                "company": clean_text(source_row.get("company")) or "Not provided",
                "location": clean_text(source_row.get("location")) or "Not specified",
                "work_mode": normalize_work_mode(source_row.get("workplace_type")),
                "job_category": category,
                "job_description": description,
                "required_skills": skills_to_string(technical_skills),
                "soft_skills": skills_to_string(soft_skills),
                "tools_mentioned": skills_to_string(tools),
                "internship_duration": extract_duration(title, description),
                "date_posted": "Not provided by source",
                "source_platform": "Hugging Face - Tech Job Postings Dataset 2026",
            }
        )

        if len(rows) >= MAX_ROWS:
            break

    return rows


def save_rows(rows: list[dict[str, str]]) -> None:
    """Write converted rows to data/raw_job_postings.csv."""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing source file: {SOURCE_PATH}. Download the real source JSON first."
        )

    rows = convert_source_dataset()
    save_rows(rows)

    print(f"Converted {len(rows)} real job-description rows.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
