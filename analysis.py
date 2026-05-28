"""Run the full CSV -> NLP -> SQLite -> outputs pipeline."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from database import DATABASE_PATH, import_csv_to_sqlite
from recommendation import DEFAULT_CURRENT_SKILLS, build_learning_plan, build_recommendations
from role_classifier import classify_role
from skill_extractor import (
    SKILL_KEYWORDS,
    extract_all_skills,
    extract_soft_skills,
    extract_technical_skills,
    extract_tools_mentioned,
    preprocess_text,
    skills_to_string,
    string_to_skills,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_CSV_PATH = DATA_DIR / "raw_job_postings.csv"
CLEANED_CSV_PATH = DATA_DIR / "cleaned_job_postings.csv"
TABLEAU_CSV_PATH = DATA_DIR / "tableau_job_market_cleaned.csv"


REQUIRED_COLUMNS = [
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


def load_raw_data(csv_path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load raw manually collected job postings."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in raw CSV: {missing_columns}")

    return df


def combine_text_fields(row: pd.Series) -> str:
    """Combine fields where skills may appear."""

    return " ".join(
        str(row.get(column, ""))
        for column in [
            "job_title",
            "job_description",
            "required_skills",
            "soft_skills",
            "tools_mentioned",
        ]
    )


def clean_and_enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean descriptions, classify roles, and extract skills/tools."""

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.fillna("")

    cleaned_df["clean_description"] = cleaned_df["job_description"].apply(
        lambda text: preprocess_text(text, remove_stopwords=False)
    )
    cleaned_df["job_category"] = cleaned_df.apply(
        lambda row: classify_role(row["job_title"], row["job_description"]),
        axis=1,
    )

    combined_text = cleaned_df.apply(combine_text_fields, axis=1)
    cleaned_df["extracted_skills"] = combined_text.apply(
        lambda text: skills_to_string(extract_all_skills(text))
    )
    cleaned_df["extracted_technical_skills"] = combined_text.apply(
        lambda text: skills_to_string(extract_technical_skills(text))
    )
    cleaned_df["extracted_tools"] = combined_text.apply(
        lambda text: skills_to_string(extract_tools_mentioned(text))
    )
    cleaned_df["extracted_soft_skills"] = combined_text.apply(
        lambda text: skills_to_string(extract_soft_skills(text))
    )

    return cleaned_df


def explode_semicolon_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Turn a semicolon-separated column into one item per row."""

    values: List[str] = []
    if column not in df.columns:
        return pd.Series(values, dtype="object")

    for item_list in df[column].fillna(""):
        values.extend(string_to_skills(item_list))

    return pd.Series(values, dtype="object")


def count_items(df: pd.DataFrame, column: str, item_name: str) -> pd.DataFrame:
    """Count demand frequency for a semicolon-separated skill/tool column."""

    values = explode_semicolon_column(df, column)
    if values.empty:
        return pd.DataFrame(columns=[item_name, "demand_count", "demand_percent"])

    counts = values.value_counts().reset_index()
    counts.columns = [item_name, "demand_count"]
    counts["demand_percent"] = (counts["demand_count"] / len(df) * 100).round(1)
    return counts


def analyze_skills_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Count skills by job role category."""

    rows = []
    for _, row in df.iterrows():
        for skill in string_to_skills(row.get("extracted_technical_skills", "")):
            rows.append({"job_category": row["job_category"], "skill": skill})

    if not rows:
        return pd.DataFrame(columns=["job_category", "skill", "demand_count"])

    skill_role_df = pd.DataFrame(rows)
    return (
        skill_role_df.groupby(["job_category", "skill"])
        .size()
        .reset_index(name="demand_count")
        .sort_values(["job_category", "demand_count"], ascending=[True, False])
    )


def analyze_skill_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Find skill pairs that appear together in the same posting."""

    pair_counter: Counter[tuple[str, str]] = Counter()

    for value in df["extracted_technical_skills"].fillna(""):
        skills = sorted(set(string_to_skills(value)))
        for index, first_skill in enumerate(skills):
            for second_skill in skills[index + 1 :]:
                pair_counter[(first_skill, second_skill)] += 1

    rows = [
        {"skill_1": pair[0], "skill_2": pair[1], "cooccurrence_count": count}
        for pair, count in pair_counter.most_common(25)
    ]
    return pd.DataFrame(rows)


def analyze_tfidf_terms(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Use TF-IDF to identify important terms across job descriptions."""

    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = df["clean_description"].fillna("").tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )
    matrix = vectorizer.fit_transform(documents)
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame({"term": terms, "tfidf_score": scores})
    return tfidf_df.sort_values("tfidf_score", ascending=False).head(top_n)


def add_tableau_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 columns that make Tableau calculations easier for beginners."""

    tableau_df = df.copy()
    all_known_skills = sorted(
        {
            skill
            for category_skills in SKILL_KEYWORDS.values()
            for skill in category_skills.keys()
        }
    )

    for skill in all_known_skills:
        safe_name = (
            skill.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "")
        )
        tableau_df[f"skill_flag_{safe_name}"] = tableau_df["extracted_skills"].apply(
            lambda value, target=skill: 1 if target in string_to_skills(value) else 0
        )

    return tableau_df


def export_tableau_long_format(df: pd.DataFrame) -> None:
    """Create an optional long-format export for stacked Tableau charts."""

    rows = []
    for _, row in df.iterrows():
        for skill in string_to_skills(row.get("extracted_technical_skills", "")):
            rows.append(
                {
                    "job_id": row["job_id"],
                    "job_title": row["job_title"],
                    "company": row["company"],
                    "location": row["location"],
                    "work_mode": row["work_mode"],
                    "job_category": row["job_category"],
                    "date_posted": row["date_posted"],
                    "source_platform": row["source_platform"],
                    "skill": skill,
                    "skill_group": "Technical/Tool",
                }
            )
        for skill in string_to_skills(row.get("extracted_soft_skills", "")):
            rows.append(
                {
                    "job_id": row["job_id"],
                    "job_title": row["job_title"],
                    "company": row["company"],
                    "location": row["location"],
                    "work_mode": row["work_mode"],
                    "job_category": row["job_category"],
                    "date_posted": row["date_posted"],
                    "source_platform": row["source_platform"],
                    "skill": skill,
                    "skill_group": "Soft Skill",
                }
            )

    if rows:
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tableau_skill_role_long.csv", index=False)


def save_analysis_outputs(df: pd.DataFrame) -> None:
    """Save analysis CSVs for portfolio evidence and Tableau."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    top_skills = count_items(df, "extracted_technical_skills", "skill")
    top_tools = count_items(df, "extracted_tools", "tool")
    top_soft_skills = count_items(df, "extracted_soft_skills", "soft_skill")
    skills_by_role = analyze_skills_by_category(df)
    cooccurrence = analyze_skill_cooccurrence(df)
    recommendations = build_recommendations(df, DEFAULT_CURRENT_SKILLS, top_n=10)

    top_skills.to_csv(OUTPUT_DIR / "top_skills.csv", index=False)
    top_tools.to_csv(OUTPUT_DIR / "top_tools.csv", index=False)
    top_soft_skills.to_csv(OUTPUT_DIR / "top_soft_skills.csv", index=False)
    skills_by_role.to_csv(OUTPUT_DIR / "skills_by_job_category.csv", index=False)
    cooccurrence.to_csv(OUTPUT_DIR / "skill_cooccurrence.csv", index=False)
    recommendations.to_csv(OUTPUT_DIR / "skill_gap_recommendation.csv", index=False)

    if os.getenv("RUN_TFIDF", "0") == "1":
        try:
            analyze_tfidf_terms(df).to_csv(OUTPUT_DIR / "top_tfidf_terms.csv", index=False)
        except ValueError:
            print("Skipped TF-IDF output because the dataset is too small after filtering.")
    else:
        pd.DataFrame(columns=["term", "tfidf_score"]).to_csv(
            OUTPUT_DIR / "top_tfidf_terms.csv", index=False
        )

    add_tableau_flag_columns(df).to_csv(TABLEAU_CSV_PATH, index=False)
    export_tableau_long_format(df)

    learning_plan = build_learning_plan(recommendations)
    with open(OUTPUT_DIR / "learning_plan.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(f"{index}. {item}" for index, item in enumerate(learning_plan, start=1)))


def main() -> None:
    """Run the complete project pipeline."""

    raw_df = load_raw_data()
    cleaned_df = clean_and_enrich_data(raw_df)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEANED_CSV_PATH, index=False)
    save_analysis_outputs(cleaned_df)
    import_csv_to_sqlite(CLEANED_CSV_PATH, DATABASE_PATH, replace=True)

    print("Pipeline completed successfully.")
    print(f"Raw rows: {len(raw_df)}")
    print(f"Cleaned CSV: {CLEANED_CSV_PATH}")
    print(f"SQLite database: {DATABASE_PATH}")
    print(f"Tableau CSV: {TABLEAU_CSV_PATH}")
    print(f"Outputs folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
