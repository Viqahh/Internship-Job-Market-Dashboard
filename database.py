"""SQLite database utilities for the internship job market project."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "database" / "job_market.db"


JOB_POSTING_COLUMNS = [
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
    "clean_description",
    "extracted_skills",
    "extracted_technical_skills",
    "extracted_tools",
    "extracted_soft_skills",
]


def create_database(db_path: Path = DATABASE_PATH, replace: bool = False) -> None:
    """Create the SQLite database and job_postings table."""

    db_path.parent.mkdir(parents=True, exist_ok=True)

    if replace and db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS job_postings (
                job_id TEXT PRIMARY KEY,
                job_title TEXT,
                company TEXT,
                location TEXT,
                work_mode TEXT,
                job_category TEXT,
                job_description TEXT,
                required_skills TEXT,
                soft_skills TEXT,
                tools_mentioned TEXT,
                internship_duration TEXT,
                date_posted TEXT,
                source_platform TEXT,
                clean_description TEXT,
                extracted_skills TEXT,
                extracted_technical_skills TEXT,
                extracted_tools TEXT,
                extracted_soft_skills TEXT
            )
            """
        )
        conn.commit()


def import_csv_to_sqlite(
    csv_path: Path,
    db_path: Path = DATABASE_PATH,
    replace: bool = True,
) -> None:
    """Import cleaned CSV rows into SQLite."""

    df = pd.read_csv(csv_path)

    for column in JOB_POSTING_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[JOB_POSTING_COLUMNS].fillna("")

    create_database(db_path=db_path, replace=replace)

    placeholders = ", ".join(["?"] * len(JOB_POSTING_COLUMNS))
    column_names = ", ".join(JOB_POSTING_COLUMNS)
    insert_sql = f"INSERT OR REPLACE INTO job_postings ({column_names}) VALUES ({placeholders})"

    with sqlite3.connect(db_path) as conn:
        conn.executemany(insert_sql, df.astype(str).values.tolist())
        conn.commit()


def read_job_postings(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read all job postings from SQLite into a pandas DataFrame."""

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. Run `python analysis.py` first."
        )

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM job_postings", conn)


def run_example_query(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Example SQL query for learning and interview demonstration."""

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT job_category, COUNT(*) AS postings
            FROM job_postings
            GROUP BY job_category
            ORDER BY postings DESC
            """,
            conn,
        )


if __name__ == "__main__":
    create_database(replace=False)
    print(f"SQLite database is ready at: {DATABASE_PATH}")
