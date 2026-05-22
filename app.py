"""Streamlit dashboard for internship job market intelligence."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from database import DATABASE_PATH, read_job_postings
from recommendation import DEFAULT_CURRENT_SKILLS, build_learning_plan, build_recommendations
from skill_extractor import string_to_skills


PROJECT_ROOT = Path(__file__).resolve().parent
CLEANED_CSV_PATH = PROJECT_ROOT / "data" / "cleaned_job_postings.csv"


JOB_TABLE_COLUMNS = {
    "job_id": "Job ID",
    "job_title": "Job Title",
    "company": "Company",
    "location": "Location",
    "work_mode": "Work Mode",
    "job_category": "Role Category",
    "extracted_technical_skills": "Technical Skills Found",
    "extracted_tools": "Tools Found",
    "extracted_soft_skills": "Soft Skills Found",
    "source_platform": "Source Platform",
}


st.set_page_config(
    page_title="CS Internship Job Market Intelligence",
    page_icon="bar_chart",
    layout="wide",
)


def load_dashboard_data() -> pd.DataFrame:
    """Load data from SQLite first, then fall back to cleaned CSV."""

    try:
        return read_job_postings(DATABASE_PATH)
    except Exception:
        if CLEANED_CSV_PATH.exists():
            return pd.read_csv(CLEANED_CSV_PATH)
        return pd.DataFrame()


def split_values(value: object) -> list[str]:
    return string_to_skills(value)


def parse_current_skills(skill_text: str) -> list[str]:
    """Parse comma or 'and' separated skill input from the dashboard."""

    cleaned = re.sub(r"\band\b", ",", skill_text, flags=re.IGNORECASE)
    return [skill.strip() for skill in cleaned.split(",") if skill.strip()]


def format_skill_list(value: object) -> str:
    """Make semicolon-separated skill strings easier to read in tables."""

    skills = split_values(value)
    return ", ".join(skills)


def count_semicolon_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    values = []
    if column not in df.columns:
        return pd.DataFrame(columns=["item", "count"])

    for value in df[column].fillna(""):
        values.extend(split_values(value))

    if not values:
        return pd.DataFrame(columns=["item", "count"])

    counts = pd.Series(values).value_counts().reset_index()
    counts.columns = ["item", "count"]
    return counts


def plot_horizontal_bar(data: pd.DataFrame, label_column: str, value_column: str, title: str, color: str):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plot_df = data.sort_values(value_column, ascending=True)
    ax.barh(plot_df[label_column], plot_df[value_column], color=color)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Number of postings")
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")

        category_options = sorted(df["job_category"].dropna().unique())
        location_options = sorted(df["location"].dropna().unique())
        work_mode_options = sorted(df["work_mode"].dropna().unique())
        source_options = sorted(df["source_platform"].dropna().unique())

        selected_categories = st.multiselect("Job category", category_options)
        selected_locations = st.multiselect("Location", location_options)
        selected_work_modes = st.multiselect("Work mode", work_mode_options)
        selected_sources = st.multiselect("Source platform", source_options)

    selected_categories = selected_categories or category_options
    selected_locations = selected_locations or location_options
    selected_work_modes = selected_work_modes or work_mode_options
    selected_sources = selected_sources or source_options

    filtered_df = df[
        df["job_category"].isin(selected_categories)
        & df["location"].isin(selected_locations)
        & df["work_mode"].isin(selected_work_modes)
        & df["source_platform"].isin(selected_sources)
    ].copy()

    return filtered_df


def show_kpis(df: pd.DataFrame) -> None:
    total_postings = len(df)
    total_companies = df["company"].nunique() if not df.empty else 0
    top_category = df["job_category"].mode().iloc[0] if not df.empty else "N/A"
    top_category_display = top_category.replace(" Intern", "")

    top_skills = count_semicolon_values(df, "extracted_technical_skills")
    top_skill = top_skills.iloc[0]["item"] if not top_skills.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Job postings", total_postings)
    col2.metric("Companies", total_companies)
    col3.metric("Top role", top_category_display)
    col4.metric("Most demanded skill", top_skill)


def show_distribution_charts(df: pd.DataFrame) -> None:
    role_counts = df["job_category"].value_counts().reset_index()
    role_counts.columns = ["job_category", "count"]

    work_counts = df["work_mode"].value_counts().reset_index()
    work_counts.columns = ["work_mode", "count"]

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_horizontal_bar(role_counts, "job_category", "count", "Role Distribution", "#2f6f73"))
    with col2:
        st.pyplot(plot_horizontal_bar(work_counts, "work_mode", "count", "Work Mode Distribution", "#8f5f2a"))


def show_skill_charts(df: pd.DataFrame) -> None:
    top_skills = count_semicolon_values(df, "extracted_technical_skills").head(10)
    top_tools = count_semicolon_values(df, "extracted_tools").head(10)
    top_soft = count_semicolon_values(df, "extracted_soft_skills").head(10)

    col1, col2 = st.columns(2)
    with col1:
        if not top_skills.empty:
            st.pyplot(plot_horizontal_bar(top_skills, "item", "count", "Top 10 Technical Skills", "#3563a8"))
        else:
            st.info("No skill data available for the selected filters.")

    with col2:
        if not top_tools.empty:
            st.pyplot(plot_horizontal_bar(top_tools, "item", "count", "Top 10 Tools", "#6b8f2a"))
        else:
            st.info("No tool data available for the selected filters.")

    if not top_soft.empty:
        st.pyplot(plot_horizontal_bar(top_soft, "item", "count", "Most Common Soft Skills", "#9b4d83"))


def show_skills_by_category(df: pd.DataFrame) -> None:
    rows = []
    for _, row in df.iterrows():
        for skill in split_values(row.get("extracted_technical_skills", "")):
            rows.append({"job_category": row["job_category"], "skill": skill})

    if not rows:
        st.info("No skills by category available for the selected filters.")
        return

    skill_role_df = pd.DataFrame(rows)
    top_skills = skill_role_df["skill"].value_counts().head(12).index.tolist()
    skill_role_df = skill_role_df[skill_role_df["skill"].isin(top_skills)]

    pivot = (
        skill_role_df.groupby(["job_category", "skill"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("Skills by Job Category", fontsize=13, weight="bold")
    ax.set_xlabel("Job category")
    ax.set_ylabel("Number of mentions")
    ax.legend(title="Skill", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)


def show_location_distribution(df: pd.DataFrame) -> None:
    location_counts = df["location"].value_counts().head(12).reset_index()
    location_counts.columns = ["location", "count"]

    if not location_counts.empty:
        st.pyplot(
            plot_horizontal_bar(
                location_counts,
                "location",
                "count",
                "Location Distribution",
                "#c06c40",
            )
        )


def show_recommendations(df: pd.DataFrame) -> None:
    st.subheader("Skill Gap Recommendation")
    st.caption("This section follows the filters you choose in the sidebar, so you can focus on one target role category.")

    current_skill_text = st.text_input(
        "Your current skills",
        value=", ".join(DEFAULT_CURRENT_SKILLS),
    )
    current_skills = parse_current_skills(current_skill_text)

    recommendations = build_recommendations(df, current_skills=current_skills, top_n=10)
    learning_plan = build_learning_plan(recommendations)

    if recommendations.empty:
        st.success("No recommendation available because the filtered dataset is empty.")
        return

    display_recommendations = recommendations.copy()
    display_recommendations["Market Demand"] = display_recommendations.apply(
        lambda row: f"{int(row['demand_count'])} postings ({row['demand_percent']}%)",
        axis=1,
    )
    display_recommendations = display_recommendations.rename(
        columns={
            "rank": "Rank",
            "skill": "Skill to Learn",
            "skill_area": "Skill Type",
            "priority": "Priority",
            "learning_focus": "What to Learn",
            "suggested_action": "Practice Task",
        }
    )

    st.table(
        display_recommendations[
            [
                "Rank",
                "Priority",
                "Skill to Learn",
                "Skill Type",
                "Market Demand",
                "What to Learn",
                "Practice Task",
            ]
        ].set_index("Rank")
    )

    st.markdown("**What You Should Learn First**")
    for item in learning_plan:
        st.write(item)


def show_searchable_table(df: pd.DataFrame) -> None:
    st.subheader("Search Job Postings")

    search_text = st.text_input("Search title, company, description, or skills").strip().lower()
    table_df = df.copy()

    if search_text:
        search_columns = [
            "job_title",
            "company",
            "job_description",
            "extracted_skills",
            "location",
        ]
        mask = False
        for column in search_columns:
            mask = mask | table_df[column].fillna("").str.lower().str.contains(search_text, regex=False)
        table_df = table_df[mask]

    display_table = table_df[list(JOB_TABLE_COLUMNS.keys())].copy()
    for column in ["extracted_technical_skills", "extracted_tools", "extracted_soft_skills"]:
        display_table[column] = display_table[column].apply(format_skill_list)
    display_table = display_table.rename(columns=JOB_TABLE_COLUMNS)

    st.dataframe(
        display_table,
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.title("Computer Science Internship Job Market Intelligence Dashboard")
    st.caption("Explore CS internship trends and see which skills are worth learning next.")

    df = load_dashboard_data()

    if df.empty:
        st.warning("No data found. Run `python analysis.py` first to create the cleaned CSV and SQLite database.")
        return

    filtered_df = filter_dataframe(df)

    if filtered_df.empty:
        st.warning("No postings match the selected filters.")
        return

    show_kpis(filtered_df)
    st.divider()
    show_skill_charts(filtered_df)
    st.divider()
    show_distribution_charts(filtered_df)
    st.divider()
    show_skills_by_category(filtered_df)
    st.divider()
    show_location_distribution(filtered_df)
    st.divider()
    show_recommendations(filtered_df)
    st.divider()
    show_searchable_table(filtered_df)


if __name__ == "__main__":
    main()
