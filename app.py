"""Streamlit dashboard for Computer Science internship market intelligence."""

from __future__ import annotations

import re
from pathlib import Path

import altair as alt
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


PAGE_OPTIONS = {
    "Overview": "Quick summary of the internship market.",
    "Detailed Charts": "Interactive charts for skills, tools, roles, and locations.",
    "Skill Gap": "Compare your skills with market demand.",
    "Job Search": "Search and inspect individual job postings.",
}


st.set_page_config(
    page_title="CS Internship Job Market Intelligence",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_styles() -> None:
    """Small UI polish for the native Streamlit sidebar button."""

    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] button {
            width: 178px !important;
            height: 42px !important;
            border-radius: 999px !important;
            background: #2f80ed !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.26) !important;
        }

        [data-testid="collapsedControl"] button svg {
            display: none !important;
        }

        [data-testid="collapsedControl"] button::before {
            content: "🔍 Search & Filters";
            color: white;
            font-weight: 700;
            font-size: 0.92rem;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    return ", ".join(split_values(value))


def count_semicolon_values(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Count semicolon-separated values and add percent of postings."""

    values = []
    if column not in df.columns:
        return pd.DataFrame(columns=[label, "Postings", "Share"])

    for value in df[column].fillna(""):
        values.extend(split_values(value))

    if not values:
        return pd.DataFrame(columns=[label, "Postings", "Share"])

    counts = pd.Series(values).value_counts().reset_index()
    counts.columns = [label, "Postings"]
    counts["Share"] = (counts["Postings"] / max(len(df), 1) * 100).round(1)
    counts["Market Demand"] = counts.apply(
        lambda row: f"{int(row['Postings'])} postings ({row['Share']}%)",
        axis=1,
    )
    return counts


def count_column_values(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    """Count normal categorical columns."""

    counts = df[column].value_counts().reset_index()
    counts.columns = [label, "Postings"]
    counts["Share"] = (counts["Postings"] / max(len(df), 1) * 100).round(1)
    return counts


def make_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    color_column: str | None = None,
    height: int = 420,
) -> alt.Chart:
    """Create a clean interactive horizontal bar chart."""

    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{x_column}:Q", title="Postings"),
            y=alt.Y(f"{y_column}:N", sort="-x", title=None),
            color=alt.Color(f"{color_column}:N", legend=None) if color_column else alt.value("#3f7f93"),
            tooltip=[
                alt.Tooltip(f"{y_column}:N", title=y_column),
                alt.Tooltip(f"{x_column}:Q", title="Postings"),
                alt.Tooltip("Share:Q", title="Share of postings", format=".1f"),
            ],
        )
        .properties(title=title, height=height)
    )
    return chart


def make_donut_chart(data: pd.DataFrame, name_column: str, title: str) -> alt.Chart:
    """Create an interactive donut chart."""

    return (
        alt.Chart(data)
        .mark_arc(innerRadius=65, outerRadius=120)
        .encode(
            theta=alt.Theta("Postings:Q"),
            color=alt.Color(f"{name_column}:N", title=name_column),
            tooltip=[
                alt.Tooltip(f"{name_column}:N", title=name_column),
                alt.Tooltip("Postings:Q"),
                alt.Tooltip("Share:Q", title="Share of postings", format=".1f"),
            ],
        )
        .properties(title=title, height=340)
    )


def build_skill_role_data(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Return long-form skill by role data for stacked charts."""

    rows = []
    for _, row in df.iterrows():
        for skill in split_values(row.get("extracted_technical_skills", "")):
            rows.append({"Role Category": row["job_category"], "Skill": skill})

    if not rows:
        return pd.DataFrame(columns=["Role Category", "Skill", "Postings"])

    long_df = pd.DataFrame(rows)
    top_skills = long_df["Skill"].value_counts().head(top_n).index
    long_df = long_df[long_df["Skill"].isin(top_skills)]
    return long_df.groupby(["Role Category", "Skill"]).size().reset_index(name="Postings")


def build_display_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return readable job table columns."""

    display_table = df[list(JOB_TABLE_COLUMNS.keys())].copy()
    for column in ["extracted_technical_skills", "extracted_tools", "extracted_soft_skills"]:
        display_table[column] = display_table[column].apply(format_skill_list)
    return display_table.rename(columns=JOB_TABLE_COLUMNS)


def choose_page() -> str:
    """Search-style page selector for the sidebar."""

    st.sidebar.title("Search & Filters")
    page_query = st.sidebar.text_input(
        "Search page",
        placeholder="Try: overview, charts, skill, jobs",
    ).strip().lower()

    page_names = list(PAGE_OPTIONS.keys())
    if page_query:
        matches = [
            page
            for page in page_names
            if page_query in page.lower() or page_query in PAGE_OPTIONS[page].lower()
        ]
        page_names = matches or page_names

    selected_page = st.sidebar.radio("Open page", page_names)
    st.sidebar.caption(PAGE_OPTIONS[selected_page])
    return selected_page


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.divider()
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

    return df[
        df["job_category"].isin(selected_categories)
        & df["location"].isin(selected_locations)
        & df["work_mode"].isin(selected_work_modes)
        & df["source_platform"].isin(selected_sources)
    ].copy()


def show_header() -> None:
    st.title("Computer Science Internship Job Market Intelligence Dashboard")
    st.caption("Explore CS internship trends and see which skills are worth learning next.")


def show_kpis(df: pd.DataFrame) -> None:
    total_postings = len(df)
    total_companies = df["company"].nunique() if not df.empty else 0
    top_category = df["job_category"].mode().iloc[0] if not df.empty else "N/A"

    top_skills = count_semicolon_values(df, "extracted_technical_skills", "Skill")
    top_skill = top_skills.iloc[0]["Skill"] if not top_skills.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Postings analyzed", total_postings)
    col2.metric("Companies", total_companies)
    col3.metric("Top role category", top_category.replace(" Intern", ""))
    col4.metric("Most demanded skill", top_skill)


def show_overview_page(df: pd.DataFrame) -> None:
    show_header()
    show_kpis(df)

    st.divider()
    col1, col2 = st.columns([1.2, 1])

    with col1:
        role_counts = count_column_values(df, "job_category", "Role Category")
        st.altair_chart(
            make_bar_chart(
                role_counts,
                "Postings",
                "Role Category",
                "Role category distribution",
                color_column="Role Category",
            ),
            use_container_width=True,
        )

    with col2:
        work_counts = count_column_values(df, "work_mode", "Work Mode")
        st.altair_chart(
            make_donut_chart(work_counts, "Work Mode", "Work mode mix"),
            use_container_width=True,
        )

    st.subheader("Market Takeaways")
    role_counts = count_column_values(df, "job_category", "Role Category")
    work_counts = count_column_values(df, "work_mode", "Work Mode")
    top_skills = count_semicolon_values(df, "extracted_technical_skills", "Skill").head(5)
    top_tools = count_semicolon_values(df, "extracted_tools", "Tool").head(5)

    top_role = role_counts.iloc[0] if not role_counts.empty else None
    top_work_mode = work_counts.iloc[0] if not work_counts.empty else None
    top_skill = top_skills.iloc[0] if not top_skills.empty else None
    top_tool_text = ", ".join(top_tools["Tool"].head(3).tolist()) if not top_tools.empty else "N/A"

    insight_col1, insight_col2 = st.columns(2)
    with insight_col1:
        if top_role is not None:
            st.info(
                f"**Strongest role signal:** {top_role['Role Category']} appears in "
                f"{int(top_role['Postings'])} postings ({top_role['Share']}%)."
            )
        if top_skill is not None:
            st.info(
                f"**Most repeated skill:** {top_skill['Skill']} appears in "
                f"{int(top_skill['Postings'])} postings ({top_skill['Share']}%)."
            )
    with insight_col2:
        if top_work_mode is not None:
            st.info(
                f"**Work mode pattern:** {top_work_mode['Work Mode']} is the most common setup "
                f"({int(top_work_mode['Postings'])} postings)."
            )
        st.info(f"**Tools to notice:** {top_tool_text}.")

    st.success(
        "Use the Job category filter first, then open Skill Gap to get learning priorities for your target internship path."
    )


def show_detailed_charts_page(df: pd.DataFrame) -> None:
    show_header()
    st.subheader("Detailed Interactive Charts")

    top_skills = count_semicolon_values(df, "extracted_technical_skills", "Skill").head(15)
    top_tools = count_semicolon_values(df, "extracted_tools", "Tool").head(15)
    top_soft = count_semicolon_values(df, "extracted_soft_skills", "Soft Skill").head(12)
    location_counts = count_column_values(df, "location", "Location").head(15)
    skill_role_df = build_skill_role_data(df)

    chart_tabs = st.tabs(["Skills", "Tools", "Roles", "Locations", "Skill by Role"])

    with chart_tabs[0]:
        st.altair_chart(
            make_bar_chart(top_skills, "Postings", "Skill", "Top skills requested by employers", "Skill", 520),
            use_container_width=True,
        )
        st.dataframe(top_skills[["Skill", "Postings", "Share", "Market Demand"]], width="stretch", hide_index=True)

    with chart_tabs[1]:
        st.altair_chart(
            make_bar_chart(top_tools, "Postings", "Tool", "Most mentioned tools and platforms", "Tool", 520),
            use_container_width=True,
        )
        st.dataframe(top_tools[["Tool", "Postings", "Share", "Market Demand"]], width="stretch", hide_index=True)

    with chart_tabs[2]:
        role_counts = count_column_values(df, "job_category", "Role Category")
        work_counts = count_column_values(df, "work_mode", "Work Mode")
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(
                make_bar_chart(role_counts, "Postings", "Role Category", "Role categories", "Role Category"),
                use_container_width=True,
            )
        with col2:
            st.altair_chart(make_donut_chart(work_counts, "Work Mode", "Work mode"), use_container_width=True)

    with chart_tabs[3]:
        st.altair_chart(
            make_bar_chart(location_counts, "Postings", "Location", "Location distribution", "Location", 520),
            use_container_width=True,
        )

    with chart_tabs[4]:
        if skill_role_df.empty:
            st.info("No skill by role data available for the selected filters.")
        else:
            stacked_chart = (
                alt.Chart(skill_role_df)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X("Postings:Q", title="Skill mentions"),
                    y=alt.Y("Role Category:N", sort="-x", title=None),
                    color=alt.Color("Skill:N", title="Skill"),
                    tooltip=["Role Category:N", "Skill:N", "Postings:Q"],
                )
                .properties(title="Top skills by role category", height=520)
            )
            st.altair_chart(stacked_chart, use_container_width=True)


def show_recommendations_page(df: pd.DataFrame) -> None:
    show_header()
    st.subheader("Skill Gap Recommendation")
    st.caption("This page follows the filters in the sidebar, so choose a role category first if you want targeted advice.")

    current_skill_text = st.text_input(
        "Your current skills",
        value=", ".join(DEFAULT_CURRENT_SKILLS),
        placeholder="Example: Java, JavaScript, React",
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

    st.subheader("What You Should Learn First")
    for item in learning_plan:
        st.write(item)


def show_job_search_page(df: pd.DataFrame) -> None:
    show_header()
    st.subheader("Search Job Postings")

    search_text = st.text_input(
        "Search by title, company, location, skill, or description",
        placeholder="Example: React, Figma, cybersecurity, data science",
    ).strip().lower()

    table_df = df.copy()
    if search_text:
        search_columns = [
            "job_title",
            "company",
            "job_description",
            "extracted_skills",
            "location",
            "job_category",
        ]
        mask = pd.Series(False, index=table_df.index)
        for column in search_columns:
            mask = mask | table_df[column].fillna("").str.lower().str.contains(search_text, regex=False)
        table_df = table_df[mask]

    st.caption(f"Showing {len(table_df)} matching postings.")
    st.dataframe(build_display_table(table_df), width="stretch", hide_index=True)


def main() -> None:
    apply_custom_styles()
    df = load_dashboard_data()

    if df.empty:
        st.warning("No data found. Run `python analysis.py` first to create the cleaned CSV and SQLite database.")
        return

    selected_page = choose_page()
    filtered_df = filter_dataframe(df)

    if filtered_df.empty:
        st.warning("No postings match the selected filters.")
        return

    if selected_page == "Overview":
        show_overview_page(filtered_df)
    elif selected_page == "Detailed Charts":
        show_detailed_charts_page(filtered_df)
    elif selected_page == "Skill Gap":
        show_recommendations_page(filtered_df)
    elif selected_page == "Job Search":
        show_job_search_page(filtered_df)


if __name__ == "__main__":
    main()
