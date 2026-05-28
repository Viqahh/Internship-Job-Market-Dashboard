# Computer Science Internship Job Market Dashboard

## About This Web App

This web app helps Computer Science students understand what skills are most requested in internship job postings.

Instead of searching for one company at a time, this dashboard looks at many internship postings together and answers:

> What skills should I learn next to become more competitive for internships?

The app is useful for students from:

- Information Systems
- Software Engineering
- AI / Machine Learning
- Data Science
- Data Analytics
- Business / Product Analytics
- Cybersecurity / IT
- Multimedia / UIUX

## What The Web App Does

The dashboard analyzes internship job postings and shows:

- Total job postings analyzed
- Most common internship role categories
- Most demanded technical skills
- Most used tools
- Work mode distribution, such as Remote, Hybrid, or On-site
- Location distribution
- Skills required by each role category
- Skill gap recommendation based on the user's current skills
- Searchable job posting table

## How It Works

The project now uses a real public job-posting dataset:

[Tech Job Postings Dataset 2026 on Hugging Face](https://huggingface.co/datasets/Sundaydream/tech-jobs-dataset-2026)

The source dataset contains real tech job titles, companies, locations, workplace type, employment type, salary range, and descriptions. This project filters the data to internship / early-career style postings, then converts it into the dashboard format.

Some fields are not available in the source:

- `date_posted` is saved as `Not provided by source`
- `internship_duration` is extracted from the title/description when possible
- skills/tools are extracted from the job description using NLP keyword matching

The system then:

1. Imports the real source JSON into `data/raw_job_postings.csv`.
2. Cleans the job posting data using Python and pandas.
3. Reads the job description text.
4. Extracts skills and tools using simple NLP keyword matching.
5. Classifies each job into a role category.
6. Stores the cleaned data in SQLite.
7. Displays the results in an interactive Streamlit dashboard.
8. Recommends what skills the user should learn next.

## Skill Gap Recommendation

Users can enter their current skills, for example:

```text
Java, JavaScript, React
```

The app compares those skills with the internship market data and suggests what to learn next.

Example recommendation:

```text
Step 1: reporting
Step 2: dashboard
Step 3: Excel
Step 4: data cleaning
Step 5: Python
```

This helps students focus on skills that appear often in internship postings.

## Tech Stack

- Python
- pandas
- scikit-learn
- SQLite
- Streamlit
- Altair
- Tableau-ready CSV export

## How To Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python real_dataset_importer.py
python analysis.py
streamlit run app.py
```

## Main Files

```text
app.py                         Streamlit dashboard
real_dataset_importer.py       Converts the real public dataset into this project schema
analysis.py                    Data cleaning and analysis pipeline
skill_extractor.py             Skill and tool extraction logic
role_classifier.py             Internship role classification logic
recommendation.py              Skill gap recommendation logic
database.py                    SQLite database functions
data/source_hf_tech_jobs_2026.json  Real source dataset
data/raw_job_postings.csv      Converted raw dataset
data/cleaned_job_postings.csv  Cleaned dataset used by the dashboard
database/job_market.db         SQLite database
```

## Why This Project Is Useful

This project helps students make better internship preparation decisions. It turns job descriptions into simple insights, so students can see which skills are actually requested by employers.

It is also useful as a portfolio project because it shows:

- Data cleaning
- NLP keyword extraction
- Role classification
- SQL database usage
- Dashboard visualization
- Business insight
- Learning recommendation
