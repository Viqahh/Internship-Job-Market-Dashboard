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

The project uses a CSV dataset of internship job postings. Each posting contains details such as job title, company, location, work mode, job description, required skills, tools, and source platform.

The system then:

1. Cleans the job posting data using Python and pandas.
2. Reads the job description text.
3. Extracts skills and tools using simple NLP keyword matching.
4. Classifies each job into a role category.
5. Stores the cleaned data in SQLite.
6. Displays the results in an interactive Streamlit dashboard.
7. Recommends what skills the user should learn next.

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
- Matplotlib
- Tableau-ready CSV export

## How To Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python analysis.py
streamlit run app.py
```

## Main Files

```text
app.py                         Streamlit dashboard
analysis.py                    Data cleaning and analysis pipeline
skill_extractor.py             Skill and tool extraction logic
role_classifier.py             Internship role classification logic
recommendation.py              Skill gap recommendation logic
database.py                    SQLite database functions
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
