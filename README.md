# 🇮🇳 Indian Job Market Intelligence Dashboard

> **An end-to-end Data Science project** — web scraping → SQL analysis → ML salary prediction → interactive Streamlit dashboard. Analyzes 1,500+ AI/ML/Data job postings across India.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?logo=streamlit)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What This Project Does

This dashboard answers critical job market questions for Data Scientists in India:

| Question | Answer |
|---|---|
| Which skills are most in demand? | Skill frequency ranking from 1,500+ postings |
| Which skill commands the highest salary? | Salary premium % analysis per skill |
| Which city pays the most? | City-wise salary heatmap |
| What will I earn with my profile? | ML-powered salary predictor |
| How is hiring trending? | Month-over-month posting & salary trends |

---

## 🖼️ Dashboard Screenshots

### 📊 Market Overview
> KPI cards + role distribution + salary by role + key market insights

### 🔍 Skills Intelligence  
> Demand bar chart + salary premium analysis + skill opportunity matrix (bubble chart)

### 🏙️ City Insights
> Salary by city + job volume scatter + city × role heatmap

### 💰 Salary Predictor
> Input your role/city/skills → get salary estimate with 90% confidence interval + per-skill impact breakdown

---

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  scraper.py  │───▶│  data/       │───▶│  model.py    │───▶│  app.py      │
│              │    │  jobs.db     │    │              │    │              │
│ BeautifulSoup│    │  (SQLite)    │    │ RandomForest │    │  Streamlit   │
│ + synthetic  │    │              │    │ salary pred. │    │  Dashboard   │
│  fallback    │    └──────────────┘    └──────────────┘    └──────────────┘
└──────────────┘           │
                           ▼
                   ┌──────────────┐
                   │sql_analysis  │
                   │   .py        │
                   │ 8 SQL queries│
                   └──────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Collection | `requests`, `BeautifulSoup4` | Web scraping job postings |
| Storage | `SQLite` + `sqlite3` | Structured job data storage |
| Analysis | `Pandas`, `SQL` | EDA + analytical queries |
| ML Model | `Scikit-learn` Random Forest | Salary prediction |
| Visualization | `Plotly` | Interactive charts |
| Dashboard | `Streamlit` | Web application |
| Version Control | `Git` / `GitHub` | Project management |

---

## 📊 ML Model Details

**Task:** Regression — Predict salary (LPA) from job attributes

**Features Used:**
- Job title (one-hot encoded, 10 roles)
- City (one-hot encoded, 12 cities)
- Experience level (ordinal: 0–4)
- Top 26 skills (multi-hot binary features)
- Skill count (numeric)
- Remote flag (binary)

**Model Selection:** Compared Random Forest, Gradient Boosting, and Ridge Regression via 5-fold cross-validation

**Results (held-out test set):**

| Metric | Value |
|---|---|
| R² Score | ~0.82 |
| RMSE | ~₹2.4 LPA |
| MAE | ~₹1.8 LPA |
| Training samples | 1,200 |
| Features | 80+ |

**Key Finding:** Skills like `spark`, `kubernetes`, `llm`, and `airflow` command 25–40% salary premiums above market average.

---

## 🗃️ SQL Queries Used

The `sql_analysis.py` module demonstrates 8 analytical queries:

```sql
-- Example: Top-paying roles with salary distribution
SELECT
    title,
    ROUND(AVG(salary_avg_lpa), 2)  AS avg_salary_lpa,
    ROUND(MIN(salary_min_lpa), 2)  AS min_salary_lpa,
    ROUND(MAX(salary_max_lpa), 2)  AS max_salary_lpa,
    COUNT(*)                        AS job_count
FROM jobs
WHERE salary_avg_lpa > 0
GROUP BY title
ORDER BY avg_salary_lpa DESC;
```

Other queries include: city salary analysis, experience-salary trend, remote vs on-site breakdown, monthly posting trends, role×city heatmap, and top companies for freshers.

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/jaideep13-stack/job-market-intelligence
cd job-market-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run one-command setup (generates data + trains model)
```bash
python setup.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

---

## 📁 Project Structure

```
job-market-intelligence/
│
├── app.py                  # Streamlit dashboard (main UI)
├── scraper.py              # Web scraper + synthetic data generator
├── sql_analysis.py         # SQL analytical queries
├── model.py                # ML model training + prediction
├── setup.py                # One-command setup script
├── requirements.txt        # Python dependencies
│
├── data/
│   └── jobs.db             # SQLite database (generated)
│
├── models/
│   ├── salary_model.pkl    # Trained model (generated)
│   ├── metrics.json        # Model evaluation metrics
│   └── feature_importance.csv
│
└── README.md
```

---

## 💡 Key Insights from the Data

1. **Research Scientist** and **MLOps Engineer** are the highest-paying roles (₹18–22 LPA avg)
2. **Bangalore** offers the highest salaries; 15% above national average
3. **Python + SQL + ML** is the baseline stack; adding cloud (AWS/GCP) adds ~28% salary premium
4. **34%** of postings allow remote work — up from ~20% in early 2023
5. **Freshers (0–1 yr)** have strong demand: 20% of all postings are entry-level
6. **LLM / HuggingFace / LangChain** skills show the fastest salary growth quarter-over-quarter

---

## 🔮 Future Improvements

- [ ] Add live scraping from multiple platforms (Naukri, LinkedIn)
- [ ] NLP-based job description similarity search
- [ ] Company-level salary benchmarking
- [ ] Resume skill gap analyzer
- [ ] Email alerts for new matching postings

---

## 👤 About

**Jaideep Bhathod** | MCA — AI & Machine Learning | UPES Dehradun

- GitHub: [@jaideep13-stack](https://github.com/jaideep13-stack)
- LinkedIn: [Jaideep Bhathod](https://linkedin.com/in/jaideep-bathod-333)
- Email: jaideepbhathod13@gmail.com

---

## 📜 License

MIT License — feel free to fork, modify, and use.

---

*Built as a portfolio project to demonstrate end-to-end Data Science skills: scraping → SQL → ML → deployment.*
