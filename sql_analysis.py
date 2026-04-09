"""
sql_analysis.py — Job Market Intelligence Dashboard
Runs SQL-based analytical queries on the jobs database.
These queries demonstrate SQL proficiency for interviews.

Usage:
    python sql_analysis.py
"""

import sqlite3
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)
DB_PATH = Path("data/jobs.db")


def get_connection() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run scraper.py first.")
    return sqlite3.connect(DB_PATH)


# ── All queries are pure SQL — demonstrating SQL skills for interviewers ─────

QUERIES = {

    "top_paying_roles": """
        SELECT
            title,
            ROUND(AVG(salary_avg_lpa), 2)   AS avg_salary_lpa,
            ROUND(MIN(salary_min_lpa), 2)   AS min_salary_lpa,
            ROUND(MAX(salary_max_lpa), 2)   AS max_salary_lpa,
            COUNT(*)                         AS job_count
        FROM jobs
        WHERE salary_avg_lpa > 0
        GROUP BY title
        ORDER BY avg_salary_lpa DESC
    """,

    "top_hiring_companies": """
        SELECT
            company,
            COUNT(*) AS total_postings,
            ROUND(AVG(salary_avg_lpa), 2) AS avg_salary_lpa,
            GROUP_CONCAT(DISTINCT city) AS cities
        FROM jobs
        GROUP BY company
        ORDER BY total_postings DESC
        LIMIT 20
    """,

    "city_salary_analysis": """
        SELECT
            city,
            COUNT(*)                         AS job_count,
            ROUND(AVG(salary_avg_lpa), 2)   AS avg_salary_lpa,
            ROUND(MAX(salary_max_lpa), 2)   AS max_salary_lpa,
            ROUND(100.0 * SUM(is_remote) / COUNT(*), 1) AS remote_pct
        FROM jobs
        WHERE salary_avg_lpa > 0
        GROUP BY city
        HAVING job_count >= 5
        ORDER BY avg_salary_lpa DESC
    """,

    "experience_salary_trend": """
        SELECT
            experience,
            COUNT(*)                       AS job_count,
            ROUND(AVG(salary_avg_lpa), 2) AS avg_salary_lpa,
            ROUND(MIN(salary_min_lpa), 2) AS min_lpa,
            ROUND(MAX(salary_max_lpa), 2) AS max_lpa
        FROM jobs
        WHERE salary_avg_lpa > 0
        GROUP BY experience
        ORDER BY
            CASE experience
                WHEN '0-1 years' THEN 1
                WHEN '1-3 years' THEN 2
                WHEN '3-5 years' THEN 3
                WHEN '5-8 years' THEN 4
                WHEN '8+ years'  THEN 5
            END
    """,

    "remote_vs_onsite": """
        SELECT
            CASE is_remote WHEN 1 THEN 'Remote / WFH' ELSE 'On-site' END AS work_type,
            COUNT(*)                         AS job_count,
            ROUND(AVG(salary_avg_lpa), 2)   AS avg_salary_lpa
        FROM jobs
        GROUP BY is_remote
    """,

    "monthly_posting_trend": """
        SELECT
            SUBSTR(posted_date, 1, 7)        AS month,
            COUNT(*)                          AS postings,
            ROUND(AVG(salary_avg_lpa), 2)    AS avg_salary_lpa
        FROM jobs
        WHERE posted_date IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
        LIMIT 12
    """,

    "role_city_heatmap": """
        SELECT
            title,
            city,
            COUNT(*) AS job_count,
            ROUND(AVG(salary_avg_lpa), 2) AS avg_salary_lpa
        FROM jobs
        WHERE salary_avg_lpa > 0
          AND city NOT IN ('Remote', 'Work From Home')
        GROUP BY title, city
        HAVING job_count >= 2
        ORDER BY job_count DESC
        LIMIT 60
    """,

    "top_companies_for_freshers": """
        SELECT
            company,
            title,
            COUNT(*) AS postings,
            ROUND(AVG(salary_avg_lpa), 2) AS avg_salary_lpa
        FROM jobs
        WHERE experience IN ('0-1 years', '1-3 years')
          AND salary_avg_lpa > 0
        GROUP BY company, title
        ORDER BY avg_salary_lpa DESC
        LIMIT 20
    """,
}


def run_all_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    """Run all analytical queries and return results as DataFrames."""
    conn = get_connection()
    results = {}
    for name, sql in QUERIES.items():
        try:
            results[name] = pd.read_sql(sql, conn)
            log.info(f"Query '{name}' → {len(results[name])} rows")
        except Exception as e:
            log.error(f"Query '{name}' failed: {e}")
            results[name] = pd.DataFrame()
    conn.close()
    return results


def get_skill_frequency(db_path: Path = DB_PATH, top_n: int = 25) -> pd.DataFrame:
    """
    Parse the comma-separated skills column and compute frequency.
    This is done in Python since SQLite lacks array unnesting.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT skills, salary_avg_lpa FROM jobs WHERE skills != ''", conn)
    conn.close()

    skill_records = []
    for _, row in df.iterrows():
        for skill in str(row["skills"]).split(","):
            skill = skill.strip().lower()
            if skill:
                skill_records.append({"skill": skill, "salary_avg_lpa": row["salary_avg_lpa"]})

    skill_df = pd.DataFrame(skill_records)
    freq = (
        skill_df.groupby("skill")
        .agg(
            count=("skill", "count"),
            avg_salary_lpa=("salary_avg_lpa", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    freq["avg_salary_lpa"] = freq["avg_salary_lpa"].round(2)
    return freq


def get_skill_salary_premium(db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Compute salary premium for each skill vs. baseline average.
    Answers: 'Which skill commands the highest salary bump?'
    This is the most recruiter-memorable insight.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT skills, salary_avg_lpa FROM jobs WHERE salary_avg_lpa > 0", conn)
    conn.close()

    baseline = df["salary_avg_lpa"].mean()

    skill_records = []
    for _, row in df.iterrows():
        for skill in str(row["skills"]).split(","):
            skill = skill.strip().lower()
            if skill:
                skill_records.append({"skill": skill, "salary_avg_lpa": row["salary_avg_lpa"]})

    skill_df = pd.DataFrame(skill_records)
    premium = (
        skill_df.groupby("skill")
        .agg(count=("skill", "count"), avg_sal=("salary_avg_lpa", "mean"))
        .reset_index()
        .query("count >= 10")
    )
    premium["salary_premium_pct"] = ((premium["avg_sal"] - baseline) / baseline * 100).round(1)
    premium = premium.sort_values("salary_premium_pct", ascending=False).head(20)
    return premium


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_queries()
    for name, df in results.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(df.to_string(index=False))

    print("\n\n=== SKILL FREQUENCY ===")
    print(get_skill_frequency().to_string(index=False))

    print("\n\n=== SKILL SALARY PREMIUM ===")
    print(get_skill_salary_premium().to_string(index=False))
