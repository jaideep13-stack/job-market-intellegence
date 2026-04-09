import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
import random
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ── Page config
st.set_page_config(
    page_title="Indian Job Market Intelligence",
    page_icon="IN",
    layout="wide",
)

# ── Constants
DB_PATH    = Path("data/jobs.db")
MODEL_PATH = Path("models/salary_model.pkl")

ROLES = [
    "Data Scientist", "ML Engineer", "Data Analyst",
    "Data Engineer", "AI Engineer", "NLP Engineer"
]

CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad",
    "Pune", "Chennai", "Remote"
]

COMPANIES = [
    "Flipkart", "Razorpay", "Swiggy", "Zomato", "CRED",
    "Groww", "Meesho", "PhonePe", "Infosys", "TCS",
    "Wipro", "Amazon India", "Google India", "Microsoft India"
]

SKILLS_POOL = [
    "python", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "spark", "aws", "tableau",
    "power bi", "nlp", "docker", "git", "pandas", "numpy"
]

EXPERIENCE_LEVELS = [
    "0-1 years", "1-3 years", "3-5 years", "5-8 years", "8+ years"
]

SALARY_MAP = {
    "Data Scientist" : (8,  25),
    "ML Engineer"    : (10, 30),
    "Data Analyst"   : (4,  15),
    "Data Engineer"  : (8,  28),
    "AI Engineer"    : (12, 35),
    "NLP Engineer"   : (10, 30),
}


# ── Generate data if not exists
def generate_data():
    random.seed(42)
    np.random.seed(42)
    records = []
    for i in range(1500):
        role      = random.choice(ROLES)
        city      = random.choice(CITIES)
        company   = random.choice(COMPANIES)
        exp       = random.choice(EXPERIENCE_LEVELS)
        skills    = ", ".join(random.sample(SKILLS_POOL, random.randint(3, 6)))
        is_remote = int(city == "Remote")
        s_min, s_max = SALARY_MAP[role]
        salary    = round(random.uniform(s_min, s_max), 1)
        days_ago  = random.randint(0, 90)
        posted    = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        records.append({
            "job_id"      : f"JOB{1000 + i}",
            "title"       : role,
            "company"     : company,
            "city"        : city,
            "experience"  : exp,
            "salary_lpa"  : salary,
            "salary_avg_lpa": salary,
            "skills"      : skills,
            "is_remote"   : is_remote,
            "posted_date" : posted,
        })
    return pd.DataFrame(records)


def ensure_data():
    os.makedirs("data", exist_ok=True)
    if not DB_PATH.exists():
        df = generate_data()
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("jobs", conn, if_exists="replace", index=False)
        conn.close()


def ensure_model():
    os.makedirs("models", exist_ok=True)
    if not MODEL_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM jobs", conn)
        conn.close()
        df["title_enc"] = df["title"].astype("category").cat.codes
        df["city_enc"]  = df["city"].astype("category").cat.codes
        df["exp_enc"]   = df["experience"].map({
            "0-1 years": 0, "1-3 years": 1, "3-5 years": 2,
            "5-8 years": 3, "8+ years": 4
        })
        df["skill_count"] = df["skills"].apply(lambda x: len(x.split(",")))
        X = df[["title_enc", "city_enc", "exp_enc", "is_remote", "skill_count"]].fillna(0)
        y = df["salary_lpa"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)


# ── Load data
@st.cache_data
def load_data():
    ensure_data()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df["month"] = df["posted_date"].dt.to_period("M").astype(str)
    return df


@st.cache_resource
def load_model():
    ensure_data()
    ensure_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Load everything
df    = load_data()
model = load_model()

# ── Sidebar
st.sidebar.title("Job Market Intel")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Market Overview",
    "Skills Analysis",
    "City Insights",
    "Salary Predictor",
    "Raw Data"
])

st.sidebar.markdown("---")
role_filter = st.sidebar.multiselect("Filter by Role",  sorted(df["title"].unique()))
city_filter = st.sidebar.multiselect("Filter by City",  sorted(df["city"].unique()))

# Apply filters
dff = df.copy()
if role_filter:
    dff = dff[dff["title"].isin(role_filter)]
if city_filter:
    dff = dff[dff["city"].isin(city_filter)]


# ══════════════════════════════════════════════
# PAGE 1 - MARKET OVERVIEW
# ══════════════════════════════════════════════
if page == "Market Overview":
    st.title("Indian Job Market Intelligence Dashboard")
    st.markdown("Analyzing 1,500+ AI/ML/Data job postings across India")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs",    f"{len(dff):,}")
    col2.metric("Avg Salary",    f"Rs {dff['salary_lpa'].mean():.1f} LPA")
    col3.metric("Remote Jobs",   f"{dff['is_remote'].mean()*100:.1f}%")
    col4.metric("Top City",      dff["city"].value_counts().idxmax())

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Jobs by Role")
        role_counts = dff["title"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]
        fig = px.bar(role_counts, x="Count", y="Role", orientation="h",
                     color="Count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Average Salary by Role (LPA)")
        role_sal = dff.groupby("title")["salary_lpa"].mean().reset_index().sort_values("salary_lpa")
        fig2 = px.bar(role_sal, x="salary_lpa", y="title", orientation="h",
                      color="salary_lpa", color_continuous_scale="Greens")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Salary distribution
    st.subheader("Salary Distribution")
    fig3 = px.histogram(dff, x="salary_lpa", nbins=30, color_discrete_sequence=["steelblue"])
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 - SKILLS ANALYSIS
# ══════════════════════════════════════════════
elif page == "Skills Analysis":
    st.title("Skills Intelligence")
    st.markdown("Which skills are most demanded and which command highest salary?")
    st.markdown("---")

    # Count skills
    skill_records = []
    for _, row in dff.iterrows():
        for skill in str(row["skills"]).split(","):
            skill_records.append({
                "skill": skill.strip().lower(),
                "salary": row["salary_lpa"]
            })

    skill_df   = pd.DataFrame(skill_records)
    skill_freq = skill_df.groupby("skill").agg(
        count=("skill", "count"),
        avg_salary=("salary", "mean")
    ).reset_index().sort_values("count", ascending=False).head(15)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 15 In-Demand Skills")
        fig = px.bar(skill_freq, x="count", y="skill", orientation="h",
                     color="count", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Avg Salary by Skill (LPA)")
        skill_sal = skill_freq.sort_values("avg_salary", ascending=True)
        fig2 = px.bar(skill_sal, x="avg_salary", y="skill", orientation="h",
                      color="avg_salary", color_continuous_scale="Greens")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=450)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 - CITY INSIGHTS
# ══════════════════════════════════════════════
elif page == "City Insights":
    st.title("City-wise Market Analysis")
    st.markdown("---")

    city_stats = dff.groupby("city").agg(
        job_count=("job_id", "count"),
        avg_salary=("salary_lpa", "mean"),
        remote_pct=("is_remote", "mean")
    ).reset_index()
    city_stats["avg_salary"] = city_stats["avg_salary"].round(2)
    city_stats["remote_pct"] = (city_stats["remote_pct"] * 100).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Avg Salary by City")
        fig = px.bar(city_stats.sort_values("avg_salary"), x="avg_salary", y="city",
                     orientation="h", color="avg_salary", color_continuous_scale="Purples")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Job Count by City")
        fig2 = px.bar(city_stats.sort_values("job_count"), x="job_count", y="city",
                      orientation="h", color="job_count", color_continuous_scale="Blues")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Remote vs Onsite")
    remote_counts = dff["is_remote"].value_counts().reset_index()
    remote_counts.columns = ["type", "count"]
    remote_counts["type"] = remote_counts["type"].map({1: "Remote", 0: "Onsite"})
    fig3 = px.pie(remote_counts, names="type", values="count",
                  color_discrete_sequence=["#42A5F5", "#66BB6A"])
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 - SALARY PREDICTOR
# ══════════════════════════════════════════════
elif page == "Salary Predictor":
    st.title("AI Salary Predictor")
    st.markdown("Enter your profile and get market salary estimate")
    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        title      = st.selectbox("Job Role",        ROLES)
        city       = st.selectbox("City",            CITIES)
        experience = st.selectbox("Experience",      EXPERIENCE_LEVELS)
        skill_count = st.slider("Number of Skills", 1, 15, 5)
        is_remote  = st.checkbox("Remote Role")

    with col_b:
        if st.button("Predict Salary", use_container_width=True):
            title_codes = {r: i for i, r in enumerate(sorted(df["title"].unique()))}
            city_codes  = {c: i for i, c in enumerate(sorted(df["city"].unique()))}
            exp_codes   = {"0-1 years":0,"1-3 years":1,"3-5 years":2,"5-8 years":3,"8+ years":4}

            sample = pd.DataFrame([{
                "title_enc"  : title_codes.get(title, 0),
                "city_enc"   : city_codes.get(city, 0),
                "exp_enc"    : exp_codes.get(experience, 1),
                "is_remote"  : int(is_remote),
                "skill_count": skill_count,
            }])

            pred = model.predict(sample)[0]
            st.success(f"Predicted Salary: Rs {pred:.2f} LPA")
            st.info(f"Role: {title} | City: {city} | Exp: {experience}")

            # salary range context
            s_min, s_max = SALARY_MAP.get(title, (5, 20))
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Predicted LPA"},
                gauge={
                    "axis": {"range": [s_min, s_max]},
                    "bar": {"color": "#60A5FA"},
                    "steps": [
                        {"range": [s_min, (s_min+s_max)/2], "color": "#E3F2FD"},
                        {"range": [(s_min+s_max)/2, s_max], "color": "#BBDEFB"},
                    ],
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill your profile and click Predict Salary")


# ══════════════════════════════════════════════
# PAGE 5 - RAW DATA
# ══════════════════════════════════════════════
elif page == "Raw Data":
    st.title("Raw Dataset")
    st.markdown(f"{len(dff):,} job postings")
    st.markdown("---")

    search = st.text_input("Search by company, role, or city")

    display = dff.copy()
    if search:
        mask = (
            display["title"].str.contains(search, case=False, na=False) |
            display["company"].str.contains(search, case=False, na=False) |
            display["city"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    st.dataframe(
        display[["job_id","title","company","city","experience",
                  "salary_lpa","is_remote","posted_date"]].head(200),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download CSV",
        data=dff.to_csv(index=False).encode(),
        file_name="job_market_data.csv",
        mime="text/csv"
    )
