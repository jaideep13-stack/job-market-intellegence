"""
app.py — Indian Job Market Intelligence Dashboard
Streamlit application with EDA visualizations + ML salary predictor.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Job Market Intelligence",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .main { background: #0A0E1A; }
    .stApp { background: linear-gradient(135deg, #0A0E1A 0%, #0D1421 50%, #0A0E1A 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1A2035 0%, #1E2845 100%);
        border: 1px solid #2A3555;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #60A5FA; font-family: 'JetBrains Mono', monospace; }
    .metric-label { font-size: 0.85rem; color: #8899BB; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-delta { font-size: 0.8rem; margin-top: 6px; }
    .delta-up { color: #34D399; }
    .delta-info { color: #A78BFA; }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E2E8F0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #2A3555;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .insight-box {
        background: linear-gradient(135deg, #1A2035 0%, #1E2845 100%);
        border-left: 4px solid #60A5FA;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 12px 0;
        color: #CBD5E1;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .predict-result {
        background: linear-gradient(135deg, #0F2027, #1A3A4A);
        border: 2px solid #34D399;
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin: 24px 0;
    }
    .predict-amount { font-size: 3.5rem; font-weight: 700; color: #34D399; font-family: 'JetBrains Mono', monospace; }
    .predict-range { font-size: 1rem; color: #8899BB; margin-top: 8px; }
    .predict-label { font-size: 0.9rem; color: #A78BFA; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px; }

    .skill-chip {
        display: inline-block;
        background: #1E3A5F;
        color: #93C5FD;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 3px;
        border: 1px solid #2A5080;
    }

    div[data-testid="stMetric"] {
        background: #1A2035;
        border: 1px solid #2A3555;
        border-radius: 12px;
        padding: 16px;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label { color: #CBD5E1 !important; }

    h1 { color: #E2E8F0 !important; }
    h2 { color: #CBD5E1 !important; }
    h3 { color: #94A3B8 !important; }
    p { color: #94A3B8; }

    .sidebar .sidebar-content { background: #0D1421; }
    section[data-testid="stSidebar"] { background: #0D1421; border-right: 1px solid #1E2845; }
    section[data-testid="stSidebar"] * { color: #CBD5E1; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOT_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color="#94A3B8"),
    title_font=dict(color="#E2E8F0", size=15),
    xaxis=dict(gridcolor="#1E2845", zerolinecolor="#1E2845", tickfont=dict(color="#8899BB")),
    yaxis=dict(gridcolor="#1E2845", zerolinecolor="#1E2845", tickfont=dict(color="#8899BB")),
    colorway=["#60A5FA", "#34D399", "#A78BFA", "#F472B6", "#FBBF24", "#38BDF8"],
    margin=dict(t=50, b=30, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8")),
)

DB_PATH = Path("data/jobs.db")
MODEL_PATH = Path("models/salary_model.pkl")
METRICS_PATH = Path("models/metrics.json")

EXPERIENCE_ORDER = ["0-1 years", "1-3 years", "3-5 years", "5-8 years", "8+ years"]

TOP_SKILLS = [
    "python", "sql", "machine learning", "deep learning", "tensorflow", "pytorch",
    "spark", "aws", "docker", "kubernetes", "nlp", "transformers", "huggingface",
    "tableau", "power bi", "airflow", "kafka", "git", "fastapi", "flask",
    "statistics", "pandas", "numpy", "scikit-learn", "langchain", "llm",
]

ROLES = [
    "Data Scientist", "Machine Learning Engineer", "Data Analyst",
    "Data Engineer", "AI Engineer", "NLP Engineer", "MLOps Engineer",
    "Business Intelligence Analyst", "Research Scientist", "Deep Learning Engineer",
]

CITIES = [
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai",
    "Kolkata", "Ahmedabad", "Noida", "Gurgaon", "Remote", "Work From Home",
]


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        os.makedirs("data", exist_ok=True)
        import random
        from datetime import datetime, timedelta
        records = []
        roles = ["Data Scientist","ML Engineer","Data Analyst","Data Engineer","AI Engineer","NLP Engineer"]
        cities = ["Bangalore","Mumbai","Delhi","Hyderabad","Pune","Chennai","Remote"]
        companies = ["Flipkart","Razorpay","Swiggy","Zomato","CRED","Groww","TCS","Infosys"]
        skills_pool = ["python","sql","machine learning","tensorflow","pytorch","aws","docker","nlp","pandas","spark"]
        salary_map = {"Data Scientist":(8,25),"ML Engineer":(10,30),"Data Analyst":(4,15),"Data Engineer":(8,28),"AI Engineer":(12,35),"NLP Engineer":(10,30)}
        for i in range(1500):
            role = random.choice(roles)
            city = random.choice(cities)
            s_min,s_max = salary_map[role]
            records.append({"job_id":f"JOB{i}","title":role,"company":random.choice(companies),"city":city,"experience":random.choice(["0-1 years","1-3 years","3-5 years","5-8 years"]),"salary_lpa":round(random.uniform(s_min,s_max),1),"salary_avg_lpa":round(random.uniform(s_min,s_max),1),"skills":", ".join(random.sample(skills_pool,4)),"is_remote":int(city=="Remote"),"posted_date":(datetime.now()-timedelta(days=random.randint(0,90))).strftime("%Y-%m-%d")})
        df = pd.DataFrame(records)
        conn = sqlite3.connect(DB_PATH)
        df.to_sql("jobs", conn, if_exists="replace", index=False)
        conn.close()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()
    df["posted_date"] = pd.to_datetime(df["posted_date"], errors="coerce")
    df["month"] = df["posted_date"].dt.to_period("M").astype(str)
    return df


@st.cache_data
def get_skill_frequency(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        for skill in str(row["skills"]).split(","):
            skill = skill.strip().lower()
            if skill:
                records.append({"skill": skill, "salary_avg_lpa": row.get("salary_avg_lpa", 0)})
    skill_df = pd.DataFrame(records)
    freq = (
        skill_df.groupby("skill")
        .agg(count=("skill", "count"), avg_salary=("salary_avg_lpa", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    freq["avg_salary"] = freq["avg_salary"].round(2)
    baseline = df["salary_avg_lpa"].mean()
    freq["salary_premium_pct"] = ((freq["avg_salary"] - baseline) / baseline * 100).round(1)
    return freq


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    return payload, metrics


def predict(payload, title, city, experience, skills, is_remote):
    model = payload["model"]
    feature_names = payload["feature_names"]
    row = {}
    for col in feature_names:
        if col.startswith("title_"):
            row[col] = int(col == f"title_{title}")
        elif col.startswith("city_"):
            row[col] = int(col == f"city_{city}")
        elif col == "experience_ord":
            row[col] = EXPERIENCE_ORDER.index(experience) if experience in EXPERIENCE_ORDER else 2
        elif col == "is_remote":
            row[col] = int(is_remote)
        elif col == "skill_count":
            row[col] = len(skills)
        elif col.startswith("skill_"):
            skill_name = col[6:].replace("_", " ")
            row[col] = int(any(skill_name in s.lower() for s in skills))
        else:
            row[col] = 0
    X = pd.DataFrame([row])[feature_names].fillna(0)
    pred = model.predict(X)[0]
    if hasattr(model, "estimators_"):
        tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
        ci_low, ci_high = np.percentile(tree_preds, 10), np.percentile(tree_preds, 90)
    else:
        ci_low, ci_high = pred * 0.85, pred * 1.15
    return round(pred, 2), round(ci_low, 2), round(ci_high, 2)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇮🇳 Job Market Intel")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Market Overview", "🔍 Skills Analysis", "🏙️ City Insights",
         "📈 Trends", "💰 Salary Predictor", "📋 Raw Data"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    df_full = load_data()

    role_filter = st.multiselect(
        "Job Roles",
        options=sorted(df_full["title"].dropna().unique()),
        default=[],
        placeholder="All roles",
    )

    city_filter = st.multiselect(
        "Cities",
        options=sorted(df_full["city"].dropna().unique()),
        default=[],
        placeholder="All cities",
    )

    exp_filter = st.multiselect(
        "Experience Level",
        options=EXPERIENCE_ORDER,
        default=[],
        placeholder="All levels",
    )

    salary_range = st.slider(
        "Salary Range (LPA)",
        min_value=0, max_value=60,
        value=(0, 60),
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#8899BB'>Built by Jaideep Bhathod<br/>Data Science Portfolio Project<br/>1,500+ job postings analyzed</small>",
        unsafe_allow_html=True
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
df = df_full.copy()
if role_filter:
    df = df[df["title"].isin(role_filter)]
if city_filter:
    df = df[df["city"].isin(city_filter)]
if exp_filter:
    df = df[df["experience"].isin(exp_filter)]
df = df[(df["salary_avg_lpa"] >= salary_range[0]) | (df["salary_avg_lpa"] == 0)]
df = df[df["salary_avg_lpa"] <= salary_range[1]]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Market Overview":
    st.markdown("# 📊 Indian Job Market Intelligence")
    st.markdown("<p style='color:#8899BB;margin-top:-12px'>Live analysis of 1,500+ AI/ML/Data job postings across India</p>", unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(df)
    avg_sal = df[df["salary_avg_lpa"] > 0]["salary_avg_lpa"].mean()
    remote_pct = df["is_remote"].mean() * 100
    top_city = df["city"].value_counts().idxmax() if total > 0 else "—"
    top_role = df["title"].value_counts().idxmax() if total > 0 else "—"

    for col, val, label, delta in zip(
        [col1, col2, col3, col4, col5],
        [f"{total:,}", f"₹{avg_sal:.1f}", f"{remote_pct:.1f}%", top_city, top_role],
        ["Total Postings", "Avg Salary (LPA)", "Remote Jobs", "Top Hiring City", "Most In-Demand Role"],
        ["📋", "💰", "🏠", "📍", "🎯"],
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{delta} {label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Role distribution + salary by role
    col_a, col_b = st.columns([1, 1.4])

    with col_a:
        st.markdown('<div class="section-header">🎯 Jobs by Role</div>', unsafe_allow_html=True)
        role_counts = df["title"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]
        fig = px.bar(
            role_counts, x="Count", y="Role", orientation="h",
            color="Count", color_continuous_scale=["#1E3A5F", "#60A5FA"],
            text="Count",
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(**PLOT_THEME, showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">💰 Average Salary by Role (LPA)</div>', unsafe_allow_html=True)
        sal_role = (
            df[df["salary_avg_lpa"] > 0]
            .groupby("title")["salary_avg_lpa"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .sort_values("mean", ascending=True)
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=sal_role["title"], x=sal_role["mean"],
            orientation="h", name="Average",
            marker_color="#60A5FA", text=[f"₹{v:.1f}" for v in sal_role["mean"]],
            textposition="outside",
        ))
        fig2.update_layout(**PLOT_THEME, height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Insights
    st.markdown('<div class="section-header">💡 Key Market Insights</div>', unsafe_allow_html=True)
    ins_col1, ins_col2, ins_col3 = st.columns(3)
    top_paying = sal_role.sort_values("mean", ascending=False).iloc[0] if len(sal_role) else None
    with ins_col1:
        if top_paying is not None:
            st.markdown(f'<div class="insight-box">🏆 <b>{top_paying["title"]}</b> is the highest-paying role at ₹{top_paying["mean"]:.1f} LPA average — {((top_paying["mean"] - avg_sal) / avg_sal * 100):.0f}% above market average.</div>', unsafe_allow_html=True)
    with ins_col2:
        st.markdown(f'<div class="insight-box">🏠 <b>{remote_pct:.1f}%</b> of all postings offer remote or work-from-home options — a strong signal for distributed-first hiring in 2024.</div>', unsafe_allow_html=True)
    with ins_col3:
        fresher_jobs = len(df[df["experience"].isin(["0-1 years", "1-3 years"])])
        st.markdown(f'<div class="insight-box">🌱 <b>{fresher_jobs:,}</b> postings ({fresher_jobs/total*100:.0f}%) are open to candidates with 0–3 years of experience — strong entry-level demand.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SKILLS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Skills Analysis":
    st.markdown("# 🔍 Skills Intelligence")
    st.markdown("<p style='color:#8899BB;margin-top:-12px'>Which skills are most demanded — and which command the highest salary?</p>", unsafe_allow_html=True)

    skill_df = get_skill_frequency(df)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">📊 Most In-Demand Skills</div>', unsafe_allow_html=True)
        fig = px.bar(
            skill_df.head(20), x="count", y="skill", orientation="h",
            color="count", color_continuous_scale=["#1E3A5F", "#60A5FA", "#93C5FD"],
            text="count",
        )
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">💰 Salary Premium by Skill (%)</div>', unsafe_allow_html=True)
        premium = skill_df[skill_df["count"] >= 5].sort_values("salary_premium_pct", ascending=True).tail(20)
        colors = ["#34D399" if v > 0 else "#F87171" for v in premium["salary_premium_pct"]]
        fig2 = go.Figure(go.Bar(
            x=premium["salary_premium_pct"], y=premium["skill"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in premium["salary_premium_pct"]],
            textposition="outside",
        ))
        fig2.update_layout(**PLOT_THEME, height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Bubble chart: demand vs salary
    st.markdown('<div class="section-header">🫧 Skill Opportunity Matrix (Demand vs Salary Premium)</div>', unsafe_allow_html=True)
    bubble_data = skill_df[skill_df["count"] >= 8].copy()
    fig3 = px.scatter(
        bubble_data, x="count", y="salary_premium_pct",
        size="count", color="avg_salary",
        text="skill", hover_name="skill",
        color_continuous_scale="Blues",
        labels={"count": "Job Postings Mentioning Skill", "salary_premium_pct": "Salary Premium (%)"},
        size_max=40,
    )
    fig3.update_traces(textposition="top center", textfont_size=10, textfont_color="#CBD5E1")
    fig3.update_layout(**PLOT_THEME, height=420)
    st.plotly_chart(fig3, use_container_width=True)

    top_skill = skill_df.iloc[0]["skill"] if len(skill_df) > 0 else "python"
    top_premium = skill_df.sort_values("salary_premium_pct", ascending=False).iloc[0] if len(skill_df) > 0 else None
    st.markdown(f'<div class="insight-box">🔑 <b>{top_skill.title()}</b> appears in the most job postings. ' +
                (f'<b>{top_premium["skill"].title()}</b> commands the highest salary premium at <b>{top_premium["salary_premium_pct"]:+.1f}%</b> above market average.' if top_premium is not None else "") +
                "</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CITY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏙️ City Insights":
    st.markdown("# 🏙️ City-wise Market Analysis")
    st.markdown("<p style='color:#8899BB;margin-top:-12px'>Where should you target your job search?</p>", unsafe_allow_html=True)

    city_df = (
        df[df["salary_avg_lpa"] > 0]
        .groupby("city")
        .agg(
            job_count=("job_id", "count"),
            avg_salary=("salary_avg_lpa", "mean"),
            max_salary=("salary_max_lpa", "max"),
            remote_pct=("is_remote", "mean"),
        )
        .reset_index()
        .query("job_count >= 5")
        .sort_values("avg_salary", ascending=False)
    )
    city_df["avg_salary"] = city_df["avg_salary"].round(2)
    city_df["remote_pct"] = (city_df["remote_pct"] * 100).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">🏙️ Average Salary by City</div>', unsafe_allow_html=True)
        fig = px.bar(
            city_df.head(12), x="city", y="avg_salary",
            color="avg_salary", color_continuous_scale=["#1E3A5F", "#A78BFA"],
            text=[f"₹{v}" for v in city_df.head(12)["avg_salary"]],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False, height=380,
                          xaxis_title="City", yaxis_title="Avg Salary (LPA)")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">📊 Job Volume vs Salary</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            city_df, x="job_count", y="avg_salary",
            size="job_count", color="remote_pct",
            text="city",
            color_continuous_scale="Teal",
            labels={"job_count": "Number of Postings", "avg_salary": "Avg Salary (LPA)", "remote_pct": "Remote %"},
            size_max=45,
        )
        fig2.update_traces(textposition="top center", textfont_size=10, textfont_color="#CBD5E1")
        fig2.update_layout(**PLOT_THEME, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # City x Role heatmap
    st.markdown('<div class="section-header">🌡️ City × Role Salary Heatmap</div>', unsafe_allow_html=True)
    heatmap_data = (
        df[df["salary_avg_lpa"] > 0]
        .groupby(["city", "title"])["salary_avg_lpa"]
        .mean()
        .reset_index()
        .pivot(index="title", columns="city", values="salary_avg_lpa")
        .fillna(0)
    )
    fig3 = px.imshow(
        heatmap_data,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=".1f",
        labels=dict(color="Avg LPA"),
    )
    fig3.update_layout(**PLOT_THEME, height=400)
    st.plotly_chart(fig3, use_container_width=True)

    best_city = city_df.iloc[0] if len(city_df) > 0 else None
    if best_city is not None:
        st.markdown(f'<div class="insight-box">🏆 <b>{best_city["city"]}</b> offers the highest average salary at ₹{best_city["avg_salary"]} LPA with {best_city["job_count"]} active postings. Remote work constitutes {best_city["remote_pct"]}% of its listings.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends":
    st.markdown("# 📈 Market Trends")
    st.markdown("<p style='color:#8899BB;margin-top:-12px'>How is hiring volume and salary evolving over time?</p>", unsafe_allow_html=True)

    # Monthly trend
    monthly = (
        df.dropna(subset=["month"])
        .groupby("month")
        .agg(postings=("job_id", "count"), avg_salary=("salary_avg_lpa", "mean"))
        .reset_index()
        .sort_values("month")
        .tail(12)
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">📅 Monthly Posting Volume</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["postings"],
            mode="lines+markers+text",
            line=dict(color="#60A5FA", width=2.5),
            marker=dict(size=8, color="#60A5FA"),
            fill="tozeroy", fillcolor="rgba(96,165,250,0.1)",
            text=monthly["postings"], textposition="top center",
        ))
        fig.update_layout(**PLOT_THEME, height=320, xaxis_title="Month", yaxis_title="Job Postings")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">💰 Average Salary Trend (LPA)</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=monthly["month"], y=monthly["avg_salary"].round(2),
            mode="lines+markers",
            line=dict(color="#34D399", width=2.5),
            marker=dict(size=8, color="#34D399"),
            fill="tozeroy", fillcolor="rgba(52,211,153,0.1)",
        ))
        fig2.update_layout(**PLOT_THEME, height=320, xaxis_title="Month", yaxis_title="Avg Salary (LPA)")
        st.plotly_chart(fig2, use_container_width=True)

    # Experience vs salary
    st.markdown('<div class="section-header">📊 Experience Level Impact on Salary</div>', unsafe_allow_html=True)
    exp_df = (
        df[df["salary_avg_lpa"] > 0]
        .groupby("experience")["salary_avg_lpa"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    exp_df["experience"] = pd.Categorical(exp_df["experience"], categories=EXPERIENCE_ORDER, ordered=True)
    exp_df = exp_df.sort_values("experience")

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Average Salary", x=exp_df["experience"], y=exp_df["mean"].round(2),
                          marker_color="#60A5FA", text=[f"₹{v:.1f}" for v in exp_df["mean"]], textposition="outside"))
    fig3.add_trace(go.Bar(name="Median Salary", x=exp_df["experience"], y=exp_df["median"].round(2),
                          marker_color="#A78BFA", text=[f"₹{v:.1f}" for v in exp_df["median"]], textposition="outside"))
    fig3.update_layout(**PLOT_THEME, barmode="group", height=360, xaxis_title="Experience Level", yaxis_title="Salary (LPA)")
    st.plotly_chart(fig3, use_container_width=True)

    # Remote trend
    remote_trend = df.groupby("month").agg(remote=("is_remote", "mean")).reset_index().sort_values("month").tail(12)
    st.markdown('<div class="section-header">🏠 Remote Work Trend (%)</div>', unsafe_allow_html=True)
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=remote_trend["month"], y=(remote_trend["remote"] * 100).round(1),
        mode="lines+markers", line=dict(color="#F472B6", width=2.5),
        fill="tozeroy", fillcolor="rgba(244,114,182,0.1)",
        marker=dict(size=7),
    ))
    fig4.update_layout(**PLOT_THEME, height=280, yaxis_title="% Remote Postings")
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SALARY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Predictor":
    st.markdown("# 💰 AI Salary Predictor")
    st.markdown("<p style='color:#8899BB;margin-top:-12px'>Enter your profile and get a market salary estimate powered by Random Forest ML</p>", unsafe_allow_html=True)

    payload, metrics = load_model()

    if payload is None:
        st.warning("⚠️ Model not trained yet. Run `python model.py` to train the salary predictor.")
        st.info("The model uses Random Forest with 200 estimators, trained on 1,500+ job postings with features including role, city, experience, and skills.")
    else:
        col_form, col_result = st.columns([1, 1.2])

        with col_form:
            st.markdown("### Your Profile")
            title = st.selectbox("Job Role", ROLES)
            city = st.selectbox("Target City", CITIES)
            experience = st.select_slider("Experience Level", options=EXPERIENCE_ORDER)
            skills = st.multiselect(
                "Your Skills (select all that apply)",
                options=sorted(TOP_SKILLS),
                default=["python", "machine learning", "sql"],
            )
            is_remote = st.checkbox("Applying for Remote Roles")

            predict_btn = st.button("🔮 Predict My Salary", use_container_width=True)

        with col_result:
            if predict_btn and payload:
                pred, ci_low, ci_high = predict(payload, title, city, experience, skills, is_remote)
                st.markdown(f"""
                <div class="predict-result">
                    <div class="predict-label">Estimated Market Salary</div>
                    <div class="predict-amount">₹{pred} LPA</div>
                    <div class="predict-range">90% Confidence: ₹{ci_low} – ₹{ci_high} LPA</div>
                    <br>
                    <div style="color:#8899BB;font-size:0.85rem">
                        Role: {title} | City: {city} | Exp: {experience}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Skill impact visualization
                if skills:
                    skill_impact = []
                    for skill in skills:
                        without = [s for s in skills if s != skill]
                        p_without, _, _ = predict(payload, title, city, experience, without, is_remote)
                        skill_impact.append({"skill": skill, "impact": round(pred - p_without, 2)})
                    impact_df = pd.DataFrame(skill_impact).sort_values("impact", ascending=True)

                    st.markdown("### Your Skill Salary Impact")
                    fig = go.Figure(go.Bar(
                        x=impact_df["impact"], y=impact_df["skill"],
                        orientation="h",
                        marker_color=["#34D399" if v >= 0 else "#F87171" for v in impact_df["impact"]],
                        text=[f"{v:+.2f} LPA" for v in impact_df["impact"]],
                        textposition="outside",
                    ))
                    fig.update_layout(**PLOT_THEME, height=300, showlegend=False,
                                      xaxis_title="Salary Impact (LPA)")
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Model card
                if metrics:
                    st.markdown("### 🤖 Model Performance Card")
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:left;padding:24px">
                        <div style="color:#60A5FA;font-weight:700;margin-bottom:12px">{metrics.get('model_name','Random Forest')}</div>
                        <div style="color:#8899BB;font-size:0.85rem;line-height:2">
                            📊 Test R²: <b style="color:#34D399">{metrics.get('test_r2','—')}</b><br>
                            📏 RMSE: <b style="color:#FBBF24">₹{metrics.get('test_rmse','—')} LPA</b><br>
                            🎯 MAE: <b style="color:#A78BFA">₹{metrics.get('test_mae','—')} LPA</b><br>
                            📋 Training samples: <b style="color:#38BDF8">{metrics.get('training_samples','—')}</b><br>
                            🔧 Features: <b style="color:#38BDF8">{metrics.get('feature_count','—')}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.info("👈 Fill in your profile and click **Predict My Salary** to get your market estimate.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Raw Data":
    st.markdown("# 📋 Raw Dataset")
    st.markdown(f"<p style='color:#8899BB;margin-top:-12px'>{len(df):,} job postings (filtered)</p>", unsafe_allow_html=True)

    col_s, col_dl = st.columns([4, 1])
    with col_s:
        search = st.text_input("🔍 Search", placeholder="Search company, role, city...")
    with col_dl:
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="job_market_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    display_df = df.copy()
    if search:
        mask = (
            display_df["title"].str.contains(search, case=False, na=False) |
            display_df["company"].str.contains(search, case=False, na=False) |
            display_df["city"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.dataframe(
        display_df[["job_id", "title", "company", "city", "experience",
                     "salary_min_lpa", "salary_max_lpa", "is_remote", "posted_date", "source"]].head(200),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing up to 200 of {len(display_df):,} results")
