"""
model.py — Job Market Intelligence Dashboard
Trains a Random Forest salary predictor and saves it with full evaluation metrics.

Features used:
    - Job title (encoded)
    - City (encoded)
    - Experience level (ordinal)
    - Skills (multi-hot encoded)
    - Is remote (binary)

Usage:
    python model.py
"""

import pandas as pd
import numpy as np
import sqlite3
import pickle
import json
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DB_PATH = Path("data/jobs.db")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

EXPERIENCE_ORDER = ["0-1 years", "1-3 years", "3-5 years", "5-8 years", "8+ years"]

TOP_SKILLS = [
    "python", "sql", "machine learning", "deep learning", "tensorflow", "pytorch",
    "spark", "aws", "docker", "kubernetes", "nlp", "transformers", "huggingface",
    "tableau", "power bi", "airflow", "kafka", "git", "fastapi", "flask",
    "statistics", "pandas", "numpy", "scikit-learn", "langchain", "llm",
]


def load_and_prepare(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load jobs from DB and engineer features."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM jobs WHERE salary_avg_lpa > 3 AND salary_avg_lpa < 80",
        conn
    )
    conn.close()

    log.info(f"Loaded {len(df)} records with valid salaries.")

    # Ordinal encode experience
    df["experience_ord"] = df["experience"].map(
        {v: i for i, v in enumerate(EXPERIENCE_ORDER)}
    ).fillna(2)

    # Multi-hot encode top skills
    def has_skill(skills_str, skill):
        return int(skill.lower() in str(skills_str).lower())

    for skill in TOP_SKILLS:
        col = "skill_" + skill.replace(" ", "_").replace("-", "_")
        df[col] = df["skills"].apply(lambda s: has_skill(s, skill))

    # Skill count feature
    df["skill_count"] = df["skills"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip()])
    )

    return df


def build_feature_matrix(df: pd.DataFrame):
    """Construct feature matrix X and target y."""
    cat_features = ["title", "city"]
    num_features = ["experience_ord", "is_remote", "skill_count"]
    skill_features = [c for c in df.columns if c.startswith("skill_")]

    X_cat = pd.get_dummies(df[cat_features], drop_first=False, dtype=int)
    X_num = df[num_features].fillna(0)
    X_skills = df[skill_features].fillna(0)

    X = pd.concat([X_cat, X_num, X_skills], axis=1)
    y = df["salary_avg_lpa"]

    return X, y, X.columns.tolist()


def train_and_evaluate(db_path: Path = DB_PATH) -> dict:
    """
    Train multiple models, compare with cross-validation, save best model.
    Returns evaluation metrics dict.
    """
    df = load_and_prepare(db_path)
    X, y, feature_names = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log.info(f"Training set: {len(X_train)} | Test set: {len(X_test)}")
    log.info(f"Feature count: {X.shape[1]}")

    # ── Model comparison ──────────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            random_state=42,
        ),
        "Ridge Regression": Ridge(alpha=10.0),
    }

    cv_scores = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        cv_scores[name] = scores.mean()
        log.info(f"{name}: CV R² = {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    log.info(f"\n✅ Best model: {best_name} (CV R² = {cv_scores[best_name]:.4f})")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # ── Evaluation on held-out test set ───────────────────────────────────────
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    log.info(f"\n📊 Test Set Results:")
    log.info(f"   RMSE : ₹{rmse:.2f} LPA")
    log.info(f"   MAE  : ₹{mae:.2f} LPA")
    log.info(f"   R²   : {r2:.4f}")

    # ── Feature importance ────────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)
        importances.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        log.info(f"\nTop 5 features:\n{importances.head(5).to_string(index=False)}")

    # ── Save model + metadata ─────────────────────────────────────────────────
    model_payload = {
        "model": best_model,
        "feature_names": feature_names,
        "model_name": best_name,
    }
    with open(MODEL_DIR / "salary_model.pkl", "wb") as f:
        pickle.dump(model_payload, f)

    metrics = {
        "model_name": best_name,
        "cv_r2_scores": {k: round(v, 4) for k, v in cv_scores.items()},
        "test_rmse": round(rmse, 4),
        "test_mae": round(mae, 4),
        "test_r2": round(r2, 4),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": X.shape[1],
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"\n✅ Model saved to {MODEL_DIR}/salary_model.pkl")
    return metrics


def predict_salary(
    title: str,
    city: str,
    experience: str,
    skills: list[str],
    is_remote: bool = False,
) -> dict:
    """
    Load saved model and predict salary for given inputs.
    Returns dict with prediction + confidence interval.
    """
    model_path = MODEL_DIR / "salary_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not trained yet. Run model.py first.")

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    feature_names = payload["feature_names"]

    # Build input row matching training feature matrix
    row = {}

    # One-hot: title
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

    X_input = pd.DataFrame([row])[feature_names].fillna(0)
    pred = model.predict(X_input)[0]

    # Estimate confidence interval using tree predictions (RF only)
    if hasattr(model, "estimators_"):
        tree_preds = np.array([t.predict(X_input)[0] for t in model.estimators_])
        ci_low = np.percentile(tree_preds, 10)
        ci_high = np.percentile(tree_preds, 90)
    else:
        ci_low = pred * 0.85
        ci_high = pred * 1.15

    return {
        "predicted_lpa": round(pred, 2),
        "ci_low_lpa": round(ci_low, 2),
        "ci_high_lpa": round(ci_high, 2),
        "model_used": payload["model_name"],
    }


if __name__ == "__main__":
    metrics = train_and_evaluate()
    print("\n\n📊 Final Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Test prediction
    print("\n\n🔮 Sample Prediction:")
    result = predict_salary(
        title="Data Scientist",
        city="Bangalore",
        experience="1-3 years",
        skills=["python", "machine learning", "sql", "tensorflow"],
        is_remote=False,
    )
    print(f"  Predicted Salary: ₹{result['predicted_lpa']} LPA")
    print(f"  90% CI: ₹{result['ci_low_lpa']} – ₹{result['ci_high_lpa']} LPA")
