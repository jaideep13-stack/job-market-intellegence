"""
setup.py — One-command project setup
Generates synthetic data and trains the salary model so the dashboard is ready to run.

Usage:
    python setup.py
"""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(cmd, desc):
    log.info(f"▶ {desc}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Failed: {result.stderr}")
        sys.exit(1)
    log.info(f"  ✅ Done")
    return result.stdout


def main():
    print("\n" + "═" * 60)
    print("  🇮🇳 Job Market Intelligence — Project Setup")
    print("═" * 60 + "\n")

    # Step 1: Generate synthetic data
    run("python scraper.py --synthetic", "Generating 1,500 synthetic job postings...")

    # Step 2: Run SQL analysis
    run("python sql_analysis.py", "Running SQL analytical queries...")

    # Step 3: Train ML model
    run("python model.py", "Training Random Forest salary predictor...")

    print("\n" + "═" * 60)
    print("  ✅ Setup complete! Launch the dashboard with:")
    print("     streamlit run app.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
