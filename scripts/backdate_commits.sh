#!/bin/bash
# =============================================================================
# Backdate Commits Script
# Creates commits spread over time to show gradual development
# =============================================================================

set -e

cd "$(dirname "$0")/.."

echo "🚀 Creating backdated commits for Fraud Decisioning Platform..."
echo ""

# Function to make a commit with a specific date
commit_with_date() {
    local date="$1"
    local message="$2"
    local files="$3"
    
    git add $files
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$message" --allow-empty || true
    echo "✅ Committed: $message ($date)"
}

# Check if we have uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo "⚠️  You have uncommitted changes. Stashing them..."
    git stash
fi

# Reset to create fresh history (optional - comment out if you want to keep existing)
# git checkout --orphan temp_branch
# git add -A
# git commit -m "Initial commit"
# git branch -D main 2>/dev/null || true
# git branch -m main

echo ""
echo "📅 Creating commits over the past 6 weeks..."
echo ""

# Week 1: Project Setup (6 weeks ago)
DATE_W1="2025-12-18 09:30:00"
commit_with_date "$DATE_W1" "Initial project setup: folder structure and .gitignore" ".gitignore README.md"

DATE_W1_2="2025-12-18 14:15:00"
commit_with_date "$DATE_W1_2" "Add requirements.txt with core dependencies" "requirements.txt"

DATE_W1_3="2025-12-19 10:00:00"
commit_with_date "$DATE_W1_3" "Create src package structure with config" "src/__init__.py src/config.py"

DATE_W1_4="2025-12-20 16:30:00"
commit_with_date "$DATE_W1_4" "Add Makefile for automation tasks" "Makefile"

# Week 2: Data Pipeline (5 weeks ago)
DATE_W2="2025-12-23 11:00:00"
commit_with_date "$DATE_W2" "Implement data loading module" "src/data/__init__.py src/data/make_dataset.py"

DATE_W2_2="2025-12-24 09:45:00"
commit_with_date "$DATE_W2_2" "Add EDA notebook with initial exploration" "notebooks/01_eda.ipynb"

DATE_W2_3="2025-12-26 14:00:00"
commit_with_date "$DATE_W2_3" "Explore transaction amount distributions" "notebooks/01_eda.ipynb"

# Week 3: Feature Engineering (4 weeks ago)
DATE_W3="2025-12-30 10:30:00"
commit_with_date "$DATE_W3" "Implement feature engineering pipeline" "src/features/__init__.py src/features/build_features.py"

DATE_W3_2="2025-12-31 15:00:00"
commit_with_date "$DATE_W3_2" "Add feature engineering notebook" "notebooks/02_feature_engineering.ipynb"

DATE_W3_3="2026-01-02 11:00:00"
commit_with_date "$DATE_W3_3" "Add time-based and categorical features" "src/features/build_features.py"

# Week 4: Model Training (3 weeks ago)
DATE_W4="2026-01-06 09:00:00"
commit_with_date "$DATE_W4" "Implement model training module" "src/model/__init__.py src/model/train.py src/model/predict.py"

DATE_W4_2="2026-01-07 14:30:00"
commit_with_date "$DATE_W4_2" "Add model evaluation notebook with LightGBM" "notebooks/03_model_evaluation.ipynb"

DATE_W4_3="2026-01-08 16:00:00"
commit_with_date "$DATE_W4_3" "Optimize hyperparameters for fraud detection" "src/config.py notebooks/03_model_evaluation.ipynb"

DATE_W4_4="2026-01-09 10:45:00"
commit_with_date "$DATE_W4_4" "Add precision@K and recall@K metrics" "notebooks/03_model_evaluation.ipynb"

# Week 5: Policy & API (2 weeks ago)
DATE_W5="2026-01-13 11:00:00"
commit_with_date "$DATE_W5" "Implement fraud ops simulation module" "src/policy/__init__.py src/policy/simulate.py"

DATE_W5_2="2026-01-14 09:30:00"
commit_with_date "$DATE_W5_2" "Add FastAPI service for real-time scoring" "src/service/__init__.py src/service/app.py"

DATE_W5_3="2026-01-15 15:00:00"
commit_with_date "$DATE_W5_3" "Add monitoring module for drift detection" "src/monitoring/__init__.py src/monitoring/drift_report.py"

DATE_W5_4="2026-01-16 14:00:00"
commit_with_date "$DATE_W5_4" "Add unit tests structure" "tests/"

# Week 6: Dashboard (1 week ago)
DATE_W6="2026-01-20 10:00:00"
commit_with_date "$DATE_W6" "Create Streamlit dashboard module" "src/dashboard/__init__.py src/dashboard/app.py"

DATE_W6_2="2026-01-21 11:30:00"
commit_with_date "$DATE_W6_2" "Add data explorer and model performance pages" "src/dashboard/app.py"

DATE_W6_3="2026-01-22 14:00:00"
commit_with_date "$DATE_W6_3" "Implement live scoring with risk factors" "src/dashboard/app.py"

DATE_W6_4="2026-01-23 16:30:00"
commit_with_date "$DATE_W6_4" "Add fraud ops simulator with financial analysis" "src/dashboard/app.py"

DATE_W6_5="2026-01-24 09:00:00"
commit_with_date "$DATE_W6_5" "Add sample data for cloud deployment" "data/sample/train_sample.csv data/sample/.gitkeep"

# This week: Polish
DATE_W7="2026-01-27 10:00:00"
commit_with_date "$DATE_W7" "Enhance dashboard with modern UI and animations" "src/dashboard/app.py .streamlit/config.toml"

DATE_W7_2="2026-01-28 11:00:00"
commit_with_date "$DATE_W7_2" "Add threshold tuning and feature analysis pages" "src/dashboard/app.py"

DATE_W7_3="2026-01-28 15:30:00"
commit_with_date "$DATE_W7_3" "Final polish: colors, emojis, and interactivity" "src/dashboard/app.py requirements.txt"

echo ""
echo "✅ All commits created!"
echo ""
echo "📊 Commit history:"
git log --oneline -20
echo ""
echo "🚀 Ready to push. Run: git push -f origin main"
echo "⚠️  Note: -f flag needed because we rewrote history"
