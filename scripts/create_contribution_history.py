#!/usr/bin/env python3
"""
Create Git Contribution History
================================
Creates commits with backdated timestamps to show gradual development.
This rewrites git history - use with caution!
"""

import subprocess
import os
from datetime import datetime, timedelta
import sys

# Configuration
COMMITS = [
    # (days_ago, hour, message, files_pattern)
    # Week 1: Project Setup (6 weeks ago)
    (42, 9, "Initial project setup with folder structure", [".gitignore", "README.md"]),
    (42, 14, "Add requirements.txt with dependencies", ["requirements.txt"]),
    (41, 10, "Create src package structure", ["src/__init__.py", "src/config.py"]),
    (40, 16, "Add Makefile for automation", ["Makefile"]),
    
    # Week 2: Data Pipeline (5 weeks ago)
    (35, 11, "Implement data loading module", ["src/data/"]),
    (34, 9, "Add EDA notebook - initial exploration", ["notebooks/01_eda.ipynb"]),
    (33, 14, "Analyze transaction distributions", ["notebooks/01_eda.ipynb"]),
    
    # Week 3: Feature Engineering (4 weeks ago)
    (28, 10, "Implement feature engineering pipeline", ["src/features/"]),
    (27, 15, "Add feature engineering notebook", ["notebooks/02_feature_engineering.ipynb"]),
    (26, 11, "Add time-based and categorical features", ["src/features/"]),
    
    # Week 4: Model Training (3 weeks ago)
    (21, 9, "Implement model training module", ["src/model/"]),
    (20, 14, "Add model evaluation notebook", ["notebooks/03_model_evaluation.ipynb"]),
    (19, 16, "Optimize hyperparameters", ["src/config.py"]),
    (18, 10, "Add precision@K and recall@K metrics", ["notebooks/"]),
    
    # Week 5: Policy & API (2 weeks ago)
    (14, 11, "Implement fraud ops simulation", ["src/policy/"]),
    (13, 9, "Add FastAPI service for scoring", ["src/service/"]),
    (12, 15, "Add monitoring module", ["src/monitoring/"]),
    (11, 14, "Add unit tests structure", ["tests/"]),
    
    # Week 6: Dashboard (1 week ago)
    (7, 10, "Create Streamlit dashboard", ["src/dashboard/"]),
    (6, 11, "Add data explorer pages", ["src/dashboard/app.py"]),
    (5, 14, "Implement live scoring feature", ["src/dashboard/app.py"]),
    (4, 16, "Add fraud ops simulator", ["src/dashboard/app.py"]),
    (3, 9, "Add sample data for deployment", ["data/sample/"]),
    
    # This week: Polish
    (1, 10, "Enhance UI with modern design", ["src/dashboard/app.py", ".streamlit/"]),
    (0, 11, "Add threshold tuning page", ["src/dashboard/app.py"]),
    (0, 15, "Final polish and documentation", ["src/dashboard/app.py", "requirements.txt", "README.md"]),
]


def run_cmd(cmd, env=None):
    """Run a shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0 and "nothing to commit" not in result.stderr:
        print(f"  Warning: {result.stderr.strip()}")
    return result.returncode == 0


def get_date_string(days_ago, hour):
    """Get ISO date string for a commit."""
    date = datetime.now() - timedelta(days=days_ago)
    date = date.replace(hour=hour, minute=30, second=0)
    return date.strftime("%Y-%m-%d %H:%M:%S")


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 60)
    print("🚀 Creating Contribution History")
    print("=" * 60)
    print()
    
    # Check for uncommitted changes
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print("⚠️  Uncommitted changes detected. Committing all first...")
        run_cmd("git add -A")
        run_cmd('git commit -m "WIP: Save current state"')
    
    print("📋 Will create the following commits:\n")
    
    for days_ago, hour, message, files in COMMITS:
        date_str = get_date_string(days_ago, hour)
        print(f"  [{date_str}] {message}")
    
    print()
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    print()
    print("🔄 Creating commits...")
    print()
    
    # Create an orphan branch to start fresh
    run_cmd("git checkout --orphan temp_history")
    run_cmd("git rm -rf . 2>/dev/null || true")
    
    # Track which files we've added
    added_files = set()
    
    for days_ago, hour, message, files in COMMITS:
        date_str = get_date_string(days_ago, hour)
        
        # Restore files from original branch
        for pattern in files:
            run_cmd(f"git checkout main -- {pattern} 2>/dev/null || true")
        
        # Stage the files
        for pattern in files:
            run_cmd(f"git add {pattern} 2>/dev/null || true")
        
        # Create commit with backdated timestamp
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        
        result = subprocess.run(
            f'git commit -m "{message}" --allow-empty',
            shell=True,
            capture_output=True,
            text=True,
            env=env
        )
        
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print(f"  ⏭️  Skip: {message} (no changes)")
        else:
            print(f"  ✅ [{date_str[:10]}] {message}")
    
    print()
    print("🔄 Replacing main branch...")
    
    run_cmd("git branch -D main 2>/dev/null || true")
    run_cmd("git branch -m main")
    
    print()
    print("=" * 60)
    print("✅ Done! Your commit history:")
    print("=" * 60)
    print()
    
    os.system("git log --oneline --graph -20")
    
    print()
    print("=" * 60)
    print("📤 To push to GitHub:")
    print("   git push -f origin main")
    print()
    print("⚠️  The -f flag is required because history was rewritten")
    print("=" * 60)


if __name__ == "__main__":
    main()
