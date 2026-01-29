#!/usr/bin/env python3
"""
Backdate Commits for Any Repository
====================================
Creates a fresh git history with commits spread over time.

Usage:
    1. Copy this script to your project folder
    2. Edit the COMMITS list below with your files
    3. Run: python3 backdate_any_repo.py
    4. Push: git push -f origin main

WARNING: This rewrites git history! Only use on personal repos.
"""

import subprocess
import os
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION - EDIT THIS FOR YOUR PROJECT
# =============================================================================

# How many weeks to spread commits over
WEEKS_BACK = 6

# Define your commits: (week_number, day_of_week, hour, message, files)
# week_number: 1 = oldest (WEEKS_BACK ago), higher = more recent
# day_of_week: 0 = Monday, 6 = Sunday
# files: list of file patterns to include in this commit

COMMITS = [
    # Week 1: Initial setup
    (1, 0, 9, "Initial project setup", ["README.md", ".gitignore", "requirements.txt"]),
    (1, 1, 14, "Add core configuration", ["*.py", "*.json", "*.yaml", "*.toml"]),
    
    # Week 2: Core functionality
    (2, 0, 10, "Implement core modules", ["src/", "lib/", "app/"]),
    (2, 2, 15, "Add data processing", ["data/", "*.csv", "*.json"]),
    
    # Week 3: Features
    (3, 0, 11, "Add main features", ["*.py"]),
    (3, 3, 14, "Implement utilities", ["utils/", "helpers/", "common/"]),
    
    # Week 4: Testing
    (4, 1, 9, "Add unit tests", ["tests/", "test_*.py", "*_test.py"]),
    (4, 3, 16, "Add integration tests", ["tests/"]),
    
    # Week 5: Documentation & Polish
    (5, 0, 10, "Add documentation", ["docs/", "*.md", "*.rst"]),
    (5, 2, 14, "Code cleanup and refactoring", ["*.py"]),
    
    # Week 6: Final touches
    (6, 0, 11, "Add deployment configuration", ["Dockerfile", "docker-compose*", ".github/", "Makefile"]),
    (6, 3, 15, "Final polish and README update", ["README.md", "*.py"]),
]

# =============================================================================
# SCRIPT - DON'T EDIT BELOW
# =============================================================================

def run(cmd):
    """Run shell command."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def get_date(week_num, day_of_week, hour):
    """Calculate date based on week number and day."""
    today = datetime.now()
    weeks_ago = WEEKS_BACK - week_num + 1
    target_date = today - timedelta(weeks=weeks_ago)
    
    # Adjust to correct day of week
    current_dow = target_date.weekday()
    target_date += timedelta(days=(day_of_week - current_dow))
    
    target_date = target_date.replace(hour=hour, minute=30, second=0)
    return target_date.strftime("%Y-%m-%d %H:%M:%S")

def main():
    print("=" * 60)
    print("🚀 Backdate Repository Commits")
    print("=" * 60)
    print()
    
    # Check if we're in a git repo
    success, _, _ = run("git rev-parse --git-dir")
    if not success:
        print("❌ Not a git repository! Run 'git init' first.")
        return
    
    # Show planned commits
    print("📋 Planned commits:\n")
    for week, dow, hour, msg, files in COMMITS:
        date = get_date(week, dow, hour)
        print(f"  [{date[:10]}] {msg}")
        print(f"              Files: {', '.join(files)}")
    
    print()
    response = input("Continue? This will REWRITE git history! (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return
    
    print()
    print("🔄 Saving current state...")
    
    # Save current files
    run("git stash --include-untracked")
    
    # Create orphan branch
    run("git checkout --orphan temp_backdate")
    run("git rm -rf . 2>/dev/null")
    
    # Restore files from stash
    run("git stash pop 2>/dev/null")
    
    print("📝 Creating commits...\n")
    
    added_files = set()
    
    for week, dow, hour, msg, patterns in COMMITS:
        date_str = get_date(week, dow, hour)
        
        # Add files matching patterns
        for pattern in patterns:
            run(f"git add {pattern} 2>/dev/null")
        
        # Create commit with backdated timestamp
        env_vars = f'GIT_AUTHOR_DATE="{date_str}" GIT_COMMITTER_DATE="{date_str}"'
        success, out, err = run(f'{env_vars} git commit -m "{msg}" --allow-empty')
        
        if "nothing to commit" in (out + err):
            print(f"  ⏭️  [{date_str[:10]}] {msg} (no new files)")
        else:
            print(f"  ✅ [{date_str[:10]}] {msg}")
    
    # Add any remaining files
    print()
    print("📦 Adding remaining files...")
    run("git add -A")
    
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    env_vars = f'GIT_AUTHOR_DATE="{today}" GIT_COMMITTER_DATE="{today}"'
    run(f'{env_vars} git commit -m "Additional files and updates" --allow-empty')
    
    # Replace main branch
    print()
    print("🔄 Replacing main branch...")
    run("git branch -D main 2>/dev/null")
    run("git branch -m main")
    
    print()
    print("=" * 60)
    print("✅ Done! New commit history:")
    print("=" * 60)
    print()
    
    os.system("git log --oneline --date=short --format='%h %ad %s' | head -15")
    
    print()
    print("=" * 60)
    print("📤 To push to GitHub:")
    print("   git push -f origin main")
    print()
    print("⚠️  Use -f because history was rewritten")
    print("=" * 60)

if __name__ == "__main__":
    main()
