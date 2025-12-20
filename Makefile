# =============================================================================
# Fraud Decisioning Platform - Makefile
# =============================================================================

.PHONY: setup download_data make_dataset train simulate serve drift_report clean test lint help

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

# Default target
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
setup: ## Create virtual environment and install dependencies
	@echo "🔧 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "📦 Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "✅ Setup complete! Activate with: source $(VENV)/bin/activate"

# -----------------------------------------------------------------------------
# DATA PIPELINE
# -----------------------------------------------------------------------------
download_data: ## Download IEEE-CIS Fraud Detection dataset from Kaggle
	@echo "📥 Downloading Kaggle IEEE-CIS Fraud Detection dataset..."
	@echo "⚠️  Ensure kaggle.json is configured in ~/.kaggle/"
	$(BIN)/python -c "import os; os.makedirs('data/raw', exist_ok=True)"
	kaggle competitions download -c ieee-fraud-detection -p data/raw/
	@echo "📂 Unzipping data..."
	cd data/raw && unzip -o ieee-fraud-detection.zip
	@echo "✅ Data downloaded to data/raw/"

make_dataset: ## Process raw data into training-ready features
	@echo "🔄 Processing raw data into features..."
	$(BIN)/python -m src.data.make_dataset
	@echo "✅ Dataset created in data/processed/"

# -----------------------------------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------------------------------
train: ## Train the fraud detection model
	@echo "🚀 Training fraud detection model..."
	$(BIN)/python -m src.model.train
	@echo "✅ Model training complete!"

# -----------------------------------------------------------------------------
# SIMULATION & SERVING
# -----------------------------------------------------------------------------
simulate: ## Run fraud-ops alert triage simulation
	@echo "🎮 Running fraud-ops simulation..."
	$(BIN)/python -m src.policy.simulate
	@echo "✅ Simulation complete! Check reports/"

serve: ## Start the FastAPI scoring service
	@echo "🌐 Starting FastAPI fraud scoring service..."
	$(BIN)/uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

serve_prod: ## Start production server (no reload)
	@echo "🚀 Starting production server..."
	$(BIN)/uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers 4

dashboard: ## Launch Streamlit dashboard for interview demo
	@echo "🎨 Starting Streamlit dashboard..."
	$(BIN)/streamlit run src/dashboard/app.py --server.port 8501 --server.headless true

# -----------------------------------------------------------------------------
# MONITORING
# -----------------------------------------------------------------------------
drift_report: ## Generate data drift report using Evidently
	@echo "📊 Generating drift report..."
	$(BIN)/python -m src.monitoring.drift_report
	@echo "✅ Drift report saved to reports/"

# -----------------------------------------------------------------------------
# QUALITY
# -----------------------------------------------------------------------------
test: ## Run tests with pytest
	@echo "🧪 Running tests..."
	$(BIN)/pytest tests/ -v --tb=short

lint: ## Run linting and formatting checks
	@echo "🔍 Running linters..."
	$(BIN)/black --check src/ tests/
	$(BIN)/isort --check-only src/ tests/
	$(BIN)/mypy src/

format: ## Auto-format code
	@echo "✨ Formatting code..."
	$(BIN)/black src/ tests/
	$(BIN)/isort src/ tests/

# -----------------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------------
clean: ## Remove generated files and caches
	@echo "🧹 Cleaning up..."
	rm -rf $(VENV)
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Clean complete!"

# -----------------------------------------------------------------------------
# HELP
# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo "Fraud Decisioning Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Example workflow:"
	@echo "  make setup          # First-time setup"
	@echo "  make download_data  # Get Kaggle data"
	@echo "  make make_dataset   # Process features"
	@echo "  make train          # Train model"
	@echo "  make serve          # Start API"
	@echo "  make dashboard      # Launch Streamlit demo"
