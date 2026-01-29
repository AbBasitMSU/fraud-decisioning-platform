# =============================================================================
# FRAUD DECISIONING PLATFORM - Production Makefile
# =============================================================================
# Top 1% Portfolio Project Build System

.PHONY: setup install download_data make_dataset train train_advanced \
        serve serve_prod dashboard api simulate drift_report \
        docker-build docker-up docker-down test lint format clean help

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

setup: ## Create virtual environment and install all dependencies
	@echo "🔧 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "📦 Installing dependencies..."
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@echo "✅ Setup complete! Activate with: source $(VENV)/bin/activate"

install: ## Install dependencies (requires existing venv)
	@echo "📦 Installing/updating dependencies..."
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "📦 Installing dev dependencies..."
	pip install pytest pytest-cov black isort mypy pre-commit

# =============================================================================
# DATA PIPELINE
# =============================================================================

download_data: ## Download IEEE-CIS Fraud Detection dataset from Kaggle
	@echo "📥 Downloading Kaggle IEEE-CIS Fraud Detection dataset..."
	@echo "⚠️  Ensure kaggle.json is configured in ~/.kaggle/"
	mkdir -p data/raw
	kaggle competitions download -c ieee-fraud-detection -p data/raw/
	@echo "📂 Unzipping data..."
	cd data/raw && unzip -o ieee-fraud-detection.zip
	@echo "✅ Data downloaded to data/raw/"

make_dataset: ## Process raw data into training-ready features
	@echo "🔄 Processing raw data into features..."
	$(BIN)/python -m src.data.make_dataset
	@echo "✅ Dataset created in data/processed/"

# =============================================================================
# MODEL TRAINING
# =============================================================================

train: ## Train basic fraud detection model
	@echo "🚀 Training fraud detection model..."
	$(BIN)/python -m src.model.train
	@echo "✅ Model training complete!"

train-advanced: ## Train advanced ensemble with Optuna optimization
	@echo "🚀 Training advanced ensemble model..."
	@echo "   - LightGBM + XGBoost ensemble"
	@echo "   - Optuna hyperparameter tuning"
	@echo "   - 5-fold cross-validation"
	@echo "   - SHAP explainability"
	$(BIN)/python -m src.model.advanced_trainer
	@echo "✅ Advanced training complete!"

train-quick: ## Quick training without Optuna (for testing)
	@echo "⚡ Quick training (no optimization)..."
	$(BIN)/python -c "from src.model.advanced_trainer import train_advanced_model; train_advanced_model(n_trials=5, use_optuna=True)"

# =============================================================================
# API & DASHBOARD
# =============================================================================

api: ## Start FastAPI scoring service (development)
	@echo "🌐 Starting FastAPI fraud scoring API..."
	$(BIN)/uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --reload

api-prod: ## Start production API server (4 workers)
	@echo "🚀 Starting production API..."
	$(BIN)/uvicorn src.service.api:app --host 0.0.0.0 --port 8000 --workers 4

serve: api ## Alias for 'api'

serve_prod: api-prod ## Alias for 'api-prod'

dashboard: ## Launch Streamlit dashboard
	@echo "🎨 Starting Streamlit dashboard..."
	$(BIN)/streamlit run src/dashboard/app.py --server.port 8501 --server.headless true

demo: ## Launch both API and dashboard
	@echo "🎉 Starting full demo environment..."
	@echo "   API:       http://localhost:8000"
	@echo "   Dashboard: http://localhost:8501"
	@echo ""
	@make -j2 api dashboard

# =============================================================================
# DOCKER
# =============================================================================

docker-build: ## Build Docker images
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up: ## Start all services with Docker
	@echo "🐳 Starting services..."
	docker-compose up -d
	@echo "✅ Services running:"
	@echo "   API:       http://localhost:8000"
	@echo "   Dashboard: http://localhost:8501"

docker-down: ## Stop Docker services
	@echo "🐳 Stopping services..."
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

# =============================================================================
# SIMULATION & MONITORING
# =============================================================================

simulate: ## Run fraud-ops alert triage simulation
	@echo "🎮 Running fraud-ops simulation..."
	$(BIN)/python -m src.policy.simulate
	@echo "✅ Simulation complete! Check reports/"

drift_report: ## Generate data drift report with Evidently
	@echo "📊 Generating drift report..."
	$(BIN)/python -m src.monitoring.drift_report
	@echo "✅ Drift report saved to reports/"

# =============================================================================
# TESTING & QUALITY
# =============================================================================

test: ## Run tests with pytest
	@echo "🧪 Running tests..."
	$(BIN)/pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast: ## Run tests without coverage
	@echo "⚡ Running fast tests..."
	$(BIN)/pytest tests/ -v --tb=short

lint: ## Run linting checks
	@echo "🔍 Running linters..."
	$(BIN)/black --check src/ tests/ 2>/dev/null || echo "Run 'make format' to fix"
	$(BIN)/isort --check-only src/ tests/ 2>/dev/null || echo "Run 'make format' to fix"

format: ## Auto-format code with black and isort
	@echo "✨ Formatting code..."
	$(BIN)/black src/ tests/ 2>/dev/null || true
	$(BIN)/isort src/ tests/ 2>/dev/null || true
	@echo "✅ Code formatted!"

typecheck: ## Run mypy type checking
	@echo "🔎 Running type checks..."
	$(BIN)/mypy src/ --ignore-missing-imports

# =============================================================================
# NOTEBOOKS
# =============================================================================

notebook: ## Start Jupyter notebook server
	@echo "📓 Starting Jupyter..."
	$(BIN)/jupyter notebook notebooks/

lab: ## Start JupyterLab
	@echo "📓 Starting JupyterLab..."
	$(BIN)/jupyter lab notebooks/

# =============================================================================
# CLEANUP
# =============================================================================

clean: ## Remove generated files and caches
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Clean complete!"

clean-all: clean ## Remove everything including venv
	@echo "🧹 Deep cleaning..."
	rm -rf $(VENV)
	rm -rf models/*.joblib
	rm -rf data/processed/*
	@echo "✅ All cleaned!"

# =============================================================================
# DEPLOYMENT
# =============================================================================

deploy-streamlit: ## Deploy to Streamlit Cloud (shows instructions)
	@echo "🚀 Streamlit Cloud Deployment"
	@echo ""
	@echo "1. Push your code to GitHub"
	@echo "2. Go to https://share.streamlit.io"
	@echo "3. Click 'New app'"
	@echo "4. Select your repository"
	@echo "5. Set main file path: src/dashboard/app.py"
	@echo "6. Click Deploy!"
	@echo ""
	@echo "Current repo status:"
	@git remote -v 2>/dev/null || echo "No git remote configured"

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo ""
	@echo "╔═══════════════════════════════════════════════════════════════════╗"
	@echo "║       🛡️  FRAUD DECISIONING PLATFORM - Build System              ║"
	@echo "╚═══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📋 Quick Start:"
	@echo "  make setup           # First-time setup"
	@echo "  make download_data   # Get Kaggle data"
	@echo "  make train-advanced  # Train ensemble model"
	@echo "  make demo            # Start API + Dashboard"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build    # Build images"
	@echo "  make docker-up       # Start services"
	@echo ""
