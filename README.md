<p align="center">
  <img src="https://img.shields.io/badge/🛡️-Fraud%20Decisioning%20Platform-6366f1?style=for-the-badge" alt="FDP">
</p>

<h1 align="center">Fraud Decisioning Platform</h1>

<p align="center">
  <strong>Production-grade ML fraud detection with real-time scoring, ensemble learning, and intelligent alert triage</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#api">API</a> •
  <a href="#dashboard">Dashboard</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/LightGBM-4.0+-02569B?style=flat-square" alt="LightGBM">
  <img src="https://img.shields.io/badge/XGBoost-2.0+-ff6600?style=flat-square" alt="XGBoost">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
</p>

---

## 🎯 Problem Statement

Financial fraud costs institutions **billions annually**. This platform addresses two critical challenges:

| Challenge | Solution |
|-----------|----------|
| **Real-time Detection** | Sub-100ms prediction latency with ensemble ML models |
| **Capacity Constraints** | Intelligent alert triage that maximizes expected value |
| **Explainability** | SHAP-based model interpretability for compliance |
| **Operations Optimization** | Fraud-ops simulation with ROI analysis |

### Dataset

Uses the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset by Vesta Corporation:

- **590,000+** transactions
- **400+** engineered features
- **3.5%** fraud rate (highly imbalanced)
- Rich features: transaction amounts, card info, device fingerprints, time deltas

---

## ✨ Features

### 🤖 Advanced Machine Learning

- **Ensemble Models**: LightGBM + XGBoost weighted ensemble
- **Hyperparameter Optimization**: Optuna with TPE sampler
- **Cross-Validation**: Stratified K-fold for robust evaluation
- **Calibrated Probabilities**: Isotonic regression calibration
- **Class Imbalance**: Stratified sampling, cost-sensitive learning

### 📊 Model Performance

| Metric | Score |
|--------|-------|
| **AUC-ROC** | 0.94+ |
| **AUC-PR** | 0.65+ |
| **Precision@500** | 45%+ |
| **Recall@500** | 12%+ |

### 🔍 Explainability

- **SHAP Values**: TreeExplainer for feature contributions
- **Feature Importance**: Ensemble-weighted importance
- **Risk Factors**: Human-readable explanations
- **Waterfall Plots**: Individual prediction breakdown

### 🚀 Production Ready

- **FastAPI Backend**: Async, documented, with OpenAPI spec
- **Docker Support**: Multi-stage builds, compose orchestration
- **Health Checks**: Liveness and readiness probes
- **Metrics**: Latency tracking, error rates, risk distribution
- **Monitoring**: Evidently for data/model drift detection

### 🎨 Interactive Dashboard

- **7 Pages**: Overview, Data Explorer, Model Performance, Live Scoring, Fraud Ops, Feature Analysis, Threshold Tuning
- **Real-time Scoring**: Score transactions interactively
- **Fraud Ops Simulator**: Configure team capacity, run ROI analysis
- **Beautiful UI**: Modern design with animations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      🛡️ FRAUD DECISIONING PLATFORM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────────────┐  │
│  │  Raw Data   │────▶│    Feature      │────▶│    Ensemble Training     │  │
│  │  (Kaggle)   │     │   Engineering   │     │  ┌────────┐ ┌────────┐   │  │
│  │  590K txns  │     │  400+ features  │     │  │LightGBM│+│XGBoost │   │  │
│  └─────────────┘     └─────────────────┘     │  └────────┘ └────────┘   │  │
│                                              │         │                 │  │
│                                              │    Optuna HPO             │  │
│                                              │    5-Fold CV              │  │
│                                              └──────────┬───────────────┘  │
│                                                         │                   │
│  ┌──────────────────────────────────────────────────────┼───────────────┐  │
│  │                        SERVING LAYER                 ▼               │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │  │
│  │  │   FastAPI   │◀───│   Scoring   │◀───│   Ensemble Predictor    │  │  │
│  │  │  REST API   │    │   Engine    │    │   (0.6×LGB + 0.4×XGB)   │  │  │
│  │  │  <100ms     │    │             │    └─────────────────────────┘  │  │
│  │  └──────┬──────┘    └──────┬──────┘                                 │  │
│  │         │                  │                                         │  │
│  │         ▼                  ▼                                         │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │  │
│  │  │   Metrics   │    │    SHAP     │    │     Risk Tiering        │  │  │
│  │  │  Dashboard  │    │ Explainer   │    │  CRITICAL│HIGH│MED│LOW  │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      MONITORING & OPERATIONS                          │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │  │
│  │  │  Evidently  │    │   MLflow    │    │   Fraud-Ops Simulator   │  │  │
│  │  │Drift Reports│    │  Tracking   │    │   Capacity Planning     │  │  │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation

```bash
# Clone repository
git clone https://github.com/AbBasitMSU/fraud-decisioning-platform.git
cd fraud-decisioning-platform

# Setup environment
make setup
source .venv/bin/activate

# Download data (requires Kaggle API)
make download_data

# Train advanced ensemble model
make train-advanced

# Start demo (API + Dashboard)
make demo
```

### Docker Deployment

```bash
# Build images
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

**Services:**
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 📡 API Reference

### Score Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "TXN_12345",
    "TransactionAmt": 150.00,
    "ProductCD": "W",
    "card4": "visa",
    "DeviceType": "mobile",
    "hour": 14
  }'
```

**Response:**

```json
{
  "transaction_id": "TXN_12345",
  "fraud_probability": 0.0342,
  "risk_tier": "LOW",
  "recommended_action": "APPROVE",
  "confidence": 0.932,
  "latency_ms": 12.4,
  "risk_factors": ["Mobile device"],
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Risk Tiers

| Tier | Probability | Action | Description |
|------|------------|--------|-------------|
| 🚨 **CRITICAL** | ≥80% | BLOCK | Auto-decline, flag for investigation |
| ⚠️ **HIGH** | ≥50% | REVIEW | Manual review required |
| 📊 **MEDIUM** | ≥20% | CHALLENGE | Step-up authentication |
| ✅ **LOW** | <20% | APPROVE | Auto-approve |

### Batch Scoring

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/metrics` | GET | Performance metrics |
| `/score` | POST | Score single transaction |
| `/score/batch` | POST | Batch scoring |
| `/explain/{id}` | POST | SHAP explanation |

---

## 🎨 Dashboard

### Pages

| Page | Features |
|------|----------|
| 🏠 **Overview** | Architecture diagram, key metrics, Precision@K table |
| 📊 **Data Explorer** | Fraud distribution, amount analysis, category breakdown |
| 🎯 **Model Performance** | AUC gauges, ROC/PR curves, threshold analysis |
| ⚡ **Live Scoring** | Interactive scoring, random samples, batch upload |
| 🏢 **Fraud Ops Simulator** | Team configuration, capacity planning, ROI analysis |
| 🔬 **Feature Analysis** | Feature importance, top predictors |
| 🎚️ **Threshold Tuning** | Precision/recall tradeoff, alert volume optimization |

### Screenshots

Launch with `make dashboard` and navigate to http://localhost:8501

---

## 📊 Metrics

### Model Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **AUC-ROC** | Discrimination across thresholds | >0.90 |
| **AUC-PR** | Performance on imbalanced data | >0.50 |
| **Precision@K** | Fraud rate in top-K alerts | Maximize |
| **Recall@K** | Fraud capture in top-K | >80% |

### Business Metrics (Expected Value Framework)

```
Expected Value = Σ P(fraud|score) × TxnAmount × RecoveryRate - ReviewCost
```

| Metric | Description |
|--------|-------------|
| **$ Saved** | Fraud prevented by interventions |
| **$ Lost** | Fraud missed (capacity constraints) |
| **Review Cost** | Analyst time × hourly rate |
| **Net Value** | Saved - Lost - Cost |

### Operational Metrics

| Metric | Description |
|--------|-------------|
| **Queue Depth** | Pending alerts |
| **Alert Latency** | Time to review |
| **Utilization** | Capacity usage % |
| **FPR** | False positive rate |

---

## 📁 Project Structure

```
fraud-decisioning-platform/
├── 📊 data/
│   ├── raw/                    # Kaggle data (gitignored)
│   ├── processed/              # Engineered features
│   └── sample/                 # Sample for cloud deploy
├── 📓 notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── 🤖 models/                  # Trained models
├── 📈 reports/                 # Drift reports
├── 🔧 src/
│   ├── config.py              # Configuration
│   ├── data/                  # Data processing
│   ├── features/              # Feature engineering
│   ├── model/
│   │   ├── train.py           # Basic training
│   │   ├── advanced_trainer.py # Ensemble + Optuna
│   │   └── predict.py         # Inference
│   ├── policy/                # Triage simulation
│   ├── service/
│   │   └── api.py             # FastAPI backend
│   ├── monitoring/            # Drift detection
│   └── dashboard/
│       └── app.py             # Streamlit dashboard
├── 🧪 tests/                   # Test suite
├── 🐳 Dockerfile              # Multi-stage build
├── 🐳 docker-compose.yml      # Service orchestration
├── 📋 Makefile                # Build commands
├── 📦 requirements.txt        # Dependencies
└── 📖 README.md               # This file
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Quick tests (no coverage)
make test-fast

# Linting
make lint

# Format code
make format
```

---

## 🔧 Configuration

Edit `src/config.py` to customize:

- File paths
- Model hyperparameters
- Risk tier thresholds
- Fraud-ops settings
- API configuration

---

## 📚 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Exploratory data analysis, fraud patterns |
| `02_feature_engineering.ipynb` | Feature creation, encoding strategies |
| `03_model_evaluation.ipynb` | Model training, metrics, SHAP analysis |

---

## 🚢 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set main file: `src/dashboard/app.py`
5. Deploy!

### Docker/Kubernetes

```bash
# Build production image
docker build -t fraud-api:latest --target production .

# Run container
docker run -p 8000:8000 fraud-api:latest
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) by Vesta Corporation
- Built with: LightGBM, XGBoost, FastAPI, Streamlit, SHAP, Optuna

---

<p align="center">
  <strong>Built for interview demonstrations</strong><br>
  <em>Showcasing production ML system design, end-to-end</em>
</p>
