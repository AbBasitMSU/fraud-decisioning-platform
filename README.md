# Fraud Decisioning Platform (FDP)

> Real-time fraud scoring + capacity-aware alert triage simulator with fraud-ops metrics and monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Problem Statement

Financial fraud costs institutions billions annually. This platform addresses two critical challenges:

1. **Real-time Fraud Scoring**: Sub-100ms prediction latency for transaction risk assessment
2. **Capacity-Aware Triage**: Fraud-ops teams have limited capacity—prioritize alerts that maximize expected value

### Dataset

Uses the [IEEE-CIS Fraud Detection (Vesta)](https://www.kaggle.com/c/ieee-fraud-detection) dataset:
- **590K+ transactions** with 400+ features
- **3.5% fraud rate** (highly imbalanced)
- Rich features: transaction amounts, card info, device fingerprints, time deltas

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRAUD DECISIONING PLATFORM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Raw Data   │───▶│   Feature    │───▶│    Model     │               │
│  │  (Kaggle)    │    │  Engineering │    │   Training   │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Monitoring  │◀───│   Scoring    │◀───│   LightGBM   │               │
│  │  (Evidently) │    │   Service    │    │    Model     │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                                            │
│         ▼                   ▼                                            │
│  ┌──────────────┐    ┌──────────────┐                                   │
│  │    Drift     │    │   Policy     │                                   │
│  │   Reports    │    │  Simulator   │                                   │
│  └──────────────┘    └──────────────┘                                   │
│                             │                                            │
│                             ▼                                            │
│                      ┌──────────────┐                                   │
│                      │  Fraud-Ops   │                                   │
│                      │   Metrics    │                                   │
│                      └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| `src/data/` | Data loading, cleaning, train/test splitting |
| `src/features/` | Feature engineering pipeline (aggregations, encoding) |
| `src/model/` | LightGBM training, evaluation, SHAP explanations |
| `src/policy/` | Capacity-aware triage simulation (queue management) |
| `src/service/` | FastAPI real-time scoring endpoint |
| `src/monitoring/` | Evidently drift detection & reporting |

---

## Metrics

### Model Performance

| Metric | Description | Target |
|--------|-------------|--------|
| **AUC-ROC** | Discrimination ability across all thresholds | > 0.90 |
| **Precision@K** | Precision in top-K riskiest transactions | Maximize |
| **Recall@K** | Fraud capture rate in top-K alerts | > 80% |

### Business Metrics (Expected Value Framework)

```
Expected Value = Σ P(fraud|score) × TxnAmount × RecoveryRate - ReviewCost
```

| Metric | Description |
|--------|-------------|
| **$ Saved** | Fraud losses prevented by interventions |
| **$ Lost** | Fraud that slipped through (capacity constraints) |
| **Review Cost** | Analyst time × hourly rate × cases reviewed |
| **Net Value** | $ Saved - $ Lost - Review Cost |

### Operational Metrics

| Metric | Description |
|--------|-------------|
| **Queue Depth** | Pending alerts awaiting review |
| **Alert Latency** | Time from transaction to review |
| **Analyst Utilization** | % of capacity used |
| **False Positive Rate** | Non-fraud alerts reviewed |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)

### Installation

```bash
# Clone repository
git clone https://github.com/yourorg/fraud-decisioning-platform.git
cd fraud-decisioning-platform

# Setup environment
make setup

# Activate virtual environment
source .venv/bin/activate
```

### Data Pipeline

```bash
# Download IEEE-CIS dataset from Kaggle
make download_data

# Process raw data into features
make make_dataset
```

### Model Training

```bash
# Train LightGBM model
make train
```

### Running the Service

```bash
# Start FastAPI scoring service (development)
make serve

# Production mode (4 workers)
make serve_prod
```

### Simulation & Monitoring

```bash
# Run fraud-ops triage simulation
make simulate

# Generate drift report
make drift_report
```

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Score Transaction

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": "12345",
    "TransactionAmt": 150.00,
    "card1": 1234,
    "card2": 567.0,
    "addr1": 123.0,
    "P_emaildomain": "gmail.com",
    "DeviceType": "mobile"
  }'
```

**Response:**

```json
{
  "transaction_id": "12345",
  "fraud_probability": 0.0342,
  "risk_tier": "LOW",
  "recommended_action": "APPROVE",
  "latency_ms": 12.4
}
```

### Batch Scoring

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

---

## Demo Walkthrough

### 1. Explore the Data

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train & Evaluate

```bash
make train
# Outputs: AUC, Precision@K, Recall@K, confusion matrix
```

### 3. Run Simulation

```bash
make simulate
# Simulates 24-hour fraud-ops with:
# - 10 analysts, 8-hour shifts
# - 15-min review time per alert
# - Priority queue by expected value
```

### 4. Check Drift

```bash
make drift_report
# Generates HTML report in reports/
```

### 5. Live Scoring

```bash
make serve
# Open http://localhost:8000/docs for Swagger UI
```

### 6. Launch Interview Dashboard

```bash
make dashboard
# Open http://localhost:8501 for Streamlit dashboard
```

---

## 🎨 Streamlit Dashboard

The platform includes a comprehensive **interview-ready dashboard** with:

| Page | Features |
|------|----------|
| 🏠 **Overview** | Architecture diagram, problem statement, key metrics |
| 📊 **Data Exploration** | Target analysis, amount distributions, missing values |
| 🔧 **Feature Engineering** | Feature categories, transformations, encoding strategies |
| 🤖 **Model Performance** | AUC gauges, ROC/PR curves, feature importance, SHAP |
| 🎯 **Live Scoring Demo** | Interactive transaction scoring, batch upload |
| 🏢 **Fraud-Ops Simulation** | Capacity configuration, ROI analysis, sensitivity |
| 📈 **Monitoring** | Drift detection, performance trends, alerts |

### Screenshots

Launch with `make dashboard` and navigate to `http://localhost:8501`

---

## Project Structure

```
fraud-decisioning-platform/
├── data/
│   ├── raw/                 # Original Kaggle data (gitignored)
│   └── processed/           # Engineered features (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── reports/
│   └── drift_report.html
├── src/
│   ├── __init__.py
│   ├── config.py            # Paths, constants, hyperparams
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── policy/
│   │   ├── __init__.py
│   │   └── simulate.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_report.py
│   └── dashboard/
│       ├── __init__.py
│       └── app.py              # Streamlit dashboard
├── tests/
│   └── ...
├── .gitignore
├── Makefile
├── README.md
└── requirements.txt
```

---

## Configuration

Edit `src/config.py` to customize:

- File paths
- Model hyperparameters
- Triage policy settings
- API configuration

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) by Vesta Corporation
- Built with: LightGBM, FastAPI, Evidently, SHAP
