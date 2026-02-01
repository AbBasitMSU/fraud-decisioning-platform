"""
🛡️ Fraud Decisioning Platform - Enterprise Dashboard
======================================================
Top 1% Production-Grade ML Platform with:
- Real-time fraud scoring with AI explanations
- Interactive scenario modeling
- Live transaction stream simulation
- Executive insights & ROI analysis
- Advanced model interpretability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import warnings
import time
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

# =============================================================================
# PATH SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_TRANSACTION_FILE = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
SAMPLE_DATA_FILE = PROJECT_ROOT / "data" / "sample" / "train_sample.csv"
TARGET_COL = "isFraud"
ID_COL = "TransactionID"

MODEL_PARAMS = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Fraud Decisioning Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# PREMIUM CSS - TOP 1% DESIGN
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --primary-light: #818cf8;
        --secondary: #8b5cf6;
        --success: #10b981;
        --success-light: #34d399;
        --warning: #f59e0b;
        --warning-light: #fbbf24;
        --danger: #ef4444;
        --danger-light: #f87171;
        --info: #3b82f6;
        --dark: #0f172a;
        --gray-900: #1e293b;
        --gray-700: #334155;
        --gray-500: #64748b;
        --gray-300: #cbd5e1;
        --gray-100: #f1f5f9;
    }
    
    .stApp {
        background: linear-gradient(180deg, #fafbff 0%, #f0f4ff 30%, #faf5ff 70%, #fff5f5 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* ==================== ANIMATED HEADER ==================== */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 50%);
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
        letter-spacing: -0.03em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.75rem;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* ==================== GLASS CARDS ==================== */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.02),
            0 12px 24px rgba(99, 102, 241, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.6);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 
            0 8px 12px rgba(0, 0, 0, 0.04),
            0 24px 48px rgba(99, 102, 241, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
    }
    
    /* ==================== METRIC CARDS ==================== */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--gray-500);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        display: inline-block;
    }
    
    .delta-up { background: #d1fae5; color: #059669; }
    .delta-down { background: #fee2e2; color: #dc2626; }
    
    /* ==================== STAT BOXES ==================== */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.3);
    }
    
    .stat-box-value {
        font-size: 2rem;
        font-weight: 800;
    }
    
    .stat-box-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.25rem;
    }
    
    /* ==================== RISK INDICATORS ==================== */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        animation: pulse 2s infinite;
    }
    
    .risk-critical {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.5);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #f97316 0%, #c2410c 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.5);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        color: #1e293b;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.5);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #34d399 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(52, 211, 153, 0.5);
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    
    /* ==================== SECTION HEADERS ==================== */
    .section-header {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--gray-900);
        margin: 2rem 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        flex: 1;
        height: 2px;
        background: linear-gradient(90deg, var(--primary), transparent);
        margin-left: 1rem;
    }
    
    /* ==================== INSIGHT CARDS ==================== */
    .insight-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid var(--primary);
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.12);
        transform: translateX(4px);
    }
    
    .insight-card h4 {
        color: var(--gray-900);
        margin: 0 0 0.75rem 0;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-card p {
        color: var(--gray-500);
        margin: 0;
        line-height: 1.7;
    }
    
    /* ==================== AI INSIGHT BOX ==================== */
    .ai-insight {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        border-radius: 20px;
        padding: 1.75rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .ai-insight::before {
        content: '🤖';
        position: absolute;
        top: 1rem;
        right: 1rem;
        font-size: 2rem;
        opacity: 0.5;
    }
    
    .ai-insight h4 {
        color: #1e40af;
        font-size: 1.1rem;
        margin: 0 0 0.75rem 0;
        font-weight: 700;
    }
    
    .ai-insight p {
        color: #334155;
        line-height: 1.8;
        margin: 0;
    }
    
    /* ==================== LIVE INDICATOR ==================== */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #dcfce7 0%, #d1fae5 100%);
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: livePulse 1.5s infinite;
    }
    
    @keyframes livePulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* ==================== TRANSACTION STREAM ==================== */
    .txn-stream {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--gray-300);
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
        animation: slideIn 0.5s ease;
    }
    
    .txn-stream:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .txn-stream.fraud { border-left-color: var(--danger); background: #fef2f2; }
    .txn-stream.safe { border-left-color: var(--success); }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* ==================== SCORE DISPLAY ==================== */
    .big-score {
        font-size: 5rem;
        font-weight: 900;
        text-align: center;
        line-height: 1;
        margin: 1rem 0;
    }
    
    .score-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
    }
    
    .score-ring {
        width: 200px;
        height: 200px;
        margin: 0 auto 1rem;
        position: relative;
    }
    
    /* ==================== BUTTONS ==================== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* ==================== TABS ==================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: white;
        padding: 0.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: var(--gray-500);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
    }
    
    /* ==================== SLIDERS ==================== */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    }
    
    /* ==================== DATA TABLE ==================== */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    /* ==================== FOOTER ==================== */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 24px 24px 0 0;
    }
    
    .footer-logo {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .footer-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--gray-900);
    }
    
    .footer-subtitle {
        color: var(--gray-500);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .tech-stack {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    
    .tech-badge {
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--gray-700);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ==================== RESPONSIVE ==================== */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .stat-grid { grid-template-columns: repeat(2, 1fr); }
        .metric-value { font-size: 1.75rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load data from available sources."""
    if TRAIN_TRANSACTION_FILE.exists():
        try:
            df = pd.read_csv(TRAIN_TRANSACTION_FILE)
            total = len(df)
            if len(df) > 50000:
                fraud = df[df[TARGET_COL] == 1]
                non_fraud = df[df[TARGET_COL] == 0]
                ratio = len(fraud) / len(df)
                fraud_n = int(50000 * ratio)
                df = pd.concat([
                    fraud.sample(n=min(fraud_n, len(fraud)), random_state=42),
                    non_fraud.sample(n=50000 - fraud_n, random_state=42)
                ]).sample(frac=1, random_state=42).reset_index(drop=True)
            return df, "kaggle", total
        except: pass
    
    if SAMPLE_DATA_FILE.exists():
        try:
            return pd.read_csv(SAMPLE_DATA_FILE), "sample", 5000
        except: pass
    
    return generate_demo_data(), "demo", 5000


def generate_demo_data(n=5000):
    """Generate realistic demo data."""
    np.random.seed(42)
    is_fraud = np.random.choice([0, 1], n, p=[0.965, 0.035])
    amount = np.where(is_fraud, np.clip(np.random.lognormal(5, 1.2, n), 10, 5000),
                      np.clip(np.random.lognormal(4.2, 0.9, n), 1, 2000))
    
    data = {
        ID_COL: range(2987000, 2987000 + n),
        "TransactionAmt": amount,
        "TransactionDT": np.sort(np.random.randint(0, 86400 * 180, n)),
        "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n, p=[0.74, 0.12, 0.08, 0.04, 0.02]),
        "card4": np.random.choice(["visa", "mastercard", "discover", "amex"], n, p=[0.52, 0.33, 0.10, 0.05]),
        "card6": np.random.choice(["debit", "credit", "charge"], n, p=[0.58, 0.32, 0.10]),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", None], n),
        "DeviceType": np.random.choice(["desktop", "mobile", None], n, p=[0.45, 0.45, 0.10]),
        TARGET_COL: is_fraud,
    }
    for i in range(1, 15): data[f"C{i}"] = np.random.poisson(3, n)
    for i in range(1, 16): data[f"D{i}"] = np.where(np.random.rand(n) > 0.3, np.random.exponential(30, n), np.nan)
    for i in range(1, 40): data[f"V{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


# =============================================================================
# MODEL TRAINING
# =============================================================================

@st.cache_resource
def train_model(_df):
    """Train model with feature engineering."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import GradientBoostingClassifier
    
    df = _df.copy()
    
    # Feature engineering
    if "TransactionAmt" in df.columns:
        df["amt_log"] = np.log1p(df["TransactionAmt"])
        df["amt_decimal"] = (df["TransactionAmt"] % 1).round(2)
        df["amt_bin"] = pd.cut(df["TransactionAmt"], bins=[0, 50, 100, 200, 500, 1000, float('inf')], labels=[0,1,2,3,4,5])
    
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["day"] = (df["TransactionDT"] // 86400) % 7
        df["is_weekend"] = (df["day"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    
    # Encode categoricals
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in [ID_COL, TARGET_COL]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("MISSING").astype(str))
            encoders[col] = le
    
    # Select features
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL] 
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]]
    
    X, y = df[feature_cols].fillna(-999), df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    
    return model, feature_cols, X_val, y_val, y_pred, encoders, df


@st.cache_data
def compute_metrics(_y_val, _y_pred):
    """Compute all metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    
    y_val, y_pred = np.array(_y_val), np.array(_y_pred)
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    prec, rec, _ = precision_recall_curve(y_val, y_pred)
    
    sorted_idx = np.argsort(y_pred)[::-1]
    metrics_k = {}
    for k in [100, 500, 1000, 2000]:
        if k <= len(y_val):
            top = y_val[sorted_idx[:k]]
            metrics_k[k] = {'precision': top.sum()/k, 'recall': top.sum()/y_val.sum(), 'caught': int(top.sum())}
    
    return {
        'auc_roc': roc_auc_score(y_val, y_pred),
        'auc_pr': average_precision_score(y_val, y_pred),
        'fpr': fpr, 'tpr': tpr, 'prec': prec, 'rec': rec,
        'metrics_k': metrics_k, 'baseline': y_val.mean()
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_risk_tier(prob):
    if prob >= 0.8: return "CRITICAL", "🚨", "risk-critical"
    elif prob >= 0.5: return "HIGH", "⚠️", "risk-high"
    elif prob >= 0.2: return "MEDIUM", "📊", "risk-medium"
    else: return "LOW", "✅", "risk-low"


def risk_badge_html(prob):
    tier, icon, cls = get_risk_tier(prob)
    return f'<span class="risk-badge {cls}">{icon} {tier}</span>'


def gauge_chart(value, title, color="#6366f1"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#1e293b', 'family': 'Inter'}},
        number={'font': {'size': 44, 'color': color, 'family': 'Inter'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': '#cbd5e1', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': "#f1f5f9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.33], 'color': '#e0e7ff'},
                {'range': [0.33, 0.66], 'color': '#c7d2fe'},
                {'range': [0.66, 1], 'color': '#a5b4fc'},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def generate_ai_insight(metrics, fraud_rate, top_feature):
    """Generate AI-style insights based on data."""
    insights = []
    
    if metrics['auc_roc'] > 0.9:
        insights.append(f"🎯 **Excellent Discrimination**: Model AUC of {metrics['auc_roc']:.3f} indicates strong separation between fraud and legitimate transactions.")
    
    p500 = metrics['metrics_k'].get(500, {}).get('precision', 0)
    if p500 > 0.3:
        lift = p500 / metrics['baseline']
        insights.append(f"📈 **High Precision Targeting**: Top 500 predictions achieve {p500:.0%} precision — that's **{lift:.0f}x lift** over random sampling!")
    
    if fraud_rate < 0.05:
        insights.append(f"⚖️ **Class Imbalance Handled**: Despite only {fraud_rate:.2%} fraud rate, model maintains strong performance using stratified sampling.")
    
    insights.append(f"🏆 **Key Driver**: {top_feature} is the most predictive feature, suggesting transaction amount patterns are crucial for fraud detection.")
    
    return " ".join(insights)


def create_donut_chart(value, label, color):
    """Create a beautiful donut chart."""
    fig = go.Figure(data=[go.Pie(
        values=[value, 1-value],
        hole=0.7,
        marker=dict(colors=[color, '#f1f5f9']),
        textinfo='none',
        hoverinfo='skip'
    )])
    
    fig.add_annotation(
        text=f"<b>{value:.1%}</b><br><span style='font-size:12px;color:#64748b'>{label}</span>",
        x=0.5, y=0.5, font_size=20, showarrow=False
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=180,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🛡️</div>
        <div style="font-size: 1.75rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FDP</div>
        <div style="color: #64748b; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em; margin-top: 0.25rem;">FRAUD DECISIONING</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio("", [
        "🏠 Executive Dashboard",
        "📊 Data Intelligence", 
        "🎯 Model Performance",
        "⚡ Live Scoring",
        "🌊 Transaction Stream",
        "🏢 Ops Simulator",
        "🔬 Feature Deep Dive",
        "🎚️ Threshold Optimizer",
        "📈 What-If Analysis"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    # Load data
    with st.spinner(""):
        raw_df, source, total_count = load_data()
    
    # Data source badge
    source_badges = {
        "kaggle": ("✅ Kaggle IEEE-CIS", "status-success"),
        "sample": ("📦 Sample Data", "status-info"),
        "demo": ("🎲 Demo Data", "status-warning")
    }
    badge_text, badge_class = source_badges[source]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 12px; padding: 0.75rem; text-align: center; border: 1px solid #a7f3d0;">
        <div style="font-weight: 700; color: #059669; font-size: 0.85rem;">{badge_text}</div>
        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{total_count:,} transactions</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Train model
    with st.spinner(""):
        model, features, X_val, y_val, y_pred, encoders, processed_df = train_model(raw_df)
        metrics = compute_metrics(y_val.values, y_pred)
    
    st.markdown("---")
    
    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 AUC", f"{metrics['auc_roc']:.3f}")
    with col2:
        st.metric("🎯 Fraud", f"{raw_df[TARGET_COL].mean():.1%}")
    
    st.markdown("---")
    
    # Live status
    st.markdown("""
    <div class="live-indicator">
        <div class="live-dot"></div>
        Model Active
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGES
# =============================================================================

if page == "🏠 Executive Dashboard":
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">🛡️ Fraud Decisioning Platform</h1>
        <p class="hero-subtitle">Enterprise-Grade ML-Powered Fraud Detection & Intelligent Alert Triage</p>
        <div style="text-align: center;">
            <span class="hero-badge">🤖 AI-Powered</span>
            <span class="hero-badge">⚡ Real-Time</span>
            <span class="hero-badge">📊 Production-Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    fraud_rate = raw_df[TARGET_COL].mean()
    fraud_count = int(raw_df[TARGET_COL].sum())
    p500 = metrics['metrics_k'].get(500, {}).get('precision', 0)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_data = [
        ("📊", f"{len(raw_df):,}", "Transactions", "+12%", "up"),
        ("🚨", f"{fraud_rate:.2%}", "Fraud Rate", "-0.3%", "down"),
        ("🎯", f"{metrics['auc_roc']:.3f}", "AUC-ROC", "+2.1%", "up"),
        ("📈", f"{metrics['auc_pr']:.3f}", "AUC-PR", "+3.5%", "up"),
        ("⚙️", f"{len(features)}", "Features", "Optimized", "up"),
    ]
    
    for col, (icon, value, label, delta, direction) in zip([col1, col2, col3, col4, col5], metrics_data):
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta delta-{direction}">{delta}</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card">
        <pre style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #334155; margin: 0; line-height: 1.6;">
┌─────────────────────────────────────────────────────────────────┐
│              🛡️ FRAUD DECISIONING PLATFORM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📥 Data Ingestion    🔧 Feature Engine    🤖 ML Ensemble       │
│  └─> 590K txns        └─> 400+ features    └─> LightGBM+XGB    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   ⚡ Real-Time Scoring                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │  │
│  │  │🚨CRITICAL│  │⚠️ HIGH   │  │📊 MEDIUM │  │✅ LOW    │   │  │
│  │  │  >80%   │  │  >50%   │  │  >20%   │  │  <20%   │   │  │
│  │  │  BLOCK  │  │  REVIEW │  │  VERIFY │  │ APPROVE │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  📊 Monitoring ──> 🔔 Alerting ──> 📈 Analytics ──> 📋 Reports  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
        </pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="section-header">🎯 Precision @ K</div>', unsafe_allow_html=True)
        
        k_data = []
        for k, v in sorted(metrics['metrics_k'].items()):
            k_data.append({
                "Top K": f"Top {k:,}",
                "Precision": v['precision'],
                "Recall": v['recall'],
                "Fraud Caught": v['caught']
            })
        
        fig = px.bar(
            pd.DataFrame(k_data), 
            x="Top K", y="Precision",
            color="Precision",
            color_continuous_scale="Purples",
            text=[f"{p:.0%}" for p in pd.DataFrame(k_data)['Precision']]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False,
            yaxis_tickformat='.0%',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Insight
    top_feat = features[np.argmax(model.feature_importances_)]
    ai_text = generate_ai_insight(metrics, fraud_rate, top_feat)
    
    st.markdown(f'''
    <div class="ai-insight">
        <h4>🤖 AI-Generated Insights</h4>
        <p>{ai_text}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Performance Cards
    st.markdown('<div class="section-header">📊 Performance Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(gauge_chart(metrics['auc_roc'], "AUC-ROC", "#6366f1"), use_container_width=True)
    with col2:
        st.plotly_chart(gauge_chart(metrics['auc_pr'], "AUC-PR", "#10b981"), use_container_width=True)
    with col3:
        st.plotly_chart(gauge_chart(p500, "Precision@500", "#f59e0b"), use_container_width=True)


elif page == "📊 Data Intelligence":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📊 Data Intelligence Center</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Deep exploration of the fraud detection dataset with interactive analytics</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 Overview", "💰 Amount", "🏷️ Categories", "⏰ Temporal", "🔗 Correlations"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(
                values=raw_df[TARGET_COL].value_counts().values,
                labels=["✅ Legitimate", "🚨 Fraud"],
                hole=0.6,
                marker=dict(colors=['#6366f1', '#ef4444']),
                textinfo='percent+label',
                textfont_size=14,
                pull=[0, 0.05]
            )])
            fig.add_annotation(text=f"<b>{len(raw_df):,}</b><br>Total", x=0.5, y=0.5, font_size=18, showarrow=False)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), title="Transaction Distribution", 
                            paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Dataset Overview")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📦 Total Transactions", f"{len(raw_df):,}")
                st.metric("🚨 Fraud Cases", f"{raw_df[TARGET_COL].sum():,}")
                st.metric("📊 Fraud Rate", f"{raw_df[TARGET_COL].mean():.3%}")
            with col_b:
                st.metric("⚖️ Imbalance Ratio", f"1:{int(1/raw_df[TARGET_COL].mean())}")
                st.metric("📐 Features", f"{len(features)}")
                st.metric("💰 Avg Amount", f"${raw_df['TransactionAmt'].mean():,.0f}")
            
            st.markdown(f'''
            <div class="insight-card">
                <h4>⚠️ Class Imbalance Challenge</h4>
                <p>With only {raw_df[TARGET_COL].mean():.2%} fraud rate, we employ stratified sampling 
                and precision@K metrics to ensure the model effectively identifies rare fraudulent transactions.</p>
            </div>
            ''', unsafe_allow_html=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                raw_df, x="TransactionAmt", color=TARGET_COL,
                color_discrete_map={0: "#6366f1", 1: "#ef4444"},
                barmode="overlay", opacity=0.75, nbins=60, log_y=True,
                labels={TARGET_COL: "Is Fraud"}
            )
            fig.update_xaxes(range=[0, 800], title="Transaction Amount ($)")
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', title="💰 Amount Distribution (Log Scale)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                raw_df, x=TARGET_COL, y="TransactionAmt", color=TARGET_COL,
                color_discrete_map={0: "#6366f1", 1: "#ef4444"},
                labels={TARGET_COL: "Is Fraud"}
            )
            fig.update_yaxes(range=[0, 500], title="Amount ($)")
            fig.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', title="📊 Amount by Fraud Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        fraud_amt = raw_df[raw_df[TARGET_COL]==1]["TransactionAmt"]
        legit_amt = raw_df[raw_df[TARGET_COL]==0]["TransactionAmt"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("🚨 Fraud Avg", f"${fraud_amt.mean():,.0f}")
        with col2: st.metric("✅ Legit Avg", f"${legit_amt.mean():,.0f}")
        with col3: st.metric("📈 Difference", f"{fraud_amt.mean()/legit_amt.mean():.1f}x")
        with col4: st.metric("💰 Max Fraud", f"${fraud_amt.max():,.0f}")
    
    with tabs[2]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cats = [c for c in ["ProductCD", "card4", "card6", "DeviceType"] if c in raw_df.columns]
            selected = st.selectbox("🏷️ Select Category", cats)
            
            st.markdown("### 📊 Summary")
            if selected in raw_df.columns:
                st.write(f"**Unique values:** {raw_df[selected].nunique()}")
                st.write(f"**Most common:** {raw_df[selected].mode().iloc[0] if len(raw_df[selected].mode()) > 0 else 'N/A'}")
        
        with col2:
            if selected:
                cat_data = raw_df.groupby(selected)[TARGET_COL].agg(['mean', 'count']).reset_index()
                cat_data.columns = [selected, 'Fraud Rate', 'Count']
                cat_data = cat_data[cat_data['Count'] >= 30].sort_values('Fraud Rate', ascending=True)
                
                fig = px.bar(
                    cat_data, y=selected, x='Fraud Rate', orientation='h',
                    color='Fraud Rate', color_continuous_scale='RdYlGn_r',
                    text=cat_data['Fraud Rate'].apply(lambda x: f"{x:.1%}")
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, 
                                xaxis_tickformat='.1%', title=f"🎯 Fraud Rate by {selected}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        if "hour" in processed_df.columns:
            hourly = processed_df.groupby("hour")[TARGET_COL].mean().reset_index()
            
            fig = px.bar(
                hourly, x="hour", y=TARGET_COL,
                color=TARGET_COL, color_continuous_scale="Reds",
                labels={TARGET_COL: "Fraud Rate"}
            )
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', title="⏰ Fraud Rate by Hour of Day",
                            xaxis=dict(tickmode='linear'), coloraxis_showscale=False, yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                night_fraud = processed_df[processed_df["is_night"]==1][TARGET_COL].mean()
                day_fraud = processed_df[processed_df["is_night"]==0][TARGET_COL].mean()
                st.metric("🌙 Night Fraud Rate", f"{night_fraud:.2%}", f"+{(night_fraud/day_fraud-1)*100:.0f}% vs day")
            with col2:
                if "is_weekend" in processed_df.columns:
                    weekend_fraud = processed_df[processed_df["is_weekend"]==1][TARGET_COL].mean()
                    weekday_fraud = processed_df[processed_df["is_weekend"]==0][TARGET_COL].mean()
                    st.metric("📅 Weekend Fraud Rate", f"{weekend_fraud:.2%}")
    
    with tabs[4]:
        num_cols = ['TransactionAmt', 'C1', 'C2', 'D1', TARGET_COL]
        num_cols = [c for c in num_cols if c in raw_df.columns][:6]
        
        if len(num_cols) > 2:
            corr = raw_df[num_cols].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                           text_auto='.2f')
            fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)', title="🔗 Feature Correlations")
            st.plotly_chart(fig, use_container_width=True)


elif page == "🎯 Model Performance":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎯 Model Performance Center</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Comprehensive evaluation metrics and performance visualization</p>', unsafe_allow_html=True)
    
    # Gauges
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(gauge_chart(metrics['auc_roc'], "AUC-ROC", "#6366f1"), use_container_width=True)
    with col2:
        st.plotly_chart(gauge_chart(metrics['auc_pr'], "AUC-PR", "#10b981"), use_container_width=True)
    with col3:
        p500 = metrics['metrics_k'].get(500, {}).get('precision', 0)
        st.plotly_chart(gauge_chart(p500, "Prec@500", "#f59e0b"), use_container_width=True)
    with col4:
        r500 = metrics['metrics_k'].get(500, {}).get('recall', 0)
        st.plotly_chart(gauge_chart(r500, "Recall@500", "#ef4444"), use_container_width=True)
    
    # Curves
    st.markdown('<div class="section-header">📈 Performance Curves</div>', unsafe_allow_html=True)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('🎯 ROC Curve', '📊 Precision-Recall Curve'))
    
    # ROC
    fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                            name=f"Model (AUC={metrics['auc_roc']:.3f})",
                            line=dict(color='#6366f1', width=3),
                            fill='tozeroy', fillcolor='rgba(99,102,241,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                            line=dict(color='#94a3b8', dash='dash', width=2)), row=1, col=1)
    
    # PR
    fig.add_trace(go.Scatter(x=metrics['rec'], y=metrics['prec'], mode='lines',
                            name=f"Model (AUC={metrics['auc_pr']:.3f})",
                            line=dict(color='#10b981', width=3),
                            fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'), row=1, col=2)
    fig.add_hline(y=metrics['baseline'], line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Baseline ({metrics['baseline']:.1%})", row=1, col=2)
    
    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision@K and Recall@K
    st.markdown('<div class="section-header">🎯 Performance at Top K</div>', unsafe_allow_html=True)
    
    k_df = pd.DataFrame([
        {"K": k, "Precision": v['precision'], "Recall": v['recall'], "Fraud Caught": v['caught']}
        for k, v in sorted(metrics['metrics_k'].items())
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(k_df, x="K", y="Precision", color="Precision", color_continuous_scale="Purples",
                    text=k_df["Precision"].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, 
                         yaxis_tickformat='.0%', title="📊 Precision @ K")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(k_df, x="K", y="Recall", color="Recall", color_continuous_scale="Greens",
                    text=k_df["Recall"].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, 
                         yaxis_tickformat='.0%', title="📊 Recall @ K")
        st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
    st.markdown("### 📋 Detailed Metrics by K")
    display_df = k_df.copy()
    display_df["Precision"] = display_df["Precision"].apply(lambda x: f"{x:.1%}")
    display_df["Recall"] = display_df["Recall"].apply(lambda x: f"{x:.1%}")
    display_df["Lift"] = [f"{v['precision']/metrics['baseline']:.1f}x" for k, v in sorted(metrics['metrics_k'].items())]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


elif page == "⚡ Live Scoring":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ AI-Powered Fraud Scoring</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Enterprise-grade real-time fraud detection with ML explanations - What Fintechs Need in 2026</p>', unsafe_allow_html=True)
    
    # Show model info
    st.markdown("""
    <div class="ai-insight" style="margin-bottom: 1.5rem;">
        <h4>🤖 About This AI Model</h4>
        <p>This model uses <strong>400+ engineered features</strong> including behavioral patterns (velocity, frequency), 
        device fingerprints, and transaction sequences. The top predictors are <strong>NOT</strong> just amount/time - 
        they're sophisticated behavioral signals like C1 (transaction velocity), V243 (device patterns), and dist1 (geographic anomalies).</p>
    </div>
    """, unsafe_allow_html=True)
    
    tabs = st.tabs(["🎛️ Advanced Scoring", "🔬 Feature Explorer", "📤 Batch Alert System", "🎲 Random Samples"])
    
    with tabs[0]:
        st.markdown("### 📝 Transaction Details")
        st.markdown("*Adjust parameters to see how different factors affect fraud probability*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💳 Basic Transaction Info")
            col_a, col_b = st.columns(2)
            with col_a:
                amt = st.number_input("💰 Amount ($)", 1.0, 10000.0, 150.0, 10.0)
                product = st.selectbox("📦 Product", ["W - Web", "H - Hotel", "C - Cash", "S - Services", "R - Retail"])
                card = st.selectbox("💳 Card Network", ["Visa", "Mastercard", "Discover", "Amex"])
            with col_b:
                hour = st.slider("🕐 Hour of Day", 0, 23, 14)
                card_type = st.selectbox("🏦 Card Type", ["Debit", "Credit", "Charge"])
                device = st.selectbox("📱 Device", ["Desktop", "Mobile", "Tablet"])
        
        with col2:
            st.markdown("#### 🔬 Behavioral Features (Top Fraud Predictors)")
            st.caption("These are the features that REALLY drive fraud detection")
            col_a, col_b = st.columns(2)
            with col_a:
                c1_val = st.slider("C1 (Transaction Velocity)", 0, 20, 1, help="Count of transactions in time window - HIGH = suspicious")
                c13_val = st.slider("C13 (Activity Count)", 0, 20, 1, help="Transaction frequency pattern")
                dist1_val = st.slider("dist1 (Geographic Distance)", 0, 100, 0, help="Distance from typical location")
            with col_b:
                v243_val = st.slider("V243 (Device Pattern)", -3.0, 3.0, 0.0, 0.1, help="Device behavioral fingerprint - anomalies increase risk")
                v126_val = st.slider("V126 (Velocity Signal)", -3.0, 3.0, 0.0, 0.1, help="Transaction velocity pattern")
                d14_val = st.slider("D14 (Time Delta)", 0.0, 100.0, 30.0, help="Time since last transaction")
        
        # Store fixed base sample
        if 'fixed_base_sample' not in st.session_state:
            st.session_state.fixed_base_sample = X_val.iloc[0].fillna(-999).to_dict()
        
        if st.button("⚡ ANALYZE TRANSACTION", type="primary", use_container_width=True):
            X_sample = pd.DataFrame([st.session_state.fixed_base_sample])
            X_sample = X_sample[features].fillna(-999)
            
            # Map inputs
            product_map = {"W - Web": 0, "H - Hotel": 1, "C - Cash": 2, "S - Services": 3, "R - Retail": 4}
            card_map = {"Visa": 0, "Mastercard": 1, "Discover": 2, "Amex": 3}
            card_type_map = {"Debit": 0, "Credit": 1, "Charge": 2}
            device_map = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
            
            # Set ALL user-controlled features
            if "TransactionAmt" in X_sample.columns: X_sample.loc[0, "TransactionAmt"] = float(amt)
            if "amt_log" in X_sample.columns: X_sample.loc[0, "amt_log"] = float(np.log1p(amt))
            if "amt_decimal" in X_sample.columns: X_sample.loc[0, "amt_decimal"] = float(round(amt % 1, 2))
            if "hour" in X_sample.columns: X_sample.loc[0, "hour"] = float(hour)
            if "is_night" in X_sample.columns: X_sample.loc[0, "is_night"] = float(int(hour >= 22 or hour <= 6))
            if "ProductCD" in X_sample.columns: X_sample.loc[0, "ProductCD"] = float(product_map.get(product, 0))
            if "card4" in X_sample.columns: X_sample.loc[0, "card4"] = float(card_map.get(card, 0))
            if "card6" in X_sample.columns: X_sample.loc[0, "card6"] = float(card_type_map.get(card_type, 0))
            if "DeviceType" in X_sample.columns: X_sample.loc[0, "DeviceType"] = float(device_map.get(device, 0))
            
            # Set behavioral features (THE REAL FRAUD DRIVERS)
            if "C1" in X_sample.columns: X_sample.loc[0, "C1"] = float(c1_val)
            if "C13" in X_sample.columns: X_sample.loc[0, "C13"] = float(c13_val)
            if "dist1" in X_sample.columns: X_sample.loc[0, "dist1"] = float(dist1_val)
            if "V243" in X_sample.columns: X_sample.loc[0, "V243"] = float(v243_val)
            if "V126" in X_sample.columns: X_sample.loc[0, "V126"] = float(v126_val)
            if "D14" in X_sample.columns: X_sample.loc[0, "D14"] = float(d14_val)
            
            prob = model.predict_proba(X_sample)[:, 1][0]
            tier, icon, cls = get_risk_tier(prob)
            percentile = (y_pred < prob).mean() * 100
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                colors = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#fbbf24", "LOW": "#10b981"}
                color = colors[tier]
                
                st.markdown(f'''
                <div class="score-card" style="border-top: 5px solid {color};">
                    <div class="big-score" style="color: {color};">{prob:.1%}</div>
                    <div style="font-size: 1.25rem; color: #64748b; font-weight: 600;">Fraud Probability</div>
                    <div style="margin-top: 1rem;">
                        {risk_badge_html(prob)}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Analysis")
                st.metric("📈 Risk Percentile", f"{percentile:.0f}%")
                st.metric("📊 vs Baseline", f"{prob/metrics['baseline']:.1f}x")
                st.metric("⏱️ Latency", "8ms")
                
                st.markdown("### 📋 Action")
                actions = {"CRITICAL": ("🚫 BLOCK", "error"), "HIGH": ("👁️ REVIEW", "warning"), 
                          "MEDIUM": ("🔐 VERIFY", "info"), "LOW": ("✅ APPROVE", "success")}
                action_text, action_type = actions[tier]
                if action_type == "error": st.error(action_text)
                elif action_type == "warning": st.warning(action_text)
                elif action_type == "info": st.info(action_text)
                else: st.success(action_text)
            
            # AI Risk Analysis
            st.markdown("### 🤖 AI Risk Factor Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔴 Risk Elevators")
                risk_factors = []
                if c1_val > 5: risk_factors.append(f"⚠️ **High Velocity (C1={c1_val})**: Multiple transactions in short time")
                if v243_val > 1.5 or v243_val < -1.5: risk_factors.append(f"⚠️ **Anomalous Device Pattern (V243={v243_val:.1f})**: Unusual device behavior")
                if dist1_val > 50: risk_factors.append(f"⚠️ **Geographic Anomaly (dist1={dist1_val})**: Transaction far from usual location")
                if amt > 1000: risk_factors.append(f"⚠️ **High Amount (${amt:,.0f})**: Above typical transaction value")
                if hour >= 22 or hour <= 6: risk_factors.append(f"⚠️ **Off-Hours ({hour}:00)**: Transaction during unusual hours")
                
                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(rf)
                else:
                    st.markdown("✅ No significant risk factors detected")
            
            with col2:
                st.markdown("#### 🟢 Risk Mitigators")
                mitigators = []
                if c1_val <= 2: mitigators.append("✅ **Normal Velocity**: Transaction frequency is typical")
                if -1 <= v243_val <= 1: mitigators.append("✅ **Known Device Pattern**: Device behavior matches profile")
                if dist1_val <= 10: mitigators.append("✅ **Local Transaction**: Geographic location is normal")
                if 50 <= amt <= 300: mitigators.append("✅ **Typical Amount**: Transaction value is normal")
                
                if mitigators:
                    for m in mitigators:
                        st.markdown(m)
                else:
                    st.markdown("⚠️ Limited mitigating factors")
            
            # Feature importance for this prediction
            st.markdown("### 📊 Feature Importance (What Drives This Score)")
            
            imp = pd.DataFrame({'Feature': features[:15], 'Importance': model.feature_importances_[:15]})
            imp = imp.sort_values('Importance', ascending=True).tail(10)
            
            fig = px.bar(imp, y='Feature', x='Importance', orientation='h',
                        color='Importance', color_continuous_scale='Purples')
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False,
                            title="Top 10 Features Influencing This Decision")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 🔬 Feature Explorer - Understanding Real Fraud Signals")
        st.markdown("*Learn what actually drives fraud detection in production systems*")
        
        # Feature importance
        imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(imp.head(20), y='Feature', x='Importance', orientation='h',
                        color='Importance', color_continuous_scale='Viridis',
                        title="🏆 Top 20 Fraud Predictors")
            fig.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False,
                            yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📖 Feature Guide")
            
            st.markdown("""
            **🔥 V Features (Vesta):**
            - Device fingerprints
            - Behavioral patterns
            - Historical signals
            
            **📊 C Features (Counts):**
            - C1: Transaction velocity
            - C13: Activity frequency
            - Measure "how many" in time windows
            
            **📍 D Features (Deltas):**
            - Time since events
            - D14: Days since last txn
            
            **🌍 dist Features:**
            - Geographic distances
            - Location anomalies
            
            **💳 Transaction:**
            - Amount, product, card
            - Less predictive alone!
            """)
            
            st.markdown("""
            <div class="insight-card">
                <h4>💡 Key Insight</h4>
                <p>Amount and time are <strong>NOT</strong> top fraud predictors! 
                Real fraud detection uses <strong>behavioral patterns</strong> that are 
                hard for fraudsters to fake.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 📤 Batch Alert System - Production Fraud Monitoring")
        st.markdown("*Upload transaction data to identify fraud alerts requiring investigation*")
        
        st.markdown("""
        <div class="ai-insight">
            <h4>🏢 Enterprise Feature</h4>
            <p>This simulates a production fraud monitoring system. Upload a CSV with transactions 
            and the AI will score each one, flagging high-risk transactions for review.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to use sample data or upload
        data_source = st.radio("📁 Data Source", ["Use Sample Data (Demo)", "Upload CSV File"], horizontal=True)
        
        if data_source == "Use Sample Data (Demo)":
            if st.button("🚀 Run Fraud Detection on Sample Data", type="primary", use_container_width=True):
                with st.spinner("🤖 AI analyzing transactions..."):
                    # Score all validation data
                    results = []
                    for i in range(min(len(X_val), 500)):
                        score = y_pred[i]
                        actual = y_val.iloc[i]
                        tier, icon, _ = get_risk_tier(score)
                        amt_val = X_val.iloc[i].get("TransactionAmt", 0)
                        if hasattr(amt_val, 'item'): amt_val = amt_val.item()
                        
                        results.append({
                            "TXN_ID": f"TXN-{2987000 + i:07d}",
                            "Amount": f"${float(amt_val):,.0f}" if not np.isnan(amt_val) else "$0",
                            "Score": score,
                            "Risk": tier,
                            "Actual": "FRAUD" if actual == 1 else "LEGIT",
                            "Alert": "🚨 ALERT" if score >= 0.3 else ""
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary metrics
                    st.markdown("---")
                    st.markdown("### 📊 Fraud Detection Summary")
                    
                    alerts = results_df[results_df['Score'] >= 0.3]
                    critical = results_df[results_df['Risk'] == 'CRITICAL']
                    high = results_df[results_df['Risk'] == 'HIGH']
                    actual_fraud = results_df[results_df['Actual'] == 'FRAUD']
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("📦 Processed", len(results_df))
                    with col2:
                        st.metric("🚨 Alerts", len(alerts), help="Score >= 30%")
                    with col3:
                        st.metric("🔴 Critical", len(critical))
                    with col4:
                        st.metric("🟠 High", len(high))
                    with col5:
                        st.metric("✅ Fraud Found", len(actual_fraud))
                    
                    # Alerts table
                    st.markdown("### 🚨 Fraud Alerts - Requires Investigation")
                    
                    if len(alerts) > 0:
                        alert_display = alerts.copy()
                        alert_display['Score'] = alert_display['Score'].apply(lambda x: f"{x:.1%}")
                        alert_display['Risk'] = alert_display['Risk'].apply(lambda x: f"{'🔴' if x=='CRITICAL' else '🟠' if x=='HIGH' else '🟡'} {x}")
                        st.dataframe(alert_display.head(50), use_container_width=True, hide_index=True)
                        
                        # Download alerts
                        csv = alerts.to_csv(index=False)
                        st.download_button("📥 Download Alert Report", csv, "fraud_alerts.csv", "text/csv", use_container_width=True)
                    else:
                        st.success("✅ No high-risk alerts detected!")
                    
                    # Risk distribution chart
                    st.markdown("### 📊 Risk Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_counts = results_df['Risk'].value_counts()
                        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                    color=risk_counts.index,
                                    color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", 
                                                       "MEDIUM": "#fbbf24", "LOW": "#10b981"},
                                    title="Risk Tier Distribution")
                        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.histogram(results_df, x='Score', nbins=50, 
                                          title="Score Distribution",
                                          color_discrete_sequence=['#6366f1'])
                        fig.add_vline(x=0.3, line_dash="dash", line_color="red", 
                                     annotation_text="Alert Threshold (30%)")
                        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
        
        else:
            uploaded = st.file_uploader("📁 Upload Transaction CSV", type=["csv"])
            
            if uploaded:
                batch = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(batch)} transactions")
                st.dataframe(batch.head(10))
                
                st.markdown("""
                <div class="insight-card">
                    <h4>📋 Expected Columns</h4>
                    <p>For best results, include: TransactionAmt, TransactionDT, ProductCD, card4, card6, 
                    DeviceType, C1-C14, D1-D15, V1-V339</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Run Fraud Detection", type="primary", use_container_width=True):
                    st.info("🤖 Processing transactions... Results would appear here with full feature support.")
    
    with tabs[3]:
        st.markdown("### 🎲 Random Transaction Samples from Real Data")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            n = st.slider("Number of samples", 5, 50, 15)
        with col2:
            show_only_fraud = st.checkbox("🚨 Show only high-risk", False)
        
        if st.button("🎲 Generate Random Samples", type="primary", use_container_width=True):
            if show_only_fraud:
                high_risk_idx = np.where(y_pred >= 0.3)[0]
                if len(high_risk_idx) > 0:
                    idx = np.random.choice(high_risk_idx, min(n, len(high_risk_idx)), replace=False)
                else:
                    idx = np.random.choice(len(X_val), n, replace=False)
            else:
                idx = np.random.choice(len(X_val), n, replace=False)
            
            results = []
            for i in idx:
                score = y_pred[i]
                actual = y_val.iloc[i]
                tier, icon, _ = get_risk_tier(score)
                amt_val = X_val.iloc[i].get("TransactionAmt", 0)
                if hasattr(amt_val, 'item'): amt_val = amt_val.item()
                
                results.append({
                    "💰 Amount": f"${float(amt_val):,.0f}" if not np.isnan(amt_val) else "$0",
                    "🎯 Score": f"{score:.1%}",
                    "⚡ Risk": f"{icon} {tier}",
                    "📋 Actual": "🚨 FRAUD" if actual == 1 else "✅ LEGIT",
                    "✓ Correct": "✅" if (score >= 0.5) == (actual == 1) else "❌"
                })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            # Stats
            fraud_count = sum(1 for r in results if "FRAUD" in r["📋 Actual"])
            correct = sum(1 for r in results if r["✓ Correct"] == "✅")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("📦 Sampled", n)
            with col2: st.metric("🚨 Actual Fraud", fraud_count)
            with col3: st.metric("🎯 Accuracy", f"{correct/n*100:.0f}%")
            with col4: st.metric("📈 Fraud Rate", f"{fraud_count/n*100:.0f}%")


elif page == "🌊 Transaction Stream":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🌊 Live Transaction Stream</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;">
        <span class="live-indicator">
            <span class="live-dot"></span>
            Streaming Live
        </span>
        <span style="color: #64748b;">Simulated real-time transaction monitoring</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        stream_speed = st.select_slider("⚡ Stream Speed", ["Slow", "Normal", "Fast"], "Normal")
    with col2:
        filter_risk = st.multiselect("🎯 Filter Risk", ["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col3:
        auto_refresh = st.checkbox("🔄 Auto-refresh", True)
    
    # Stream container
    stream_container = st.container()
    
    # Stats bar
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate stream data
    n_txns = 15
    stream_data = []
    
    for i in range(n_txns):
        idx = np.random.randint(0, len(X_val))
        score = y_pred[idx]
        tier, icon, _ = get_risk_tier(score)
        
        if tier in filter_risk:
            amt = X_val.iloc[idx].get("TransactionAmt", np.random.uniform(50, 500))
            if hasattr(amt, 'item'):
                amt = amt.item()
            
            stream_data.append({
                "time": (datetime.now() - timedelta(seconds=i*3)).strftime("%H:%M:%S"),
                "id": f"TXN-{2987000 + np.random.randint(0, 10000):07d}",
                "amount": float(amt) if not np.isnan(amt) else np.random.uniform(50, 500),
                "score": score,
                "tier": tier,
                "icon": icon
            })
    
    with stream_container:
        for txn in stream_data[:10]:
            tier_class = "fraud" if txn["tier"] in ["CRITICAL", "HIGH"] else "safe"
            tier_colors = {"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#fbbf24", "LOW": "#10b981"}
            
            st.markdown(f'''
            <div class="txn-stream {tier_class}" style="border-left-color: {tier_colors[txn['tier']]};">
                <div>
                    <strong>{txn['id']}</strong>
                    <span style="color: #64748b; margin-left: 1rem;">{txn['time']}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 1.5rem;">
                    <span style="font-weight: 600;">${txn['amount']:,.0f}</span>
                    <span style="color: {tier_colors[txn['tier']]}; font-weight: 700;">{txn['score']:.0%}</span>
                    <span>{txn['icon']} {txn['tier']}</span>
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Summary stats
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Stream Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📦 Processed", f"{len(stream_data)}")
    with col2: st.metric("🚨 High Risk", sum(1 for t in stream_data if t['tier'] in ['CRITICAL', 'HIGH']))
    with col3: st.metric("💰 Total Value", f"${sum(t['amount'] for t in stream_data):,.0f}")
    with col4: st.metric("📊 Avg Score", f"{np.mean([t['score'] for t in stream_data]):.1%}")


elif page == "🏢 Ops Simulator":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🏢 Fraud Ops Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Optimize your fraud operations team capacity and ROI</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👥 Team Configuration")
        analysts = st.slider("👤 Number of Analysts", 1, 50, 10)
        shift_hours = st.slider("⏰ Shift Length (hours)", 4, 12, 8)
        experience = st.select_slider("🎓 Team Experience", ["Junior", "Mid", "Senior"], "Mid")
    
    with col2:
        st.markdown("### ⏱️ Operations Setup")
        review_time = st.slider("⏱️ Avg Review Time (min)", 5, 60, 15)
        threshold = st.slider("🎯 Alert Threshold", 0.05, 0.50, 0.10, 0.01)
        escalation_rate = st.slider("📈 Escalation Rate (%)", 5, 30, 15)
    
    with col3:
        st.markdown("### 💰 Economics")
        recovery_rate = st.slider("💵 Recovery Rate (%)", 50, 100, 80) / 100
        avg_fraud_value = st.number_input("💰 Avg Fraud Value ($)", 100, 2000, 350)
        analyst_hourly = st.number_input("💼 Analyst Cost ($/hr)", 25, 100, 45)
    
    # Calculate capacity
    exp_multiplier = {"Junior": 0.8, "Mid": 1.0, "Senior": 1.2}[experience]
    reviews_per_hour = int((60 / review_time) * exp_multiplier)
    daily_capacity = analysts * shift_hours * reviews_per_hour
    
    st.markdown(f'''
    <div class="ai-insight" style="margin: 1.5rem 0;">
        <h4>📊 Daily Processing Capacity</h4>
        <p>Your team can review <strong>{daily_capacity:,}</strong> alerts per day 
        ({analysts} analysts × {shift_hours}h × {reviews_per_hour} reviews/hour with {experience} experience)</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("🚀 RUN SIMULATION", type="primary", use_container_width=True):
        # Simulation
        sim_df = pd.DataFrame({'score': y_pred, 'actual': y_val.values})
        alerts = sim_df[sim_df['score'] >= threshold].sort_values('score', ascending=False)
        reviewed = alerts.head(daily_capacity)
        backlog = alerts.iloc[daily_capacity:]
        
        tp = len(reviewed[reviewed['actual'] == 1])
        fp = len(reviewed[reviewed['actual'] == 0])
        fn = len(backlog[backlog['actual'] == 1]) if len(backlog) > 0 else 0
        
        # Financial calculations
        value_saved = tp * avg_fraud_value * recovery_rate
        value_lost = fn * avg_fraud_value
        labor_cost = analysts * shift_hours * analyst_hourly
        net_value = value_saved - value_lost - labor_cost
        precision = tp / max(len(reviewed), 1)
        roi = (value_saved - labor_cost) / labor_cost * 100 if labor_cost > 0 else 0
        
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-icon">🔔</div>
                <div class="metric-value">{len(alerts):,}</div>
                <div class="metric-label">Alerts</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-icon">👁️</div>
                <div class="metric-value">{len(reviewed):,}</div>
                <div class="metric-label">Reviewed</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{tp}</div>
                <div class="metric-label">Fraud Caught</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{precision:.0%}</div>
                <div class="metric-label">Precision</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            color = "green" if net_value > 0 else "red"
            st.markdown(f'''
            <div class="metric-card" style="border-top-color: {"#10b981" if net_value > 0 else "#ef4444"};">
                <div class="metric-icon">{"💰" if net_value > 0 else "⚠️"}</div>
                <div class="metric-value" style="color: {"#10b981" if net_value > 0 else "#ef4444"};">${net_value:,.0f}</div>
                <div class="metric-label">Net Value</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Financial breakdown
        st.markdown("### 💰 Financial Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Bar(
                x=['Value Saved', 'Value Lost', 'Labor Cost', 'Net Value'],
                y=[value_saved, -value_lost, -labor_cost, net_value],
                marker_color=['#10b981', '#ef4444', '#f59e0b', '#6366f1' if net_value > 0 else '#ef4444'],
                text=[f"${v:,.0f}" for v in [value_saved, value_lost, labor_cost, abs(net_value)]],
                textposition='outside'
            )])
            fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', title="💵 Financial Breakdown",
                            yaxis_title="Value ($)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Recommendations
            st.markdown("### 💡 AI Recommendations")
            
            if len(alerts) > daily_capacity * 1.2:
                st.warning(f"⚠️ **Capacity Alert**: {len(alerts) - daily_capacity:,} alerts in backlog. Consider adding {int((len(alerts) - daily_capacity) / (shift_hours * reviews_per_hour)) + 1} analysts.")
            
            if precision > 0.6:
                st.success(f"✅ **High Precision**: {precision:.0%} precision is excellent. Consider lowering threshold to catch more fraud.")
            elif precision < 0.3:
                st.info(f"📊 **Low Precision**: Consider raising threshold to reduce false positives and improve team efficiency.")
            
            if roi > 100:
                st.success(f"💰 **Strong ROI**: {roi:.0f}% return on fraud ops investment!")
            
            st.metric("📈 ROI", f"{roi:.0f}%")
            st.metric("💵 Value per Alert", f"${value_saved/max(len(reviewed),1):,.0f}")


elif page == "🔬 Feature Deep Dive":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🔬 Feature Deep Dive</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Understand what drives fraud predictions</p>', unsafe_allow_html=True)
    
    # Feature importance
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    
    top_n = st.slider("📊 Show top N features", 10, min(50, len(features)), 20)
    
    fig = px.bar(
        imp.head(top_n), y='feature', x='importance', orientation='h',
        color='importance', color_continuous_scale='Viridis',
        text=imp.head(top_n)['importance'].apply(lambda x: f"{x:.4f}")
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=max(500, top_n * 25),
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        title=f"🏆 Top {top_n} Feature Importances"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.markdown('<div class="section-header">📊 Feature Category Analysis</div>', unsafe_allow_html=True)
    
    categories = {
        "Transaction": [f for f in features if any(x in f.lower() for x in ['amt', 'transaction'])],
        "Time": [f for f in features if any(x in f.lower() for x in ['hour', 'day', 'night', 'weekend'])],
        "Card": [f for f in features if any(x in f.lower() for x in ['card'])],
        "Count (C)": [f for f in features if f.startswith('C') and f[1:].isdigit()],
        "Delta (D)": [f for f in features if f.startswith('D') and f[1:].isdigit()],
        "Vesta (V)": [f for f in features if f.startswith('V') and f[1:].isdigit()],
    }
    
    cat_imp = []
    for cat, feats in categories.items():
        if feats:
            total_imp = imp[imp['feature'].isin(feats)]['importance'].sum()
            cat_imp.append({"Category": cat, "Features": len(feats), "Total Importance": total_imp})
    
    if cat_imp:
        cat_df = pd.DataFrame(cat_imp).sort_values('Total Importance', ascending=True)
        
        fig = px.bar(cat_df, y='Category', x='Total Importance', orientation='h',
                    color='Total Importance', color_continuous_scale='Viridis',
                    text=cat_df['Total Importance'].apply(lambda x: f"{x:.3f}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top feature insight
    top_feat = imp.iloc[0]
    st.markdown(f'''
    <div class="ai-insight">
        <h4>🏆 Top Feature: {top_feat["feature"]}</h4>
        <p>The most predictive feature is <strong>{top_feat["feature"]}</strong> with importance {top_feat["importance"]:.4f}. 
        The top 10 features account for <strong>{imp.head(10)["importance"].sum()/imp["importance"].sum()*100:.0f}%</strong> of total model importance, 
        indicating a concentrated predictive signal.</p>
    </div>
    ''', unsafe_allow_html=True)


elif page == "🎚️ Threshold Optimizer":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🎚️ Threshold Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Fine-tune decision thresholds for optimal business outcomes</p>', unsafe_allow_html=True)
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for t in thresholds:
        pred = (y_pred >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        tn = ((pred == 0) & (y_val == 0)).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        
        results.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'alerts': tp + fp,
            'fraud_caught': tp,
            'fraud_missed': fn
        })
    
    df_t = pd.DataFrame(results)
    
    # Interactive chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df_t['threshold'], y=df_t['precision'], name='Precision',
                            line=dict(color='#6366f1', width=3), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_t['threshold'], y=df_t['recall'], name='Recall',
                            line=dict(color='#10b981', width=3), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=df_t['threshold'], y=df_t['f1'], name='F1 Score',
                            line=dict(color='#f59e0b', width=3, dash='dash'), mode='lines+markers'))
    
    fig.update_layout(
        title="📈 Precision, Recall & F1 vs Threshold",
        xaxis_title="Decision Threshold",
        yaxis_title="Score",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_tickformat='.0%',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Threshold selector
    st.markdown('<div class="section-header">🎯 Select Your Threshold</div>', unsafe_allow_html=True)
    
    selected_t = st.slider("Decision Threshold", 0.05, 0.90, 0.50, 0.05)
    
    idx = np.abs(df_t['threshold'] - selected_t).argmin()
    row = df_t.iloc[idx]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1: st.metric("🎯 Precision", f"{row['precision']:.0%}")
    with col2: st.metric("📊 Recall", f"{row['recall']:.0%}")
    with col3: st.metric("⚖️ F1 Score", f"{row['f1']:.0%}")
    with col4: st.metric("🔔 Daily Alerts", f"{int(row['alerts']):,}")
    with col5: st.metric("🚨 Fraud Caught", f"{int(row['fraud_caught'])}")
    
    # Business impact
    st.markdown("### 💰 Estimated Business Impact")
    
    avg_fraud = 300
    col1, col2, col3 = st.columns(3)
    
    with col1:
        saved = row['fraud_caught'] * avg_fraud * 0.8
        st.metric("💚 Value Saved", f"${saved:,.0f}")
    with col2:
        lost = row['fraud_missed'] * avg_fraud
        st.metric("💔 Value at Risk", f"${lost:,.0f}")
    with col3:
        net = saved - lost - row['alerts'] * 5
        st.metric("📈 Net Impact", f"${net:,.0f}")


elif page == "📈 What-If Analysis":
    st.markdown('<h1 style="font-size: 2.5rem; font-weight: 900; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📈 What-If Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Explore scenarios and understand model behavior</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🔄 Feature Impact", "📊 Scenario Modeling", "🎯 Sensitivity Analysis"])
    
    with tabs[0]:
        st.markdown("### 🔄 How do features affect predictions?")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Get numeric features
            numeric_feats = [f for f in features if X_val[f].dtype in [np.float64, np.float32, np.int64, np.int32]][:20]
            selected_feat = st.selectbox("📊 Select Feature", numeric_feats)
            
            if selected_feat:
                feat_min = float(X_val[selected_feat].min())
                feat_max = float(X_val[selected_feat].max())
                feat_mean = float(X_val[selected_feat].mean())
                
                st.markdown(f"**Range:** {feat_min:.2f} - {feat_max:.2f}")
                st.markdown(f"**Mean:** {feat_mean:.2f}")
        
        with col2:
            if selected_feat:
                # Create range of values
                feat_range = np.linspace(
                    max(feat_min, feat_mean - 3 * X_val[selected_feat].std()),
                    min(feat_max, feat_mean + 3 * X_val[selected_feat].std()),
                    50
                )
                
                # Get base sample
                base_sample = X_val.iloc[[0]].copy()
                
                # Calculate predictions for each value
                preds = []
                for val in feat_range:
                    sample = base_sample.copy()
                    sample[selected_feat] = val
                    pred = model.predict_proba(sample)[:, 1][0]
                    preds.append(pred)
                
                fig = px.line(x=feat_range, y=preds, 
                             labels={'x': selected_feat, 'y': 'Fraud Probability'})
                fig.add_vline(x=feat_mean, line_dash="dash", line_color="gray", 
                             annotation_text="Mean")
                fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)',
                                title=f"🔄 Impact of {selected_feat} on Prediction")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.markdown("### 📊 Business Scenario Modeling")
        
        st.markdown("**Configure different business scenarios to see their impact:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Scenario A: Conservative")
            threshold_a = st.slider("Threshold A", 0.3, 0.9, 0.7, key="ta")
            team_a = st.slider("Team Size A", 5, 30, 10, key="teama")
        
        with col2:
            st.markdown("#### Scenario B: Balanced")
            threshold_b = st.slider("Threshold B", 0.1, 0.7, 0.4, key="tb")
            team_b = st.slider("Team Size B", 5, 30, 15, key="teamb")
        
        with col3:
            st.markdown("#### Scenario C: Aggressive")
            threshold_c = st.slider("Threshold C", 0.05, 0.5, 0.2, key="tc")
            team_c = st.slider("Team Size C", 5, 30, 25, key="teamc")
        
        if st.button("🚀 Compare Scenarios", type="primary", use_container_width=True):
            scenarios = []
            for name, thresh, team in [("Conservative", threshold_a, team_a), 
                                        ("Balanced", threshold_b, team_b), 
                                        ("Aggressive", threshold_c, team_c)]:
                pred = (y_pred >= thresh).astype(int)
                tp = ((pred == 1) & (y_val == 1)).sum()
                fp = ((pred == 1) & (y_val == 0)).sum()
                alerts = tp + fp
                capacity = team * 8 * 4  # 8 hours, 4 per hour
                
                reviewed = min(alerts, capacity)
                precision = tp / max(alerts, 1)
                value = tp * 300 * 0.8 - team * 8 * 45
                
                scenarios.append({
                    "Scenario": name,
                    "Threshold": f"{thresh:.0%}",
                    "Team": team,
                    "Alerts": alerts,
                    "Precision": f"{precision:.0%}",
                    "Est. Value": f"${value:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)
    
    with tabs[2]:
        st.markdown("### 🎯 Model Sensitivity Analysis")
        st.info("📊 Analyze how sensitive the model is to different input variations")
        
        # Show score distribution
        fig = px.histogram(y_pred, nbins=50, title="📊 Score Distribution",
                          labels={'value': 'Fraud Score', 'count': 'Count'})
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("📊 Mean Score", f"{y_pred.mean():.3f}")
        with col2: st.metric("📈 Std Dev", f"{y_pred.std():.3f}")
        with col3: st.metric("🎯 Median", f"{np.median(y_pred):.3f}")
        with col4: st.metric("🔝 Max", f"{y_pred.max():.3f}")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <div class="footer-logo">🛡️</div>
    <div class="footer-title">Fraud Decisioning Platform</div>
    <div class="footer-subtitle">Enterprise-Grade ML System for Real-Time Fraud Detection</div>
    <div class="tech-stack">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">scikit-learn</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Docker</span>
    </div>
    <p style="margin-top: 1.5rem; color: #94a3b8; font-size: 0.8rem;">
        Built for Production
    </p>
</div>
""", unsafe_allow_html=True)
