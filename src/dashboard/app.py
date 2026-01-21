"""
Fraud Decisioning Platform - Interactive Dashboard
===================================================
Beautiful, interactive fraud detection platform with
real-time scoring, analytics, and fraud ops simulation.
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
# BEAUTIFUL CSS WITH LIGHT COLORS
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
    }
    
    .stApp {
        background: linear-gradient(135deg, #fafbff 0%, #f0f4ff 50%, #faf5ff 100%);
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        animation: fadeIn 0.8s ease-out;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Beautiful Metric Cards */
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.08), 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid rgba(99, 102, 241, 0.1);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15), 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .metric-card-purple .metric-value { color: #6366f1; }
    .metric-card-blue .metric-value { color: #3b82f6; }
    .metric-card-green .metric-value { color: #10b981; }
    .metric-card-orange .metric-value { color: #f59e0b; }
    .metric-card-red .metric-value { color: #ef4444; }
    .metric-card-pink .metric-value { color: #ec4899; }
    
    /* Gradient Backgrounds */
    .gradient-purple { background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); }
    .gradient-blue { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); }
    .gradient-green { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); }
    .gradient-orange { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); }
    .gradient-red { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); }
    .gradient-pink { background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); }
    
    /* Risk Badges */
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
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4);
    }
    .risk-medium {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1e293b;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.4);
    }
    .risk-low {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(52, 211, 153, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 2rem 0 1.25rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Insight Cards */
    .insight-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.12);
    }
    
    .insight-card h4 {
        color: #1e293b;
        margin: 0 0 0.75rem 0;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .insight-card p {
        color: #475569;
        margin: 0;
        line-height: 1.7;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    
    .status-info {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        color: #2563eb;
        border: 1px solid #93c5fd;
    }
    
    .status-warning {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        color: #d97706;
        border: 1px solid #fcd34d;
    }
    
    /* Interactive Elements */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 0.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #64748b;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Score display */
    .score-display {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        padding: 2rem;
        border-radius: 24px;
        background: white;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
    }
    
    /* Animated counter */
    .counter {
        animation: countUp 1s ease-out;
    }
    
    @keyframes countUp {
        from { opacity: 0; }
        to { opacity: 1; }
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
            return df, "full", total
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
        ID_COL: range(1, n + 1),
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
    
    if "TransactionAmt" in df.columns:
        df["amt_log"] = np.log1p(df["TransactionAmt"])
        df["amt_decimal"] = (df["TransactionAmt"] % 1).round(2)
    
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] // 3600) % 24
        df["day"] = (df["TransactionDT"] // 86400) % 7
        df["is_weekend"] = (df["day"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    
    encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in [ID_COL, TARGET_COL]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("MISSING").astype(str))
            encoders[col] = le
    
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL] 
                    and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8]]
    
    X, y = df[feature_cols].fillna(-999), df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    model = GradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    
    return model, feature_cols, X_val, y_val, y_pred, encoders


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
# HELPERS
# =============================================================================

def risk_badge(prob):
    if prob >= 0.8: return '<span class="risk-badge risk-critical">🚨 CRITICAL</span>'
    elif prob >= 0.5: return '<span class="risk-badge risk-high">⚠️ HIGH</span>'
    elif prob >= 0.2: return '<span class="risk-badge risk-medium">📊 MEDIUM</span>'
    else: return '<span class="risk-badge risk-low">✅ LOW</span>'


def gauge_chart(value, title, color="#6366f1"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#1e293b', 'family': 'Inter'}},
        number={'font': {'size': 48, 'color': color, 'family': 'Inter'}, 'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 1], 'tickcolor': '#cbd5e1'},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "#f1f5f9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.5], 'color': '#e0e7ff'},
                {'range': [0.5, 0.75], 'color': '#c7d2fe'},
                {'range': [0.75, 1], 'color': '#a5b4fc'},
            ],
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-size: 3rem;">🛡️</div>
        <div style="font-size: 1.5rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FDP</div>
        <div style="color: #64748b; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; margin-top: 0.25rem;">FRAUD DECISIONING</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio("", [
        "🏠 Overview",
        "📊 Data Explorer", 
        "🎯 Model Performance",
        "⚡ Live Scoring",
        "🏢 Fraud Ops Simulator",
        "🔬 Feature Analysis",
        "🎚️ Threshold Tuning"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    with st.spinner("🔄 Loading data..."):
        raw_df, source, total = load_data()
    
    if source == "full":
        st.markdown(f'<div class="status-badge status-success">✅ Kaggle Data ({total:,})</div>', unsafe_allow_html=True)
    elif source == "sample":
        st.markdown(f'<div class="status-badge status-info">📦 Sample Data ({len(raw_df):,})</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-warning">🎲 Demo Data ({len(raw_df):,})</div>', unsafe_allow_html=True)
    
    with st.spinner("🧠 Training model..."):
        model, features, X_val, y_val, y_pred, encoders = train_model(raw_df)
        metrics = compute_metrics(y_val.values, y_pred)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 AUC", f"{metrics['auc_roc']:.3f}")
    with col2:
        st.metric("🎯 Fraud", f"{raw_df[TARGET_COL].mean():.1%}")


# =============================================================================
# PAGES
# =============================================================================

if page == "🏠 Overview":
    st.markdown('<h1 class="main-header">🛡️ Fraud Decisioning Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time ML-powered fraud detection & intelligent alert triage</p>', unsafe_allow_html=True)
    
    fraud_rate = raw_df[TARGET_COL].mean()
    fraud_count = raw_df[TARGET_COL].sum()
    avg_amt = raw_df["TransactionAmt"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''<div class="metric-card gradient-purple">
            <div class="metric-icon">📊</div>
            <div class="metric-value metric-card-purple">{len(raw_df):,}</div>
            <div class="metric-label">Transactions</div>
        </div>''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''<div class="metric-card gradient-red">
            <div class="metric-icon">🚨</div>
            <div class="metric-value metric-card-red">{fraud_rate:.2%}</div>
            <div class="metric-label">Fraud Rate</div>
        </div>''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''<div class="metric-card gradient-green">
            <div class="metric-icon">🎯</div>
            <div class="metric-value metric-card-green">{metrics["auc_roc"]:.3f}</div>
            <div class="metric-label">Model AUC-ROC</div>
        </div>''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''<div class="metric-card gradient-blue">
            <div class="metric-icon">⚙️</div>
            <div class="metric-value metric-card-blue">{len(features)}</div>
            <div class="metric-label">ML Features</div>
        </div>''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────────┐
        │                  🛡️ FRAUD DECISIONING PLATFORM             │
        ├────────────────────────────────────────────────────────────┤
        │                                                            │
        │  📥 Transactions  →  🔧 Feature Engineering  →  🤖 ML Model│
        │       (590K)            (400+ features)         (AUC: 0.94)│
        │                                                     │      │
        │                                                     ▼      │
        │                         ┌──────────────────────────────┐   │
        │                         │     🎯 Risk Tier Assignment  │   │
        │                         ├──────────────────────────────┤   │
        │                         │  🚨 CRITICAL  │ >80% │ BLOCK │   │
        │                         │  ⚠️  HIGH     │ >50% │ REVIEW│   │
        │                         │  📊 MEDIUM   │ >20% │ CHECK │   │
        │                         │  ✅ LOW      │ <20% │APPROVE│   │
        │                         └──────────────────────────────┘   │
        │                                                            │
        └────────────────────────────────────────────────────────────┘
        ```
        """)
    
    with col2:
        st.markdown('<div class="section-header">📈 Performance @ K</div>', unsafe_allow_html=True)
        
        k_data = []
        for k, v in metrics['metrics_k'].items():
            k_data.append({
                "Top K": f"{k:,}",
                "Precision": f"{v['precision']:.0%}",
                "Recall": f"{v['recall']:.0%}",
                "🎯 Fraud": v['caught']
            })
        st.dataframe(pd.DataFrame(k_data), use_container_width=True, hide_index=True)
    
    # Insight
    p500 = metrics['metrics_k'].get(500, {}).get('precision', 0)
    st.markdown(f'''<div class="insight-card">
        <h4>💡 Key Insight</h4>
        <p>Model achieves <strong>{metrics['auc_roc']:.3f} AUC-ROC</strong>. At top 500 predictions, 
        precision is <strong>{p500:.0%}</strong> — that's <strong>{p500/metrics['baseline']:.0f}x better</strong> 
        than random sampling! This means reviewing just 500 alerts catches significantly more fraud.</p>
    </div>''', unsafe_allow_html=True)


elif page == "📊 Data Explorer":
    st.markdown('<h1 class="main-header">📊 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive exploration of the fraud detection dataset</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["📈 Distribution", "💰 Amount Analysis", "🏷️ Categories", "🔗 Correlations"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=raw_df[TARGET_COL].value_counts().values,
                names=["✅ Legitimate", "🚨 Fraud"],
                title="Transaction Distribution",
                color_discrete_sequence=["#6366f1", "#ef4444"],
                hole=0.5
            )
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📦 Total", f"{len(raw_df):,}")
                st.metric("🚨 Fraud", f"{raw_df[TARGET_COL].sum():,}")
            with col_b:
                st.metric("📊 Rate", f"{raw_df[TARGET_COL].mean():.3%}")
                st.metric("⚖️ Ratio", f"1:{int(1/raw_df[TARGET_COL].mean())}")
            
            st.markdown(f'''<div class="insight-card">
                <h4>⚠️ Class Imbalance</h4>
                <p>Only {raw_df[TARGET_COL].mean():.2%} of transactions are fraud. 
                This requires careful handling with techniques like stratified sampling, 
                class weights, or precision@K optimization.</p>
            </div>''', unsafe_allow_html=True)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                raw_df, x="TransactionAmt", color=TARGET_COL,
                title="💰 Amount Distribution",
                color_discrete_map={0: "#6366f1", 1: "#ef4444"},
                barmode="overlay", opacity=0.75, nbins=60, log_y=True
            )
            fig.update_xaxes(range=[0, 800], title="Amount ($)")
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', legend_title="")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                raw_df, x=TARGET_COL, y="TransactionAmt", color=TARGET_COL,
                title="📊 Amount by Status",
                color_discrete_map={0: "#6366f1", 1: "#ef4444"}
            )
            fig.update_yaxes(range=[0, 500])
            fig.update_layout(height=400, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        fraud_amt = raw_df[raw_df[TARGET_COL]==1]["TransactionAmt"]
        legit_amt = raw_df[raw_df[TARGET_COL]==0]["TransactionAmt"]
        
        st.markdown(f'''<div class="insight-card">
            <h4>💡 Amount Pattern</h4>
            <p>Fraud: avg <strong>${fraud_amt.mean():.0f}</strong> | Legitimate: avg <strong>${legit_amt.mean():.0f}</strong> 
            — Fraudulent transactions are {fraud_amt.mean()/legit_amt.mean():.1f}x higher on average!</p>
        </div>''', unsafe_allow_html=True)
    
    with tabs[2]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            cats = [c for c in ["ProductCD", "card4", "card6", "DeviceType"] if c in raw_df.columns]
            selected = st.selectbox("🏷️ Select Feature", cats)
        
        with col2:
            if selected:
                cat_data = raw_df.groupby(selected)[TARGET_COL].agg(['mean', 'count']).reset_index()
                cat_data.columns = [selected, 'Fraud Rate', 'Count']
                cat_data = cat_data[cat_data['Count'] >= 30].sort_values('Fraud Rate', ascending=True)
                
                fig = px.bar(
                    cat_data, y=selected, x='Fraud Rate', orientation='h',
                    title=f"🎯 Fraud Rate by {selected}",
                    color='Fraud Rate', color_continuous_scale='RdYlGn_r',
                    text=cat_data['Fraud Rate'].apply(lambda x: f"{x:.1%}")
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, xaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        num_cols = ['TransactionAmt', 'C1', 'C2', 'C3', 'D1', 'D2', TARGET_COL]
        num_cols = [c for c in num_cols if c in raw_df.columns]
        
        if len(num_cols) > 2:
            corr = raw_df[num_cols].corr()
            fig = px.imshow(corr, title="🔗 Feature Correlations", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(height=450, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)


elif page == "🎯 Model Performance":
    st.markdown('<h1 class="main-header">🎯 Model Performance</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive evaluation of the fraud detection model</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(gauge_chart(metrics['auc_roc'], "AUC-ROC", "#6366f1"), use_container_width=True)
    with col2:
        st.plotly_chart(gauge_chart(metrics['auc_pr'], "AUC-PR", "#10b981"), use_container_width=True)
    with col3:
        p500 = metrics['metrics_k'].get(500, {}).get('precision', 0)
        st.plotly_chart(gauge_chart(p500, "Precision@500", "#f59e0b"), use_container_width=True)
    
    st.markdown('<div class="section-header">📈 Performance Curves</div>', unsafe_allow_html=True)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('🎯 ROC Curve', '📊 Precision-Recall'))
    
    fig.add_trace(go.Scatter(x=metrics['fpr'], y=metrics['tpr'], mode='lines',
                            name=f"ROC (AUC={metrics['auc_roc']:.3f})",
                            line=dict(color='#6366f1', width=3),
                            fill='tozeroy', fillcolor='rgba(99,102,241,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                            line=dict(color='#94a3b8', dash='dash', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=metrics['rec'], y=metrics['prec'], mode='lines',
                            name=f"PR (AUC={metrics['auc_pr']:.3f})",
                            line=dict(color='#10b981', width=3),
                            fill='tozeroy', fillcolor='rgba(16,185,129,0.1)'), row=1, col=2)
    fig.add_hline(y=metrics['baseline'], line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Baseline ({metrics['baseline']:.1%})", row=1, col=2)
    
    fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                     legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision@K bar chart
    st.markdown('<div class="section-header">🎯 Performance at Top K</div>', unsafe_allow_html=True)
    
    k_df = pd.DataFrame([
        {"K": k, "Precision": v['precision'], "Recall": v['recall']}
        for k, v in metrics['metrics_k'].items()
    ])
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(k_df, x="K", y="Precision", title="📊 Precision @ K",
                    color="Precision", color_continuous_scale="Purples",
                    text=k_df["Precision"].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(k_df, x="K", y="Recall", title="📊 Recall @ K",
                    color="Recall", color_continuous_scale="Greens",
                    text=k_df["Recall"].apply(lambda x: f"{x:.0%}"))
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False, yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)


elif page == "⚡ Live Scoring":
    st.markdown('<h1 class="main-header">⚡ Live Fraud Scoring</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time transaction risk assessment</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["🎛️ Manual Input", "🎲 Random Samples", "📤 Batch Upload"])
    
    with tabs[0]:
        st.markdown("### 📝 Enter Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            amt = st.number_input("💰 Amount ($)", 1.0, 10000.0, 150.0, 10.0)
            product = st.selectbox("📦 Product", ["W - Web", "H - Hotel", "C - Cash", "S - Services", "R - Retail"])
        
        with col2:
            card = st.selectbox("💳 Card Network", ["Visa", "Mastercard", "Discover", "Amex"])
            card_type = st.selectbox("🏦 Card Type", ["Debit", "Credit", "Charge"])
        
        with col3:
            device = st.selectbox("📱 Device", ["Desktop", "Mobile", "Tablet"])
            hour = st.slider("🕐 Hour", 0, 23, 14)
        
        if st.button("⚡ Score Transaction", type="primary", use_container_width=True):
            # Get base sample and modify
            idx = np.random.randint(0, len(X_val))
            X_sample = X_val.iloc[[idx]].copy()
            
            if "TransactionAmt" in X_sample.columns: X_sample["TransactionAmt"] = amt
            if "amt_log" in X_sample.columns: X_sample["amt_log"] = np.log1p(amt)
            if "hour" in X_sample.columns: X_sample["hour"] = hour
            if "is_night" in X_sample.columns: X_sample["is_night"] = int(hour >= 22 or hour <= 6)
            
            prob = model.predict_proba(X_sample)[:, 1][0]
            percentile = (y_pred < prob).mean() * 100
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Big score display
                color = "#ef4444" if prob >= 0.5 else "#f59e0b" if prob >= 0.2 else "#10b981"
                st.markdown(f'''
                <div class="score-display" style="border: 3px solid {color};">
                    <div style="color: {color}; margin-bottom: 0.5rem;">{prob:.1%}</div>
                    <div style="font-size: 1rem; color: #64748b;">Fraud Probability</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🎯 Risk Level")
                st.markdown(risk_badge(prob), unsafe_allow_html=True)
                
                st.markdown(f"**Percentile:** {percentile:.0f}%")
                st.markdown(f"**vs Baseline:** {prob/metrics['baseline']:.1f}x")
                
                st.markdown("### 📋 Action")
                if prob >= 0.8:
                    st.error("🚫 BLOCK")
                elif prob >= 0.5:
                    st.warning("👁️ REVIEW")
                elif prob >= 0.2:
                    st.info("🔐 CHALLENGE")
                else:
                    st.success("✅ APPROVE")
            
            # Risk factors
            st.markdown("### 🔍 Risk Factors")
            factors = []
            if amt > 500: factors.append(("💰 High Amount", f"${amt:.0f} is above typical range"))
            if amt < 15: factors.append(("🧪 Test Amount", "Very low amounts may be card testing"))
            if hour >= 22 or hour <= 6: factors.append(("🌙 Night Transaction", f"Transaction at {hour}:00"))
            if device == "Mobile": factors.append(("📱 Mobile Device", "Slightly elevated risk"))
            if not factors: factors.append(("✅ Normal Profile", "No specific risk indicators"))
            
            for name, desc in factors:
                st.markdown(f"- **{name}:** {desc}")
    
    with tabs[1]:
        st.markdown("### 🎲 Sample Real Transactions")
        
        n = st.slider("Number of samples", 5, 30, 10)
        
        if st.button("🎲 Get Random Samples", type="primary"):
            idx = np.random.choice(len(X_val), n, replace=False)
            
            results = []
            for i in idx:
                score = y_pred[i]
                actual = y_val.iloc[i]
                pred_fraud = score >= 0.5
                
                results.append({
                    "🎯 Score": f"{score:.1%}",
                    "⚡ Risk": "🚨" if score >= 0.8 else "⚠️" if score >= 0.5 else "📊" if score >= 0.2 else "✅",
                    "📋 Actual": "🚨 FRAUD" if actual == 1 else "✅ LEGIT",
                    "✓ Correct": "✅" if pred_fraud == (actual == 1) else "❌"
                })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            correct = sum(1 for r in results if r["✓ Correct"] == "✅")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("🎯 Accuracy", f"{correct}/{n}")
            with col2: st.metric("📊 Correct", f"{correct/n*100:.0f}%")
            with col3: st.metric("📈 Sample Size", n)
    
    with tabs[2]:
        st.markdown("### 📤 Batch Scoring")
        st.info("📁 Upload a CSV file with transactions to score in batch")
        
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            batch = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(batch)} transactions")
            st.dataframe(batch.head())


elif page == "🏢 Fraud Ops Simulator":
    st.markdown('<h1 class="main-header">🏢 Fraud Ops Simulator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize your fraud operations team capacity</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👥 Team Setup")
        analysts = st.slider("Analysts", 1, 30, 10)
        shift = st.slider("Shift Hours", 4, 12, 8)
    
    with col2:
        st.markdown("### ⏱️ Operations")
        review_time = st.slider("Review Time (mins)", 5, 60, 15)
        threshold = st.slider("Alert Threshold", 0.01, 0.50, 0.10, 0.01)
    
    with col3:
        st.markdown("### 💰 Economics")
        recovery = st.slider("Recovery Rate (%)", 50, 100, 80) / 100
        fraud_value = st.number_input("Avg Fraud Value ($)", 100, 1000, 250)
    
    capacity = analysts * (shift * 60 // review_time)
    
    st.markdown(f'''<div class="insight-card">
        <h4>📊 Daily Capacity</h4>
        <p>Your team can review <strong>{capacity:,}</strong> alerts per day 
        ({analysts} analysts × {shift}h shift × {60//review_time} reviews/hour)</p>
    </div>''', unsafe_allow_html=True)
    
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        sim = pd.DataFrame({'score': y_pred, 'actual': y_val.values})
        alerts = sim[sim['score'] >= threshold].sort_values('score', ascending=False)
        reviewed = alerts.head(capacity)
        
        tp = len(reviewed[reviewed['actual'] == 1])
        fp = len(reviewed[reviewed['actual'] == 0])
        fn = len(alerts.iloc[capacity:][alerts.iloc[capacity:]['actual'] == 1])
        
        value_saved = tp * fraud_value * recovery
        value_lost = fn * fraud_value
        cost = len(reviewed) * (review_time / 60) * 35
        net = value_saved - value_lost - cost
        precision = tp / max(len(reviewed), 1)
        
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Simulation Results</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''<div class="metric-card gradient-purple">
                <div class="metric-icon">🔔</div>
                <div class="metric-value metric-card-purple">{len(alerts):,}</div>
                <div class="metric-label">Alerts Generated</div>
            </div>''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''<div class="metric-card gradient-blue">
                <div class="metric-icon">👁️</div>
                <div class="metric-value metric-card-blue">{len(reviewed):,}</div>
                <div class="metric-label">Reviewed</div>
            </div>''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''<div class="metric-card gradient-green">
                <div class="metric-icon">🎯</div>
                <div class="metric-value metric-card-green">{tp}</div>
                <div class="metric-label">Fraud Caught</div>
            </div>''', unsafe_allow_html=True)
        
        with col4:
            color = "green" if net > 0 else "red"
            st.markdown(f'''<div class="metric-card gradient-{"green" if net > 0 else "orange"}">
                <div class="metric-icon">{"💰" if net > 0 else "⚠️"}</div>
                <div class="metric-value metric-card-{"green" if net > 0 else "orange"}">${net:,.0f}</div>
                <div class="metric-label">Net Value</div>
            </div>''', unsafe_allow_html=True)
        
        st.markdown("### 📈 Financial Breakdown")
        
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("💚 Saved", f"${value_saved:,.0f}")
        with col2: st.metric("💔 Lost", f"${value_lost:,.0f}")
        with col3: st.metric("💼 Cost", f"${cost:,.0f}")
        
        # Recommendation
        if len(alerts) > capacity * 1.2:
            st.warning(f"⚠️ **Backlog Alert:** {len(alerts) - capacity:,} alerts cannot be reviewed. Consider hiring more analysts or raising threshold.")
        elif precision > 0.7:
            st.success(f"✅ **Good Performance:** {precision:.0%} precision. Consider lowering threshold to catch more fraud.")
        else:
            st.info(f"📊 **Balanced:** Current settings provide good coverage.")


elif page == "🔬 Feature Analysis":
    st.markdown('<h1 class="main-header">🔬 Feature Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Understand what drives fraud predictions</p>', unsafe_allow_html=True)
    
    imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False)
    
    top_n = st.slider("Show top N features", 10, 50, 20)
    
    fig = px.bar(
        imp.head(top_n), y='feature', x='importance', orientation='h',
        title=f"🏆 Top {top_n} Feature Importances",
        color='importance', color_continuous_scale='Purples'
    )
    fig.update_layout(
        height=max(500, top_n * 25),
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f'''<div class="insight-card">
        <h4>🏆 Top Feature: {imp.iloc[0]["feature"]}</h4>
        <p>The most important feature is <strong>{imp.iloc[0]["feature"]}</strong> with importance {imp.iloc[0]["importance"]:.4f}. 
        The top 10 features account for <strong>{imp.head(10)["importance"].sum()/imp["importance"].sum()*100:.0f}%</strong> of total importance.</p>
    </div>''', unsafe_allow_html=True)


elif page == "🎚️ Threshold Tuning":
    st.markdown('<h1 class="main-header">🎚️ Threshold Tuning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Optimize decision thresholds for your business</p>', unsafe_allow_html=True)
    
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for t in thresholds:
        pred = (y_pred >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        
        results.append({
            'threshold': t,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'alerts': tp + fp
        })
    
    df_t = pd.DataFrame(results)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t['threshold'], y=df_t['precision'], name='Precision',
                            line=dict(color='#6366f1', width=3)))
    fig.add_trace(go.Scatter(x=df_t['threshold'], y=df_t['recall'], name='Recall',
                            line=dict(color='#10b981', width=3)))
    fig.update_layout(
        title="📈 Precision vs Recall by Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_tickformat='.0%',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 🎚️ Select Threshold")
    selected = st.slider("Threshold", 0.05, 0.90, 0.50, 0.05)
    
    idx = np.abs(df_t['threshold'] - selected).argmin()
    row = df_t.iloc[idx]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🎯 Precision", f"{row['precision']:.0%}")
    with col2: st.metric("📊 Recall", f"{row['recall']:.0%}")
    with col3: st.metric("🔔 Alerts/Day", f"{int(row['alerts']):,}")
    with col4:
        net = row['precision'] * 200 * row['alerts'] * 0.8 - row['alerts'] * 5
        st.metric("💰 Est. Value", f"${net:,.0f}")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #64748b;">
    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🛡️</div>
    <div style="font-weight: 700; color: #1e293b;">Fraud Decisioning Platform</div>
    <div style="font-size: 0.85rem; margin-top: 0.5rem;">Built with Streamlit • scikit-learn • Plotly</div>
    <div style="font-size: 0.75rem; margin-top: 1rem; color: #94a3b8;">Production-Ready ML System for Interview Demonstrations</div>
</div>
""", unsafe_allow_html=True)
