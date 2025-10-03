
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.base import clone
import sys
import os
from pyngrok import ngrok as ng

# --- Import project modules ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\ML
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from notebooks import data_preprocessing as dp
from notebooks import pca_analysis as pc
from notebooks.config import Config as cfg
st.set_page_config(page_title="Heart Disease – Interactive Predictor", layout="wide")

# ========================
# Helpers
# ========================


@st.cache_resource
def _start_tunnel():
    # Disconnect any existing tunnels (avoids ERR_NGROK_334 if the daemon is alive)
    try:
        for t in ng.get_tunnels():
            try:
                ng.disconnect(t.public_url)
            except Exception:
                pass
    except Exception:
        pass

    try:
        ng.kill()  # ensure a fresh daemon so 127.0.0.1:4040 is free
    except Exception:
        pass

    # Start fresh tunnel on 8501
    tun = ng.connect(addr=8501, proto="http", bind_tls=True)
    return tun

try:
    tunnel = _start_tunnel()
    st.sidebar.success(f"Public URL: {tunnel.public_url}")
    print("Public URL:", tunnel.public_url, flush=True)
except Exception as e:
    st.sidebar.warning("Ngrok not started. You can run 'ngrok http 8501' in a separate terminal.")
    print("Ngrok start error:", e)

@st.cache_data(show_spinner=False)
def load_uci_heart():
    # Uses your dp.load_data() helper (UCI id is set in cfg)
    df = dp.load_data(cfg.DATASET_ID)
    # Keep original multi-class label in "num" and also a binarized target "num_bin"
    df_bin = dp.binarize_output(df.copy(deep=True))
    return df, df_bin

@st.cache_data(show_spinner=False)
def feature_stats(df, target_col="num"):
    X = df.drop(columns=[target_col])
    desc = X.describe().T
    return X.columns.tolist(), desc

@st.cache_resource(show_spinner=False)
def build_model(model_name: str, binarized: bool, test_size: float = 0.2, random_state: int = 42):
    """Train a simple pipeline (StandardScaler + selected model) for the UI demo.
    This is intentionally light-weight and independent from your full training pipeline.
    """
    # Get data
    df, df_bin = load_uci_heart()
    df_use = df_bin if binarized else df
    X, y = dp.split_features_target(df_use, target_col="num")

    # Fill missing, split, and train
    X = dp.fill_missing(X, cfg.MISSIING_VALUES_COLUMNS)
    X_train, X_test, y_train, y_test = dp.split_train_test(X, y, test_size=test_size, random_state=random_state)

    # Pick model from your config registry
    if model_name not in cfg.CLASSIFICATION_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {list(cfg.CLASSIFICATION_MODELS)}")

    base_model = cfg.CLASSIFICATION_MODELS[model_name]
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(base_model))])
    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test

def make_user_input_form(columns, stats):
    st.subheader("Enter Patient Measurements")
    with st.form("user_input"):
        cols = st.columns(3)
        inputs = {}
        for i, col_name in enumerate(columns):
            col = cols[i % 3]
            row = stats.loc[col_name]
            median = float(row["50%"]) if "50%" in row else float(row["mean"])
            min_v = float(row["min"]) if "min" in row else median - 3.0
            max_v = float(row["max"]) if "max" in row else median + 3.0
            step = max((max_v - min_v) / 100.0, 0.01)
            value = col.number_input(col_name, value=median, min_value=min_v, max_value=max_v, step=step, key=f"in_{col_name}")
            inputs[col_name] = value
        realtime = st.checkbox("Predict in real-time (updates when you change any field)", value=True)
        submitted = st.form_submit_button("Predict")
    return inputs, submitted, realtime

def predict_and_report(model, X_test, y_test, single_row: pd.DataFrame, is_binarized: bool):
    # Single prediction
    pred = model.predict(single_row)[0]
    st.success(f"Prediction for current inputs: **{int(pred)}**" + (" (1 = disease present, 0 = no disease)" if is_binarized else " (heart-disease severity scale)"))

    # Metrics on test set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.markdown("### Test-set Confusion Matrix")
    st.dataframe(pd.DataFrame(cm))

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.markdown("### Test-set Classification Report")
    st.dataframe(pd.DataFrame(report).T)

    # Optional ROC/AUC for binarized
    if is_binarized:
        try:
            # Try to extract positive-class scores
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                if scores.ndim > 1:
                    scores = scores[:, 1]
            else:
                scores = None

            if scores is not None:
                auc = roc_auc_score(y_test, scores)
                fpr, tpr, thr = roc_curve(y_test, scores)
                st.markdown(f"**AUC (test)**: {auc:.3f}")
                roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                st.line_chart(roc_df.set_index("FPR"))
        except Exception as e:
            st.info(f"ROC/AUC not available: {e}")

# ========================
# UI
# ========================
st.title("Heart Disease Demo")

tab_predict, tab_explore = st.tabs(["🔮 Interactive Prediction", "📊 Explore Dataset & Trends"])

with tab_predict:
    st.sidebar.header("Model Settings")
    binarized = st.sidebar.radio("Target mode", ["Binarized (0/1)", "Multi-class (0–4)"], index=0) == "Binarized (0/1)"
    model_name = st.sidebar.selectbox("Classifier", list(cfg.CLASSIFICATION_MODELS.keys()), index=0)

    # Train (cached)
    with st.spinner("Preparing model and dataset..."):
        model, X_train, X_test, y_train, y_test = build_model(model_name, binarized=binarized)
        df, df_bin = load_uci_heart()
        df_use = df_bin if binarized else df
        cols, stats = feature_stats(df_use, target_col="num")

    # User input
    inputs, submitted, realtime = make_user_input_form(cols, stats)

    # Real-time or button-triggered prediction
    input_row = pd.DataFrame([inputs])
    if realtime or submitted:
        predict_and_report(model, X_test, y_test, input_row, is_binarized=binarized)

with tab_explore:
    st.subheader("Class Distribution")
    df, df_bin = load_uci_heart()

    # Choose which version to explore
    explore_mode = st.radio("Explore target", ["Binarized (0/1)", "Multi-class (0–4)"], index=0, horizontal=True)
    df_use = df_bin if explore_mode.startswith("Binarized") else df

    # Histogram
    counts = df_use["num"].value_counts().sort_index()
    st.bar_chart(counts)

    # Correlation heatmap
    st.subheader("Correlation Heatmap (features only)")
    X_only = df_use.drop(columns=["num"])
    X_only = dp.fill_missing(X_only, cfg.MISSIING_VALUES_COLUMNS)
    corr = X_only.corr(numeric_only=True)

    # Use matplotlib since seaborn may not be installed in all setups
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(corr.values, aspect='auto')
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # PCA (2D projection colored by class)
    st.subheader("PCA Projection (2 components)")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_only)
    pca, X_pca = pc.pca_analysis(X_scaled, n_components=2)

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df_use["num"], alpha=0.8)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("PCA (colored by target)")
    st.pyplot(fig2)

st.caption("Tip: Use the sidebar to switch models or targets. Enter measurements and toggle 'Predict in real-time' to see instant results.")
