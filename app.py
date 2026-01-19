"""NeuroBACE-ML Streamlit app

- Accepts SMILES strings (manual or CSV)
- Computes 2048-bit Morgan fingerprints (radius=2) via RDKit
- Predicts BACE1 inhibition probability using a pre-trained XGBoost model

This app serves a pre-trained model only (no training code here).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray

# -----------------------------
# Paths / constants
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "BACE1_trained_model_optimized.pkl"
JSON_MODEL_PATH = BASE_DIR / "BACE1_trained_model_optimized.json"

FP_BITS = 2048
FP_RADIUS = 2

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

# -----------------------------
# Theme / UI
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

st.sidebar.title("NeuroBACE-ML")
theme_choice = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)
st.session_state.theme = theme_choice

if st.session_state.theme == "Dark":
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_temp = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#000000", "#f1f5f9", "#2563eb"
    plotly_temp = "plotly_white"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}

    /* Force text visibility */
    h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{
        color: {text} !important; opacity: 1 !important;
    }}

    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}

    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
        color: white !important; font-weight: bold !important; border-radius: 8px !important;
    }}

    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Networking helpers
# -----------------------------
@st.cache_resource
def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "NeuroBACE-ML/1.0 (Streamlit; PubChem PUG REST)"})
    return s


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def get_compound_name(smiles: str) -> str:
    """Resolve a human-readable name from PubChem by SMILES.

    Important: SMILES must be URL-encoded, otherwise many valid SMILES break the request.
    """
    try:
        smiles = smiles.strip()
        if not smiles:
            return ""
        enc = quote(smiles, safe="")
        url = f"{PUBCHEM_BASE}/compound/smiles/{enc}/property/Title/JSON"
        r = _requests_session().get(url, timeout=8)
        if r.status_code != 200:
            return "Unknown"
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return "Unknown"
        return props[0].get("Title", "Unknown")
    except Exception:
        return "Unknown"


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    """Load the trained classifier.

    This project currently loads a pickled model. For long-term reproducibility,
    prefer exporting the model using XGBoost's native save_model/load_model.
    """
    try:
        # Prefer XGBoost's native model format for stability across versions.
        if JSON_MODEL_PATH.exists():
            import xgboost as xgb

            clf = xgb.XGBClassifier()
            clf.load_model(str(JSON_MODEL_PATH))
            if hasattr(clf, "predict_proba"):
                return clf

        # Fallback: legacy pickle (less stable across XGBoost versions).
        if MODEL_PATH.exists():
            with MODEL_PATH.open("rb") as f:
                model = pickle.load(f)
            if hasattr(model, "predict_proba"):
                return model

        return None
    except Exception:
        return None


model = load_model()

# -----------------------------
# Featurization + prediction
# -----------------------------

def smiles_to_fp_array(smiles: str, n_bits: int = FP_BITS, radius: int = FP_RADIUS) -> tuple[np.ndarray | None, str | None]:
    """Convert SMILES to a numpy fingerprint array; returns (array, error_message)."""
    s = (smiles or "").strip()
    if not s:
        return None, "Empty SMILES"

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None, "Invalid SMILES"

    # Prefer the newer generator API when available.
    fp = None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    except Exception:
        from rdkit.Chem import AllChem

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)

    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr, None


def predict_inhibition_probability(smiles: str) -> tuple[float | None, str | None]:
    """Return (probability, error_message)."""
    if model is None:
        return None, "Model not loaded"

    fp_arr, err = smiles_to_fp_array(smiles)
    if err:
        return None, err

    try:
        X = fp_arr.reshape(1, -1)
        proba = float(model.predict_proba(X)[0][1])
        # Clamp for safety
        proba = max(0.0, min(1.0, proba))
        return proba, None
    except Exception as e:
        return None, f"Prediction failed: {e}"


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Probability threshold (Active if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)
    resolve_names = st.toggle("Resolve compound names via PubChem", value=False, help="Optional. Disabling is faster for large batches.")
    st.caption("v1.3.1 | NeuroBACE-ML")

# -----------------------------
# Main UI
# -----------------------------
st.title("🧠 NeuroBACE-ML")
st.markdown("##### *Platform for BACE1 Inhibitor Probability Prediction*")
st.write("---")

if model is None:
    st.error(
        "Model could not be loaded. Ensure either 'BACE1_trained_model_optimized.json' (preferred) or 'BACE1_trained_model_optimized.pkl' is present in the same folder as app.py."
    )


t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics", "🔬 Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols: list[str] = []

    if in_type == "Manual Entry":
        st.caption("Tip: Enter one SMILES per line.")
        raw = st.text_area(
            "SMILES (one per line):",
            # Use a simple, RDKit-valid default SMILES.
            "CC(=O)OC1=CC=CC=C1C(=O)O",
        )
        mols = [s.strip() for s in raw.split("\n") if s.strip()]

    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df_in = pd.read_csv(f)
            st.write("Detected columns:", list(df_in.columns))
            smiles_col = st.selectbox(
                "Select the SMILES column",
                options=list(df_in.columns),
                index=list(df_in.columns).index("smiles") if "smiles" in df_in.columns else 0,
            )
            mols = [str(x).strip() for x in df_in[smiles_col].dropna().tolist()]

    col_a, col_b = st.columns([1, 3])
    with col_a:
        start = st.button("Start Virtual Screening")
    with col_b:
        st.caption("For invalid SMILES, results will include an error message.")

    if start:
        if not mols:
            st.warning("No SMILES provided.")
        elif model is None:
            st.warning("Model is not available; cannot run predictions.")
        else:
            res = []
            bar = st.progress(0)
            for i, s in enumerate(mols):
                proba, err = predict_inhibition_probability(s)

                name = ""
                if resolve_names and err is None:
                    name = get_compound_name(s)

                res.append(
                    {
                        "Compound Name": name if name else "",
                        "Inhibition Prob": proba if proba is not None else np.nan,
                        "Result": ("ACTIVE" if (proba is not None and proba >= threshold) else "INACTIVE")
                        if err is None
                        else "ERROR",
                        "Error": "" if err is None else err,
                        "SMILES": s,
                    }
                )

                bar.progress((i + 1) / max(1, len(mols)))

            df_res = pd.DataFrame(res)
            st.session_state["results"] = df_res

            # Summary metrics (ignore errors)
            df_ok = df_res[df_res["Result"] != "ERROR"].copy()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Molecules (total)", len(df_res))
            c2.metric("Scored", len(df_ok))
            c3.metric("Potent Hits", int((df_ok["Result"] == "ACTIVE").sum()))
            c4.metric("Max Probability", f"{(df_ok['Inhibition Prob'].max() if len(df_ok) else 0):.3f}")

            st.write("---")
            st.dataframe(
                df_res,
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Export Results (CSV)",
                df_res.to_csv(index=False),
                file_name="NeuroBACE_Report.csv",
                mime="text/csv",
            )

with t2:
    if "results" not in st.session_state:
        st.info("Run a screening first to see analytics.")
    else:
        df = st.session_state["results"].copy()
        df_ok = df[df["Result"] != "ERROR"].dropna(subset=["Inhibition Prob"]).copy()

        st.markdown("### Predictive Probability Distribution")

        if df_ok.empty:
            st.warning("No valid predictions available to plot.")
        else:
            # Use compound name where available, else SMILES.
            df_ok["Label"] = df_ok["Compound Name"].where(df_ok["Compound Name"].astype(bool), df_ok["SMILES"])
            df_ok = df_ok.sort_values("Inhibition Prob", ascending=True)

            fig_bar = px.bar(
                df_ok,
                y="Label",
                x="Inhibition Prob",
                orientation="h",
                color="Inhibition Prob",
                color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                template=plotly_temp,
                labels={"Inhibition Prob": "Probability"},
                height=max(420, len(df_ok) * 26),
            )
            fig_bar.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### Histogram")
            fig_hist = px.histogram(
                df_ok,
                x="Inhibition Prob",
                nbins=30,
                template=plotly_temp,
                labels={"Inhibition Prob": "Probability"},
            )
            fig_hist.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig_hist, use_container_width=True)

with t3:
    st.write("### Platform Architecture")
    st.markdown(
        """
- **Model:** Pre-trained XGBoost classifier (loaded from disk)
- **Output:** BACE1 inhibition probability (class 1) and binary label using a configurable probability threshold
- **Feature extraction:** 2048-bit Morgan fingerprints (radius = 2)
- **Optional integration:** PubChem PUG REST for compound title resolution

**Implementation notes**
- The model file is currently loaded from a Python pickle. For long-term reproducibility across XGBoost versions, consider exporting with XGBoost native `save_model` and loading with `load_model`.
        """
    )
