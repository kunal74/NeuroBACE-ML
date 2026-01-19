import os
import pickle
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048
FP_RADIUS = 2
PUBCHEM_TITLE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/property/Title/JSON"

# Global Morgan fingerprint generator (RDKit recommended API)
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# --------------------------------------------------------------------------------------
# Theme and layout
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

st.sidebar.title("NeuroBACE-ML")
st.session_state.theme = st.sidebar.radio("Appearance Mode", ["Dark", "Light"], horizontal=True)

if st.session_state.theme == "Dark":
    bg, text, card, accent = "#0f172a", "#f8fafc", "#1e293b", "#38bdf8"
    plotly_template = "plotly_dark"
else:
    bg, text, card, accent = "#ffffff", "#0b1220", "#f1f5f9", "#2563eb"
    plotly_template = "plotly_white"

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {bg} !important; color: {text} !important; }}
      [data-testid="stSidebar"] {{ background-color: {bg} !important; border-right: 1px solid {accent}33; }}

      h1, h2, h3, h4, label, span, p, [data-testid="stWidgetLabel"] p, .stMarkdown p {{
        color: {text} !important;
        opacity: 1 !important;
      }}

      [data-testid="stMetric"] {{
        background-color: {card} !important;
        border: 1px solid {accent}44 !important;
        border-radius: 12px;
      }}

      [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: 700; }}

      .stButton>button {{
        background: linear-gradient(90deg, #0ea5e9, #2563eb) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
      }}

      #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_compound_name_pubchem(smiles: str) -> str:
    """Resolve a SMILES to a PubChem Title.

    Important: SMILES must be URL encoded.
    """
    try:
        encoded = quote(smiles, safe="")
        url = PUBCHEM_TITLE_URL.format(encoded)
        r = requests.get(url, timeout=7)
        if r.status_code != 200:
            return "Unknown"
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return "Unknown"
        return props[0].get("Title", "Unknown") or "Unknown"
    except Exception:
        return "Unknown"


def guess_smiles_column(columns) -> str | None:
    for c in columns:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    for c in columns:
        if "smiles" in str(c).strip().lower():
            return c
    return None


@st.cache_resource
def load_model_bundle():
    """Load the trained model.

    Preferred: XGBoost native model JSON to avoid pickle and sklearn version coupling.
    """
    if os.path.exists(MODEL_JSON):
        booster = xgb.Booster()
        booster.load_model(MODEL_JSON)
        return {"kind": "booster", "model": booster}

    if os.path.exists(MODEL_PKL):
        with open(MODEL_PKL, "rb") as f:
            obj = pickle.load(f)
        # Try to extract a booster if it is an sklearn wrapper
        if hasattr(obj, "get_booster"):
            try:
                booster = obj.get_booster()
                return {"kind": "booster", "model": booster}
            except Exception:
                pass
        return {"kind": "pickle", "model": obj}

    return None


def featurize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    try:
        fp = FPGEN.GetFingerprint(mol)
        arr = np.zeros((FP_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.reshape(1, -1), None
    except Exception as e:
        return None, f"RDKit error: {e}"


def predict_probability(model_bundle, X: np.ndarray):
    """Return probability of the positive class (Active)."""
    kind = model_bundle.get("kind")
    model = model_bundle.get("model")

    if kind == "booster":
        dm = xgb.DMatrix(X)
        p = model.predict(dm)
        return float(p[0])

    # Fallback: pickle may contain XGBClassifier or similar
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    if hasattr(model, "predict"):
        p = model.predict(X)
        # If predict returns probability-like output
        try:
            return float(p[0])
        except Exception:
            return float(p)

    raise RuntimeError("Loaded model does not support prediction.")


# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Probability threshold (Active if >= threshold)", 0.0, 1.0, 0.70, 0.01)
    fetch_names = st.checkbox("Fetch compound names from PubChem", value=True)
    st.caption("NeuroBACE-ML app")


# --------------------------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------------------------
st.title("NeuroBACE-ML")
st.markdown("Advanced platform for BACE1 inhibitor probability prediction")
st.write("---")

model_bundle = load_model_bundle()
if model_bundle is None:
    st.error(
        "Model file not found. Upload the model file to the same folder as app.py. "
        f"Expected: {MODEL_JSON} (preferred) or {MODEL_PKL}."
    )
    st.stop()


t1, t2, t3 = st.tabs(["Screening", "Visual analytics", "Specifications"])

with t1:
    in_type = st.radio("Input source", ["Manual entry", "Batch upload (CSV)"], horizontal=True)

    smiles_list = []
    uploaded_df = None

    if in_type == "Manual entry":
        default_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin (valid example)
        raw = st.text_area("SMILES (one per line)", default_smiles, height=140)
        smiles_list = [s.strip() for s in raw.splitlines() if s.strip()]
    else:
        f = st.file_uploader("Upload a CSV file", type=["csv"])
        if f is not None:
            uploaded_df = pd.read_csv(f)
            col_guess = guess_smiles_column(uploaded_df.columns)
            col = st.selectbox(
                "Select the SMILES column",
                options=list(uploaded_df.columns),
                index=list(uploaded_df.columns).index(col_guess) if col_guess in list(uploaded_df.columns) else 0,
            )
            smiles_list = uploaded_df[col].astype(str).tolist()

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        start = st.button("Start virtual screening")
    with col_btn2:
        if st.button("Clear results"):
            st.session_state.pop("results", None)

    if start:
        if not smiles_list:
            st.warning("No SMILES provided.")
        else:
            rows = []
            prog = st.progress(0)

            for i, s in enumerate(smiles_list):
                s = str(s).strip()
                X, err = featurize_smiles(s)

                if err is not None:
                    rows.append(
                        {
                            "Compound Name": "Unknown" if not fetch_names else get_compound_name_pubchem(s),
                            "SMILES": s,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "Error": err,
                        }
                    )
                else:
                    try:
                        prob = predict_probability(model_bundle, X)
                        name = "Unknown" if not fetch_names else get_compound_name_pubchem(s)
                        rows.append(
                            {
                                "Compound Name": name,
                                "SMILES": s,
                                "Inhibition Prob": float(prob),
                                "Result": "ACTIVE" if prob >= threshold else "INACTIVE",
                                "Error": "",
                            }
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "Compound Name": "Unknown" if not fetch_names else get_compound_name_pubchem(s),
                                "SMILES": s,
                                "Inhibition Prob": np.nan,
                                "Result": "ERROR",
                                "Error": str(e),
                            }
                        )

                prog.progress((i + 1) / max(1, len(smiles_list)))

            df_res = pd.DataFrame(rows)
            st.session_state["results"] = df_res

            valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].copy()
            invalid_count = int((df_res["Result"] == "INVALID").sum())
            error_count = int((df_res["Result"] == "ERROR").sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input molecules", len(df_res))
            c2.metric("Valid predictions", len(valid))
            c3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()))
            c4.metric("Invalid or errors", invalid_count + error_count)

            if len(valid) == 0:
                st.warning("No valid molecules could be processed. Check SMILES formatting.")
            else:
                max_prob = float(valid["Inhibition Prob"].max())
                st.info(f"Max predicted probability (valid only): {max_prob:.3f}")

            st.write("---")

            # Table with a red-yellow-green gradient for probabilities
            styled = df_res.style.format({"Inhibition Prob": "{:.4f}"}).background_gradient(
                subset=["Inhibition Prob"], cmap="RdYlGn"
            )
            st.dataframe(styled, use_container_width=True)

            st.download_button(
                "Export results (CSV)",
                data=df_res.to_csv(index=False).encode("utf-8"),
                file_name="NeuroBACE_Report.csv",
                mime="text/csv",
            )

with t2:
    if "results" not in st.session_state:
        st.info("Run screening first to see analytics.")
    else:
        df = st.session_state["results"].copy()
        valid = df[df["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        if valid.empty:
            st.warning("No valid predictions available for analytics.")
        else:
            st.markdown("### Probability distribution")
            fig_hist = px.histogram(
                valid,
                x="Inhibition Prob",
                nbins=30,
                template=plotly_template,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Predicted probability"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Top predicted molecules")
            top_n = min(25, len(valid))
            top = valid.sort_values("Inhibition Prob", ascending=False).head(top_n)

            fig_bar = px.bar(
                top.sort_values("Inhibition Prob", ascending=True),
                y="Compound Name",
                x="Inhibition Prob",
                orientation="h",
                template=plotly_template,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Predicted probability"},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### ACTIVE calls (filtered)")
            hits = valid[valid["Result"] == "ACTIVE"].sort_values("Inhibition Prob", ascending=False)
            st.dataframe(hits, use_container_width=True)

with t3:
    st.write("### Platform architecture")
    st.markdown(
        """
- Model: XGBoost binary classifier (native model format preferred)
- Featurization: Morgan fingerprints, 2048-bit, radius 2
- Output: predicted probability for BACE1 inhibition class
- Optional name resolution: PubChem PUG REST (Title property)

Notes
- The probability is a machine learning score, not an experimental potency value.
- If your input molecule is outside the training chemical space, the score may be less reliable.
        """
    )
