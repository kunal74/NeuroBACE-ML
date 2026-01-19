import os
import pickle
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import xgboost as xgb

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="NeuroBACE-ML", page_icon="🧠", layout="wide")

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
    h1,h2,h3,h4,label,span,p,[data-testid="stWidgetLabel"] p,.stMarkdown p {{ color: {text} !important; opacity: 1 !important; }}
    [data-testid="stMetric"] {{ background-color: {card} !important; border: 1px solid {accent}44 !important; border-radius: 12px; }}
    [data-testid="stMetricValue"] div {{ color: {accent} !important; font-weight: bold; }}
    .stButton>button {{ background: linear-gradient(90deg, #0ea5e9, #2563eb) !important; color: white !important; font-weight: bold !important; border-radius: 8px !important; }}
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 NeuroBACE-ML")
st.markdown("##### Advanced platform for BACE1 inhibitor prediction")
st.write("---")


# =========================
# Paths and constants
# =========================
MODEL_JSON = "BACE1_trained_model_optimized.json"
MODEL_PKL  = "BACE1_trained_model_optimized.pkl"
FP_BITS = 2048

def here_path(filename: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, filename)


# =========================
# Model loading (JSON Booster preferred, PKL fallback)
# =========================
@st.cache_resource
def load_model_bundle():
    json_path = here_path(MODEL_JSON)
    pkl_path  = here_path(MODEL_PKL)

    if os.path.exists(json_path):
        booster = xgb.Booster()
        booster.load_model(json_path)
        return {"kind": "booster", "model": booster}

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        return {"kind": "pkl", "model": obj}

    return None

model_bundle = load_model_bundle()
if model_bundle is None:
    st.error(f"Model not found. Upload {MODEL_JSON} (recommended) to the same folder as app.py in GitHub.")
    st.stop()


def predict_active_prob(model_bundle, X: np.ndarray) -> float:
    kind = model_bundle["kind"]
    model = model_bundle["model"]

    if kind == "booster":
        dmat = xgb.DMatrix(X)
        p = model.predict(dmat)
        return float(p[0])

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0][1])

    p = model.predict(X)
    return float(p[0])


# =========================
# RDKit featurization
# =========================
def featurize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1), ""


def guess_smiles_column(cols):
    cols = list(cols)
    for c in cols:
        if str(c).strip().lower() in {"smiles", "smile"}:
            return c
    for c in cols:
        if "smiles" in str(c).strip().lower():
            return c
    return cols[0] if cols else None

# --- UTILITY: NAME RECOGNITION ---
def get_compound_name(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/Title/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()['PropertyTable']['Properties'][0].get('Title', "Unknown")
        return "Unknown Ligand"
    except:
        return "Novel Molecule"

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.markdown("---")
    threshold = st.slider("Sensitivity Threshold (ACTIVE if ≥ threshold)", 0.0, 1.0, 0.70, 0.01)

    st.subheader("Compound naming (optional)")
    enable_pubchem = st.checkbox("Enable PubChem naming", value=False)
    resolve_scope  = st.radio("Resolve names for", ["Top hits only", "All molecules"], index=0)
    top_n = st.slider("Top N (if Top hits only)", 5, 200, 25, 5)

    if st.button("Test PubChem connectivity"):
        ok = pubchem_connectivity_test()
        st.success("PubChem reachable from Streamlit server.") if ok else st.error("PubChem NOT reachable from Streamlit server.")

    if st.button("Clear PubChem cache"):
        st.cache_data.clear()
        st.success("Cleared PubChem cache.")


# =========================
# Tabs
# =========================
t1, t2, t3 = st.tabs(["🚀 Screening Engine", "📈 Visual Analytics", "🔬 Specifications"])

with t1:
    in_type = st.radio("Input Source", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
    mols = []

    if in_type == "Manual Entry":
        raw = st.text_area("SMILES (one per line):", "CC(=O)NC1=CC=C(C=C1)O", height=140)
        mols = [s.strip() for s in raw.split("\n") if s.strip()]
    else:
        f = st.file_uploader("Upload CSV", type=["csv"])
        if f:
            df_in = pd.read_csv(f)
            guess = guess_smiles_column(df_in.columns)
            cols = list(df_in.columns)
            idx = cols.index(guess) if guess in cols else 0
            col = st.selectbox("Select the SMILES column", options=cols, index=idx)
            mols = df_in[col].astype(str).tolist()

    c1, c2 = st.columns([1, 1])
    run_btn = c1.button("Start Virtual Screening")
    clear_btn = c2.button("Clear Results")

    if clear_btn:
        st.session_state.pop("results", None)
        st.success("Results cleared.")

    if run_btn:
        if not mols:
            st.warning("No SMILES provided.")
        else:
            with st.spinner("Running predictions..."):
                res = []
                prog = st.progress(0)

                for i, s in enumerate(mols):
                    X, err = featurize(s)
                    if err:
                        res.append({
                            "Compound Name": "Unknown",
                            "SMILES": s,
                            "Inhibition Prob": np.nan,
                            "Result": "INVALID",
                            "PubChem CID": None,
                            "Name Source": "None",
                        })
                    else:
                        p = predict_active_prob(model_bundle, X)
                        res.append({
                            "Compound Name": "Unknown",
                            "SMILES": s,
                            "Inhibition Prob": round(float(p), 4),
                            "Result": "ACTIVE" if p >= threshold else "INACTIVE",
                            "PubChem CID": None,
                            "Name Source": "None",
                        })

                    prog.progress((i + 1) / max(1, len(mols)))

                st.session_state["results"] = pd.DataFrame(res)

    # Always show results if present
    if "results" in st.session_state:
        df_res = st.session_state["results"].copy()
        valid = df_res[df_res["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Molecules", len(df_res))
        m2.metric("Valid predictions", len(valid))
        m3.metric("ACTIVE calls", int((valid["Result"] == "ACTIVE").sum()) if len(valid) else 0)
        m4.metric("Max Probability", f"{valid['Inhibition Prob'].max():.3f}" if len(valid) else "NA")

        st.write("---")
        st.dataframe(df_res.style.background_gradient(subset=["Inhibition Prob"], cmap="RdYlGn"), use_container_width=True)
        st.download_button("Export Results", df_res.to_csv(index=False), "NeuroBACE_Report.csv")

        # Optional PubChem naming (non-blocking)
        if enable_pubchem:
            if st.button("Resolve Names (PubChem)"):
                with st.spinner("Resolving names from PubChem (short timeouts)..."):
                    df_work = df_res.copy()
                    df_valid = df_work[df_work["Result"].isin(["ACTIVE", "INACTIVE"])].copy()

                    if resolve_scope == "Top hits only":
                        target = df_valid.sort_values("Inhibition Prob", ascending=False).head(top_n)
                        idxs = list(target.index)
                    else:
                        idxs = list(df_valid.index)

                    p2 = st.progress(0)
                    for j, idx in enumerate(idxs):
                        smi = df_work.at[idx, "SMILES"]
                        name, cid, src = resolve_name_recordtitle_then_iupac(smi)
                        df_work.at[idx, "Compound Name"] = name
                        df_work.at[idx, "PubChem CID"] = cid
                        df_work.at[idx, "Name Source"] = src
                        p2.progress((j + 1) / max(1, len(idxs)))

                    st.session_state["results"] = df_work
                    st.success("Name resolution complete. Table updated.")

with t2:
    if "results" in st.session_state:
        data = st.session_state["results"].copy()
        data_valid = data[data["Result"].isin(["ACTIVE", "INACTIVE"])].dropna(subset=["Inhibition Prob"]).copy()

        st.markdown("### Predictive Probability Distribution")
        if data_valid.empty:
            st.warning("No valid predictions available.")
        else:
            fig_h = px.histogram(
                data_valid,
                x="Inhibition Prob",
                nbins=30,
                template=plotly_temp,
                range_x=[0, 1],
                labels={"Inhibition Prob": "Probability Score"},
            )
            st.plotly_chart(fig_h, use_container_width=True)

            data_sorted = data_valid.sort_values("Inhibition Prob", ascending=True).tail(25)
            fig = px.bar(
                data_sorted,
                y="Compound Name",
                x="Inhibition Prob",
                orientation="h",
                color="Inhibition Prob",
                color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                template=plotly_temp,
                labels={"Inhibition Prob": "Probability Score"},
                height=max(400, len(data_sorted) * 30),
            )
            fig.update_layout(xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run screening first to see analytics.")

with t3:
    st.write("### Platform Architecture")
    st.markdown(
        """
- **Model formats supported:** XGBoost native JSON (`Booster.load_model`) or pickled sklearn-style model  
- **Prediction:** JSON Booster uses `predict(DMatrix)`; PKL uses `predict_proba` (if available)  
- **Feature Extraction:** 2048-bit Morgan fingerprints (radius = 2)  
- **PubChem naming (optional):** SMILES → CID (PUG-REST) → RecordTitle (PUG-View) → fallback IUPACName (PUG-REST)  
"""
    )
